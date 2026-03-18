import torch
import functools
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import tqdm
from models import ScoreNetAudio, ScoreNetTexture1D, ScoreNetAudioV2
from utils.sde_utils import *

import matplotlib.pyplot as plt
import argparse, yaml, os
from types import SimpleNamespace
from dataloader import TextureStatsDataset
import seaborn as sns
import pandas as pd

# for audio synthesis
from chexture_choolbox.auditorytexture.texture_model import TextureModel
from utils.params import model_params
from utils.params import statistics_set
from utils import synthesis, path, projection, audio
import torchaudio
from joblib import Parallel, delayed

from chexture_choolbox.auditorytexture.helpers import FlattenStats

ex_audio, ex_sr, ex_coch_params, ex_mod_params, ex_octmod_params = audio.load('assets/008_animals_ducks-feeding_f0.wav')

def parse(d):
  x = SimpleNamespace()
  _ = [setattr(x, k, parse(v)) if isinstance(v, dict) else setattr(x, k, v) for k, v in d.items() ]    
  return x

def compute_SNR(synth_stats, target_stats):
    stat_error = ((target_stats - synth_stats) ** 2)
    SNRs = 10 * torch.log10((target_stats ** 2) / stat_error)
    return SNRs

def train_model(score_model, cfg, ckpt_path, mode):
    # set up the optimizer
    optimizer = Adam(score_model.parameters(), lr=cfg.train.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.n_epochs)

    train_dataset = TextureStatsDataset(
                            config=cfg,
                            device=device,
                            mode=mode
                        )

    criteria = loss_fn1d if cfg.model.use_single_dim_conv else loss_fn
    # construct the dataloader
    data_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)

    # training loop
    for epoch in tqdm.tqdm(range(cfg.train.n_epochs)):
        avg_loss, num_items = 0., 0
        for x in tqdm.tqdm(data_loader):

            x = x.to(device)
            loss = criteria(score_model, x, marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()    
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]

        # print the averaged training loss so far.
        print('Average Loss: {:5f}'.format(avg_loss / num_items))
        scheduler.step()
        # update the checkpoint after each epoch of training.
        torch.save(score_model.state_dict(), ckpt_path)

def sample_from_model(score_model, cfg, ckpt, mode):
    """Sampling from the score-based model
    """
    device = 'cuda'
    score_model.load_state_dict(torch.load(ckpt))

    texture_dataset = TextureStatsDataset(
                            config=cfg,
                            device=device,
                            mode=mode
                        )
    x_orig = texture_dataset.data.squeeze().detach().cpu().numpy()
    idx = np.random.random_integers(0, x_orig.shape[0], cfg.sample.sample_batch_size)
    x_orig = x_orig[idx]

    sample_batch_size = cfg.sample.sample_batch_size
    sampler = Euler_Maruyama_sampler_1d if cfg.model.use_single_dim_conv else Euler_Maruyama_sampler
    init_dims = [1, cfg.data.n_pcs] if cfg.model.use_single_dim_conv else [1, 1, cfg.data.n_pcs]

    # generate samples using the specified sampler.
    samples, trajectory = sampler(
                            score_model,
                            marginal_prob_std_fn,
                            diffusion_coeff_fn,
                            num_steps=cfg.sample.num_steps,
                            batch_size=sample_batch_size,
                            device=device,
                            init_dims=init_dims)
    
    samples = samples.squeeze().detach().cpu().numpy()

    if cfg.data.var_scale:
        samples = samples * texture_dataset.per_feature_std.squeeze()[None, :].numpy()
        samples = samples * texture_dataset.scale

    # dump samples to a npy file
    fname = os.path.join(
        'figures',
        'diffusion_{}_stats_pc_{}_samples.npy'.format(mode, cfg.data.n_pcs)
    )
    np.save(fname, samples)

    n_features = 64
    ncols = 16
    nrows = n_features // ncols
    figsize = (ncols * 1.2, nrows * 0.8)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    for i in range(n_features):
        ax = axes[i]
        sns.kdeplot(x_orig[:, i], ax=ax, label='real', color='blue', lw=1, bw_adjust=0.5)
        sns.kdeplot(samples[:, i], ax=ax, label='diffusion', color='red', lw=1, bw_adjust=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel('')

        ax.set_title(f'{i}', fontsize=8)
        ax.set_frame_on(False)

    ax.legend(fontsize=12)
    # Hide any unused subplots (in case n_features < nrows*ncols)
    for ax in axes[n_features:]:
        ax.axis('off')

    fig.suptitle('Marginal KDEs', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fname = os.path.join(
        'figures',
        'diffusion_{}_stats_pc_{}.png'.format(mode, cfg.data.n_pcs)
    )
    plt.savefig(fname, bbox_inches='tight', dpi=300)

    # first n PCs
    n_pairwise_dims = 5
    df_real = pd.DataFrame(x_orig[:, :n_pairwise_dims], columns=[f'PC{i+1}' for i in range(n_pairwise_dims)])
    df_real['source'] = 'real'

    df_fake = pd.DataFrame(samples[:, :n_pairwise_dims], columns=[f'PC{i+1}' for i in range(n_pairwise_dims)])
    df_fake['source'] = 'diffusion'

    df_all = pd.concat([df_real, df_fake], ignore_index=True)

    sns.pairplot(df_all, hue='source', kind='kde', palette={'real': 'blue', 'diffusion': 'red'}, diag_kind='kde')
    plt.suptitle("Pairwise KDEs", y=1.02, fontsize=16)
    fname = os.path.join(
        'figures',
        'diffusion_{}_pairwise_stats_pc_{}.png'.format(mode, cfg.data.n_pcs)
    )
    plt.savefig(fname, bbox_inches='tight', dpi=300)


    # last n PCs
    n_pairwise_dims = 5
    df_real = pd.DataFrame(x_orig[:, -n_pairwise_dims:], columns=[f'PC{i+1}' for i in range(cfg.data.n_pcs - n_pairwise_dims, cfg.data.n_pcs)])
    df_real['source'] = 'real'

    df_fake = pd.DataFrame(samples[:, -n_pairwise_dims:], columns=[f'PC{i+1}' for i in range(cfg.data.n_pcs - n_pairwise_dims, cfg.data.n_pcs)])
    df_fake['source'] = 'diffusion'

    df_all = pd.concat([df_real, df_fake], ignore_index=True)

    sns.pairplot(df_all, hue='source', kind='kde', palette={'real': 'blue', 'diffusion': 'red'}, diag_kind='kde')
    plt.suptitle("Pairwise KDEs", y=1.02, fontsize=16)
    fname = os.path.join(
        'figures',
        'diffusion_{}_pairwise_stats_tail_pc_{}.png'.format(mode, cfg.data.n_pcs)
    )
    plt.savefig(fname, bbox_inches='tight', dpi=300)

def compare_mixtures_to_textures(score_model, cfg, ckpt, mode):

    device = 'cuda'
    score_model.load_state_dict(torch.load(ckpt, weights_only=True))

    from eval_statistics import load_test_data
    mix_dataset = load_test_data(
                data_path='/orcd/data/jhm/001/om2/lakshmin/audio-prior/assets/exp23_mixture_statistics_4096texturePCs.pt',
                config=cfg, 
                device=device
            )
    mix_dataset = mix_dataset.transpose(0, 2)

    texture_dataset = TextureStatsDataset(
                            config=cfg,
                            device=device,
                            mode=mode
                        )
    x_orig = texture_dataset.data.squeeze().detach().cpu().numpy()
    idx = np.random.random_integers(0, x_orig.shape[0], cfg.sample.sample_batch_size)
    x_orig = x_orig[idx]

    sample_batch_size = cfg.sample.sample_batch_size
    sampler = Euler_Maruyama_sampler_1d if cfg.model.use_single_dim_conv else Euler_Maruyama_sampler
    init_dims = [1, cfg.data.n_pcs] if cfg.model.use_single_dim_conv else [1, 1, cfg.data.n_pcs]

    n_features = 64
    ncols = 16
    nrows = n_features // ncols
    figsize = (ncols * 1.2, nrows * 0.8)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    for i in range(n_features):
        ax = axes[i]
        sns.kdeplot(x_orig[:, i], ax=ax, label='texture_stats', color='blue', lw=1, bw_adjust=0.5)
        sns.kdeplot(mix_dataset[..., i].squeeze().detach().cpu().numpy(), ax=ax, label='mixture_stats', color='red', lw=1, bw_adjust=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel('')

        ax.set_title(f'{i}', fontsize=8)
        ax.set_frame_on(False)

    ax.legend(fontsize=12)
    # Hide any unused subplots (in case n_features < nrows*ncols)
    for ax in axes[n_features:]:
        ax.axis('off')

    fig.suptitle('Marginal KDEs', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fname = os.path.join(
        'figures',
        'compare_{}_stats_pc_{}.png'.format(mode, cfg.data.n_pcs)
    )
    plt.savefig(fname, bbox_inches='tight', dpi=300)

    import ipdb; ipdb.set_trace()

    # # first n PCs
    # n_pairwise_dims = 5
    # df_real = pd.DataFrame(x_orig[:, :n_pairwise_dims], columns=[f'PC{i+1}' for i in range(n_pairwise_dims)])
    # df_real['source'] = 'real'

    # # df_fake = pd.DataFrame(samples[:, :n_pairwise_dims], columns=[f'PC{i+1}' for i in range(n_pairwise_dims)])
    # # df_fake['source'] = 'diffusion'

    # df_all = pd.concat([df_real, df_fake], ignore_index=True)

    # sns.pairplot(df_all, hue='source', kind='kde', palette={'real': 'blue', 'diffusion': 'red'}, diag_kind='kde')
    # plt.suptitle("Pairwise KDEs", y=1.02, fontsize=16)
    # fname = os.path.join(
    #     'figures',
    #     'diffusion_{}_pairwise_stats_pc_{}.png'.format(mode, cfg.data.n_pcs)
    # )
    # plt.savefig(fname, bbox_inches='tight', dpi=300)


    # # last n PCs
    # n_pairwise_dims = 5
    # df_real = pd.DataFrame(x_orig[:, -n_pairwise_dims:], columns=[f'PC{i+1}' for i in range(cfg.data.n_pcs - n_pairwise_dims, cfg.data.n_pcs)])
    # df_real['source'] = 'real'

    # # df_fake = pd.DataFrame(samples[:, -n_pairwise_dims:], columns=[f'PC{i+1}' for i in range(cfg.data.n_pcs - n_pairwise_dims, cfg.data.n_pcs)])
    # # df_fake['source'] = 'diffusion'

    # df_all = pd.concat([df_real, df_fake], ignore_index=True)

    # sns.pairplot(df_all, hue='source', kind='kde', palette={'real': 'blue', 'diffusion': 'red'}, diag_kind='kde')
    # plt.suptitle("Pairwise KDEs", y=1.02, fontsize=16)
    # fname = os.path.join(
    #     'figures',
    #     'diffusion_{}_pairwise_stats_tail_pc_{}.png'.format(mode, cfg.data.n_pcs)
    # )
    # plt.savefig(fname, bbox_inches='tight', dpi=300)

def compute_likelihood(score_model, input_stats, ckpt):
    """Computing the actual "prior" value
    If the score function can be treated as the vector field govering the temporal evolution
    of $x_t$ then we can integrate the ODE and apply a change of basis to evaluate the prior
    """
    score_model.load_state_dict(torch.load(ckpt))
    input_stats = input_stats.to(device)

    _, bpd = ode_likelihood(input_stats, score_model, marginal_prob_std_fn,
                            diffusion_coeff_fn,
                            input_stats.shape[0], device=device, eps=1e-5)
    
    return bpd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='model + data configuration')
    parser.add_argument('--train', action='store_true', help='train the score-based model')
    parser.add_argument('--sample', action='store_true', help='sample from a trained model')
    parser.add_argument('--likelihood_eval', action='store_true', help='evaluate the data likelihood')
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--mode', type=str, default='textures', choices=['textures', 'mixtures'])

    args = parser.parse_args()

    df = yaml.safe_load(open(args.config))
    cfg = parse(df)
    
    score_model = torch.nn.DataParallel(
                        ScoreNetAudioV2(
                            marginal_prob_std=marginal_prob_std_fn, 
                            channels=cfg.model.channels, 
                            embed_dim=cfg.model.embed_dim
                            )
                        )

    score_model = score_model.to(device)

    ckpt_path = cfg.model.ckpt_path.format(cfg.data.n_pcs, args.mode)
    if 'SLURM_RESTART_COUNT' in os.environ.keys() or args.restart:
        score_model.load_state_dict(torch.load(ckpt_path, weights_only=True))

    if args.train:
        score_model.train()
        train_model(score_model, cfg, ckpt_path, mode=args.mode)
    elif args.sample:
        # sample_from_model(score_model, cfg, ckpt_path, args.mode)
        compare_mixtures_to_textures(score_model, cfg, ckpt_path, args.mode)
    elif args.likelihood_eval:
        texture_dataset = TextureStatsDataset(
                        config=cfg,                        
                        device=device
                    )
        x = texture_dataset.get_random_batch(1024)
        bpd_textures = compute_likelihood(score_model, input_stats=x, ckpt=ckpt_path)
        bpd_textures = bpd_textures * (cfg.data.n_pcs * np.log(2))

        mixture_dataset = TextureStatsDataset(
                        config=cfg,                        
                        device=device,
                        mode='mixtures'
                    )
        y = mixture_dataset.get_random_batch(1024)
        bpd_mixtures = compute_likelihood(score_model, input_stats=y, ckpt=ckpt_path)
        bpd_mixtures = bpd_mixtures * (cfg.data.n_pcs * np.log(2))

        plt.figure(figsize=(8, 5))

        sns.histplot(bpd_textures.detach().cpu().numpy(), bins=50, color='tab:brown', label='Textures', stat='density', alpha=0.5, kde=True, element='step')
        sns.histplot(bpd_mixtures.detach().cpu().numpy(), bins=50, color='tab:gray', label='Mixtures', stat='density', alpha=0.5, kde=True, element='step')

        plt.xlabel(r'$-$log$p(x)$')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(
            'figures',
            'likelihood_comparison_pc_{}.png'.format(cfg.data.n_pcs)
        )
        plt.savefig(fname, bbox_inches='tight', dpi=200)

    else:
        raise NotImplementedError
