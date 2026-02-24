import torch

from texture_prior.utils import projection
from texture_prior.utils import audio

def synthesize_from_audio(target_texture, model, nPCs=100, 
                          niters=5000, nsteps=10, 
                          lr=(1e-6)*128, lr_decay=0.5, 
                          disp_every=250, snr_thresh=40):
    device = target_texture.device
    stat_projector = projection.statistics_projector(device)    
    synth_texture = audio.spectrally_matched_noise(torch.squeeze(target_texture))[None, :]
    synth_texture.requires_grad=True
    losses = torch.zeros(niters)
    criterion = torch.nn.MSELoss()    
    optimizer = torch.optim.Adam([synth_texture], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=niters/nsteps, gamma=lr_decay)    
    target_stats = stat_projector.project(model(target_texture), nPCs=nPCs)
    for i in range(niters):
        synth_stats = stat_projector.project(model(synth_texture), nPCs=nPCs)
        loss = criterion(synth_stats, target_stats)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses[i] = loss.item()
        scheduler.step()
        if torch.isnan(synth_texture.grad).sum().item():
            print('NaN found in gradient', flush=True)
            break
        if (i)%disp_every==0:
            print('iteration: ', i, flush=True)
            print('loss: ', loss.item(), flush=True)
            print('lr: ', optimizer.param_groups[0]['lr'], flush=True)
            print(' ', flush=True)
        SNRs = compute_SNR(synth_stats.detach(), target_stats)
        if SNRs.min() > snr_thresh:
            # all SNRs are above minimum threshold for convergence so stop the synthesis early
            break
    print('final', flush=True)
    print('loss: ', loss.item(), flush=True)
    print('lr: ', optimizer.param_groups[0]['lr'], flush=True)
    print(' ', flush=True)
    SNRs = compute_SNR(synth_stats.detach(), target_stats)
    return synth_texture.detach(), losses, target_stats, synth_stats.detach(), SNRs

def synthesize_from_statistics(target_stats, model, lr, niters, disp_every):
    #TODO: synthesize directly from statistics
    # should initialize with white noise? 
    # will likely need to implement after incorporating variable signal lengths into model
    # see issue #5: https://github.mit.edu/jmhicks/texture-prior/issues/5
    pass

def compute_SNR(synth_stats, target_stats):
    stat_error = ((target_stats - synth_stats) ** 2)
    SNRs = 10 * torch.log10((target_stats ** 2) / stat_error)
    return SNRs

def compute_SNR_per_class(synth_stat_dict, target_stat_dict):
    #TODO: implement function to compute SNR for each class of statistics
    # should take in stat dict
    pass