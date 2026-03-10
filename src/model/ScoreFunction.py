import sys
import torch
import functools
import matplotlib.pyplot as plt
import argparse, yaml, os
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
import seaborn as sns
import pandas as pd
import tqdm
import glob

from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from types import SimpleNamespace

from IPython.display import clear_output

sys.path.append('/om2/user/jmhicks/projects/TextureStreaming/code/')

from chexture_choolbox.auditorytexture.statistics_sets import (
    STAT_SET_FULL_MCDERMOTTSIMONCELLI as statistics_dict
)
from chexture_choolbox.auditorytexture.texture_model import TextureModel
from chexture_choolbox.auditorytexture.helpers import FlattenStats
from texture_prior.params import model_params as model_params_tm
from texture_prior.params import statistics_set, texture_dataset
from texture_prior.utils import path


import sys
import importlib.util
import os

# Add the new path
audio_prior_path = '/om2/user/lakshmin/audio-prior/'
sys.path.insert(0, audio_prior_path)  # insert at front of sys.path

from models import ScoreNetAudio, ScoreNetTexture1D, ScoreNetAudioV2
from utils import synthesis, projection, audio
from utils.sde_utils import *


def parse(d):
  x = SimpleNamespace()
  _ = [setattr(x, k, parse(v)) if isinstance(v, dict) else setattr(x, k, v) for k, v in d.items() ]    
  return x


class ScoreFunction():
    def __init__(self, 
                 mode = 'textures',
                 restart = False,
                 likelihood_eval = True,
                 sample = False,
                 train = False,
                 config="/om2/user/bjmedina/auditory-memory/memory/assets/bryan.yaml", 
                 device='cpu'):


        self.mode = mode
        self.restart = restart
        self.likelihood_eval = likelihood_eval
        self.sample = sample
        self.train = train
        self.device = device
        self.config = config

        df = yaml.safe_load(open(self.config))

        self.cfg = parse(df)

        self.score_model = torch.nn.DataParallel(
                    ScoreNetAudioV2(
                        marginal_prob_std=marginal_prob_std_fn, 
                        channels=self.cfg.model.channels, 
                        embed_dim=self.cfg.model.embed_dim
                        )
                    )

        self.score_model = self.score_model.to(self.device)
        
        self.ckpt_path = "/om2/user/lakshmin/audio-prior/" + self.cfg.model.ckpt_path.format(self.cfg.data.n_pcs, self.mode)
        if 'SLURM_RESTART_COUNT' in os.environ.keys() or self.restart:
            self.score_model.load_state_dict(torch.load(self.ckpt_path))

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return unit-norm score direction with same layout as the model output."""
        # Ensure tensor on the right device/dtype
        x = x.to(self.device)

        # Accept [D], [B, D], or already [B, 1, 1, D]
        if x.dim() == 1:
            x4 = x.reshape(1, 1, 1, x.shape[0])
        elif x.dim() == 2:
            B, D = x.shape
            x4 = x.reshape(B, 1, 1, D)
        elif x.dim() == 4:
            x4 = x
        else:
            raise ValueError(f"Unsupported x shape {tuple(x.shape)}; expected [D], [B,D], or [B,1,1,D].")

        B = x4.shape[0]
        # Make t match batch, and live on x's device/dtype
        t = torch.full((B,), 0.01, device=x4.device, dtype=x4.dtype)

        # Inference path; remove no_grad() if you need gradients through score
        with torch.no_grad():
            score = self.score_model(x4, t)  # shape: [B, 1, 1, D] (assumed)

        # Flatten per-sample, normalize to unit L2, then reshape back
        score_flat = score.reshape(B, -1)                       # [B, D]
        norms = score_flat.norm(p=2, dim=1, keepdim=True) + 1e-8
        unit_score_flat = score_flat / norms
        unit_score = unit_score_flat.reshape_as(score)          # [B, 1, 1, D]
        return unit_score


        