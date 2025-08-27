import torch
import torch.nn as nn
from scipy.io import wavfile
from scipy import signal
import numpy as np
import glob
import sys

sys.path.append('/om2/user/bjmedina/')

from chexture_choolbox.auditorytexture.statistics_sets import (
    STAT_SET_FULL_MCDERMOTTSIMONCELLI as statistics_dict
)
from chexture_choolbox.auditorytexture.texture_model import TextureModel
from chexture_choolbox.auditorytexture.helpers import FlattenStats
from texture_prior.params import model_params

from sklearn.decomposition import PCA


class AudioTextureEncoder(nn.Module):
    def __init__(self, statistics_dict, model_params, sr=20000, rms_level=0.05, duration=2.0, device='cuda'):
        super().__init__()
        self.sr = sr
        self.rms_level = rms_level
        self.duration = duration
        self.device = device

        self.flatten_stat_dict = FlattenStats(statistics_dict)
        coch_params, mod_params, octmod_params = model_params.update_param_dicts(duration)
        coch_params['rep_kwargs']['coch_filter_kwargs']['n'] = 30
        self.texture_model = TextureModel(coch_params, mod_params, octmod_params,
                                          statistics_dict=statistics_dict).to(device)

    def forward(self, filepath):
        """
        Accepts a string path to a .wav file, trims or drops based on length,
        returns a flattened texture representation tensor.
        """
        try:
            target_len = int(self.sr * self.duration)  # expected length in samples
            original_sr, sound = wavfile.read(filepath)
    
            if sound.ndim > 1:
                sound = sound.mean(axis=1)
    
            # Resample
            sound = signal.resample_poly(sound, self.sr, original_sr, axis=0)
    
            # Check length
            if len(sound) < target_len:
                # Drop short sounds
                raise ValueError(f"Sound too short after resampling: {len(sound)} < {target_len}")
            elif len(sound) > target_len:
                # Random crop
                start = np.random.randint(0, len(sound) - target_len + 1)
                sound = sound[start:start + target_len]
    
            # Normalize to zero mean and target RMS
            sound = sound - np.mean(sound)
            rms = np.sqrt(np.mean(np.square(sound)))
            sound = self.rms_level * sound / (rms + 1e-6)
    
            # Prepare tensor
            sound_tensor = torch.from_numpy(sound).float().unsqueeze(0).unsqueeze(0).to(self.device)
    
            with torch.no_grad():
                stats_dict = self.texture_model(sound_tensor)
                stats = self.flatten_stat_dict(stats_dict).squeeze(0)
            return stats
    
        except Exception as e:
            print(f"Skipping {filepath}: {e}")
            return None

class PCASpace(nn.Module):
    def __init__(self, encoder, n_components=2, whiten=False, device='cpu'):
        super().__init__()
        self.encoder = encoder
        self.n_components = n_components
        self.device = device
        self.pca = None
        self.whiten = whiten

    def fit(self, filepaths):
        """Fit PCA on a list of audio filepaths."""
        reps = []
        for path in filepaths:
            
            try:
                with torch.no_grad():
                    rep = self.encoder(path).detach().cpu().numpy()
                reps.append(rep)
            except AttributeError:
                continue
        
        X = np.vstack(reps)
        self.pca = PCA(n_components=self.n_components, whiten=self.whiten)
        self.pca.fit(X)

    def transform(self, filepaths):
        """Project new filepaths into PCA space."""
        assert self.pca is not None, "You must call .fit() first"
        reps = []
        for path in filepaths:
            rep = self.encoder(path).detach().cpu().numpy()
            reps.append(rep)
        return self.pca.transform(np.vstack(reps))

    def fit_transform(self, filepaths):
        """Convenience method to fit and then project."""
        self.fit(filepaths)
        return self.transform(filepaths)

    def forward(self, filepath):
        """Encode + project a single file."""
        assert self.pca is not None, "Call fit() first."
        rep = self.encoder(filepath).detach().cpu().numpy().reshape(1, -1)
        proj = self.pca.transform(rep)
        return torch.tensor(proj, dtype=torch.float32).squeeze(0).to(self.device)

class ZScoreSpace(nn.Module):
    def __init__(self, encoder, device='cpu', eps=1e-6):
        super().__init__()
        self.encoder = encoder
        self.device = device
        self.eps = eps
        self.mean = None
        self.std = None

    def fit(self, filepaths):
        """
        Fit the z-scoring parameters (mean and std) using a list of audio filepaths.
        """
        reps = []
        for path in filepaths:
            rep = self.encoder(path)
            if rep is not None:
                reps.append(rep.detach().cpu().numpy())

        X = np.vstack(reps)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

    def transform(self, filepaths):
        """
        Z-score transform a list of filepaths.
        """
        assert self.mean is not None and self.std is not None, "You must call .fit() first"
        reps = []
        for path in filepaths:
            rep = self.encoder(path)
            if rep is not None:
                rep = rep.detach().cpu().numpy()
                z = (rep - self.mean) / (self.std + self.eps)
                reps.append(z)
        return np.vstack(reps)

    def forward(self, filepath):
        """
        Encode and z-score a single file.
        """
        assert self.mean is not None and self.std is not None, "Call fit() first."
        rep = self.encoder(filepath)
        if rep is None:
            return None
        rep = rep.detach().cpu().numpy()
        z = (rep - self.mean) / (self.std + self.eps)
        return torch.tensor(z, dtype=torch.float32).to(self.device)


if __name__ == "__main__":
    sounds_list = glob.glob("/mindhive/mcdermott/www/mturk_stimuli/bjmedina/mem_exp_atexts_p1/*wav")
    texture_list = sounds_list
    print(sounds_list)
    
    ALL_SOUNDS = glob.glob("/om2/data/public/audioset/wavs/unbalanced_train_segments_downloads/unbalanced_train_segments_downloads_*/*wav")
