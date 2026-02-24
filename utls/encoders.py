import torch
import torch.nn as nn
from scipy.io import wavfile
from scipy import signal
import numpy as np
import glob
import sys
import os
import tempfile
from sklearn.decomposition import PCA

# sys.path.append('/om2/user/bjmedina/')

# from chexture_choolbox.auditorytexture.statistics_sets import (
#     STAT_SET_FULL_MCDERMOTTSIMONCELLI as statistics_dict
# )
# from chexture_choolbox.auditorytexture.texture_model import TextureModel
# from chexture_choolbox.auditorytexture.helpers import FlattenStats
# from texture_prior.params import model_params

sys.path.append('/om2/user/jmhicks/projects/TextureStreaming/code/')

from chexture_choolbox.auditorytexture.texture_model import TextureModel
from chexture_choolbox.auditorytexture.helpers import FlattenStats

from texture_prior.params import model_params, statistics_set, texture_dataset
from texture_prior.utils import path, normalization


sys.path.append('/om/user/jmhicks/projects/TextureSimilarity/code/')
import texture_similarity.utils as ts


import scipy.signal
import scipy.signal.windows

if not hasattr(scipy.signal, "kaiser"):
    scipy.signal.kaiser = scipy.signal.windows.kaiser


class AudioTextureEncoder(nn.Module):
    def __init__(self, statistics_dict, model_params, sr=20000, rms_level=0.05, duration=2.0, coch_filter=50, device='cuda'):
        super().__init__()
        self.sr = sr
        self.rms_level = rms_level
        self.duration = duration
        self.device = device

        self.flatten_stat_dict = FlattenStats(statistics_dict)
        coch_params, mod_params, octmod_params = model_params.update_param_dicts(duration)
        coch_params['rep_kwargs']['coch_filter_kwargs']['n'] = coch_filter
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

class AudioTextureEncoderPCA(nn.Module):
    def __init__(self, statistics_dict, model_params, pc_dims=256, sr=20000, rms_level=0.01, duration=2.0, device='cuda'):
        super().__init__()
        self.sr = sr
        self.rms_level = rms_level
        self.duration = duration
        self.device = device

        principal_components_path = "/om2/user/jmhicks/projects/TextureStreaming/code/texture_prior/assets/principal_components.pt"
        principal_components = torch.load(principal_components_path)
        principal_components = principal_components['PCs'].cuda()
        
        self.pc_transform = principal_components
        self.pc_dims = pc_dims

        self.flatten_stat_dict = FlattenStats(statistics_dict)
        coch_params, mod_params, octmod_params = model_params.update_param_dicts(duration)
        coch_params['rep_kwargs']['coch_filter_kwargs']['n'] = 48
        self.texture_model = TextureModel(coch_params, mod_params, octmod_params,
                                          statistics_dict=statistics_dict).to(device)

    def rms_np(self, x, eps=1e-8):
        """Numerically safe RMS (matches safe_process_debug)."""
        return np.sqrt(np.mean(x ** 2)) + eps

    def forward(self, filepath):
        """
        Accepts a string path to a .wav file, trims or drops based on length,
        returns a flattened texture representation tensor.
        """
        try:
            target_len = int(self.sr * self.duration)  # expected length in samples
            # original_sr, sound = wavfile.read(filepath)
    
            # if sound.ndim > 1:
            #     sound = sound.mean(axis=1)
    
            # # Resample
            # sound = signal.resample_poly(sound, self.sr, original_sr, axis=0)
    
            # # Check length
            # if len(sound) < target_len:
            #     # Drop short sounds
            #     raise ValueError(f"Sound too short after resampling: {len(sound)} < {target_len}")
            # elif len(sound) > target_len:
            #     # Random crop
            #     start = np.random.randint(0, len(sound) - target_len + 1)
            #     sound = sound[start:start + target_len]
    
            # # # Normalize to zero mean and target RMS
            # # sound = sound - np.mean(sound)
            # # rms = np.sqrt(np.mean(np.square(sound)))
            # # sound = self.rms_level * sound / (rms + 1e-6)

            # r = self.rms_np(sound)
            # sound = self.rms_level * sound / r

            original_sr, sound = wavfile.read(filepath)
        
            # Convert to float early
            if np.issubdtype(sound.dtype, np.integer):
                sound = sound.astype(np.float32) / np.iinfo(sound.dtype).max
            else:
                sound = sound.astype(np.float32)
        
            # Resample
            sound = signal.resample_poly(sound, self.sr, original_sr, axis=0)
        
            # RMS-normalize each channel BEFORE selecting mono
            if sound.ndim > 1:
                for c in range(sound.shape[1]):
                    r = self.rms_np(sound[:, c])
                    sound[:, c] /= r
        
                # Take first channel
                sound = sound[:, 0]
        
            # Final RMS normalization
            r = self.rms_np(sound)
            sound = self.rms_level * sound / r
    
            # Prepare tensor
            sound_tensor = torch.from_numpy(sound).float().unsqueeze(0).unsqueeze(0).to(self.device)

            normalization_dict = torch.load(path.relative('../assets/normalization_dict.pt'))
            norm_func = normalization.get_normalization_function(normalization_dict, self.device)
    
            with torch.no_grad():
                stats_dict = self.texture_model(sound_tensor)
                norm_stats = norm_func(stats_dict)
                stats = self.flatten_stat_dict(norm_stats).squeeze(0)


            
            proj = (stats @ self.pc_transform)[:self.pc_dims]
            return proj
    
        except Exception as e:
            print(f"Skipping {filepath}: {e}")
            return None

class Kell2018Encoder(nn.Module):

    from sklearn.decomposition import PCA
    
    sys.path.append('/om/user/jmhicks/projects/TextureSimilarity/code/')
    import texture_similarity.utils as ts

    
    import scipy.signal
    import scipy.signal.windows
    
 
    if not hasattr(scipy.signal, "kaiser"):
        scipy.signal.kaiser = scipy.signal.windows.kaiser
    
    def __init__(self, model_name, layer, sr=20000, rms_level=0.05, duration=2.0, time_avg=False, device='cuda'):
        super().__init__()
        self.model_name = model_name
        self.layer = layer
        self.sr = sr
        self.rms_level = rms_level
        self.duration = duration
        self.device = device

        cochdnn_dir = '/om2/user/bjmedina/models/cochdnn/'
        sys.path.append(cochdnn_dir)
    
        model_dir = os.path.join(cochdnn_dir, 'model_directories', model_name)
        sys.path.append(model_dir)
    
        import build_network
    
        model, ds, all_layers = build_network.main(return_metamer_layers=True)
        self.model = model
        self.ds = ds
        self.all_layers = all_layers
        self.time_avg = time_avg

        if self.time_avg:
            self.extract_from_cochdnn = self.extract_from_cochdnn_time_avgd
        else:
            self.extract_from_cochdnn = self.extract_from_cochdnn_nontime_avgd


    def _write_temp_wav(self, y):
        import tempfile
        """Write a temp WAV file at self.sr from float signal y; returns path. Caller must remove."""
        # Convert to 32-bit float WAV for fidelity
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_path = tmp.name
        tmp.close()
        # wavfile.write expects int or float; float32 is standard for PCM-float
        wavfile.write(tmp_path, self.sr, y.astype(np.float32))
        return tmp_path
    
    def _preprocess(self, filepath):
        """
        Load wav, resample, safely RMS-normalize.
        Behavior matches safe_process_debug (sans prints).
        """
        original_sr, sound = wavfile.read(filepath)
    
        # Convert to float early
        if np.issubdtype(sound.dtype, np.integer):
            sound = sound.astype(np.float32) / np.iinfo(sound.dtype).max
        else:
            sound = sound.astype(np.float32)
    
        # Resample
        sound = signal.resample_poly(sound, self.sr, original_sr, axis=0)
    
        # RMS-normalize each channel BEFORE selecting mono
        if sound.ndim > 1:
            for c in range(sound.shape[1]):
                r = self.rms_np(sound[:, c])
                sound[:, c] /= r
    
            # Take first channel
            sound = sound[:, 0]
    
        # Final RMS normalization
        r = self.rms_np(sound)
        sound = self.rms_level * sound / r
    
        return sound

    def rms_np(self, x, eps=1e-8):
        """Numerically safe RMS (matches safe_process_debug)."""
        return np.sqrt(np.mean(x ** 2)) + eps

    def process_sound(self, filepath):
        sound = self._preprocess(filepath)
        sound = np.expand_dims(sound, 0)
        sound = torch.from_numpy(sound).float().cuda()
        return sound
    
    def extract_from_cochdnn_time_avgd(self, sound_list):
        n_sounds = len(sound_list)
    
        sound = self.process_sound(sound_list[0])
        (predictions, rep, layer_returns), orig_image = self.model(sound, with_latent=True)
        n_feature_dimensions = torch.sum(torch.tensor(layer_returns[self.layer].shape) > 1).item()
        if n_feature_dimensions > 1:
            n_features = len(layer_returns[self.layer].mean(-1).flatten().detach())
        else:
            n_features = len(layer_returns[self.layer].flatten().detach())
        features = torch.zeros(n_sounds, n_features)
    
        for i in range(n_sounds):
            sound = self.process_sound(sound_list[i])
            (predictions, rep, layer_returns), orig_image = self.model(sound, with_latent=True)
            if n_feature_dimensions > 1:
                features[i, :] = layer_returns[self.layer].mean(-1).flatten().detach() # time-average                                                                                                      
            else:
                features[i, :] = layer_returns[self.layer].flatten().detach()
    
        return features

    def extract_from_cochdnn_nontime_avgd(self, sound_list):
        n_sounds = len(sound_list)
    
        sound = self.process_sound(sound_list[0])
        (predictions, rep, layer_returns), orig_image = self.model(sound, with_latent=True)
        n_feature_dimensions = torch.sum(torch.tensor(layer_returns[self.layer].shape) > 1).item()
        n_features = len(layer_returns[self.layer].flatten().detach())
        features = torch.zeros(n_sounds, n_features)

        for i in range(n_sounds):
            sound = self.process_sound(sound_list[i])
            (predictions, rep, layer_returns), orig_image = self.model(sound, with_latent=True)

            features[i, :] = layer_returns[self.layer].flatten().detach()
    
        return features

    def forward(self, filepath):
        """
        Accepts a string path to a .wav file, trims or drops based on length,
        returns a neural-layer feature tensor on self.device.
        """
        try:
            # Preprocess to match your AudioTextureEncoder behavior
            y = self._preprocess(filepath)

            # Save to a temp wav so the TextureSimilarity API can ingest it
            tmp_path = self._write_temp_wav(y)

            try:
                # Extract features from the specified model layer
                feats = self.extract_from_cochdnn([tmp_path])

                # Normalize output shape and convert to torch
                if isinstance(feats, list):
                    if len(feats) != 1:
                        raise RuntimeError(f"Unexpected feature list length: {len(feats)} (expected 1).")
                    feats = feats[0]
                feats = np.asarray(feats)
                if feats.ndim == 2 and feats.shape[0] == 1:
                    feats = feats[0]
                if feats.ndim != 1:
                    # If API returns e.g. [D] already fine; otherwise flatten
                    feats = feats.reshape(-1)

                feat_tensor = torch.from_numpy(feats).float().to(self.device, non_blocking=True)
                return feat_tensor

            finally:
                # Always cleanup temp file
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        except Exception as e:
            print(f"Skipping {filepath}: {e}")
            return None


class ResNet50Encoder(nn.Module):

    from sklearn.decomposition import PCA
    
    sys.path.append('/om/user/jmhicks/projects/TextureSimilarity/code/')
    import texture_similarity.utils as ts

    
    import scipy.signal
    import scipy.signal.windows
    
 
    if not hasattr(scipy.signal, "kaiser"):
        scipy.signal.kaiser = scipy.signal.windows.kaiser
    
    def __init__(self, model_name, layer, sr=20000, rms_level=0.05, duration=2.0, time_avg=False, device='cuda'):
        super().__init__()
        self.model_name = model_name
        self.layer = layer
        self.sr = sr
        self.rms_level = rms_level
        self.duration = duration
        self.device = device

        cochdnn_dir = '/om2/user/bjmedina/models/cochdnn/'
        sys.path.append(cochdnn_dir)
    
        model_dir = os.path.join(cochdnn_dir, 'model_directories', model_name)
        sys.path.append(model_dir)
    
        import build_network
    
        model, ds, all_layers = build_network.main(return_metamer_layers=True)
        self.model = model
        self.ds = ds
        self.all_layers = all_layers
        self.time_avg = time_avg

        if self.time_avg:
            self.extract_from_cochdnn = self.extract_from_cochdnn_time_avgd
        else:
            self.extract_from_cochdnn = self.extract_from_cochdnn_nontime_avgd


    def _write_temp_wav(self, y):
        import tempfile
        """Write a temp WAV file at self.sr from float signal y; returns path. Caller must remove."""
        # Convert to 32-bit float WAV for fidelity
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_path = tmp.name
        tmp.close()
        # wavfile.write expects int or float; float32 is standard for PCM-float
        wavfile.write(tmp_path, self.sr, y.astype(np.float32))
        return tmp_path
    
    def _preprocess(self, filepath):
        """
        Load wav, resample, safely RMS-normalize.
        Behavior matches safe_process_debug (sans prints).
        """
        original_sr, sound = wavfile.read(filepath)
    
        # Convert to float early
        if np.issubdtype(sound.dtype, np.integer):
            sound = sound.astype(np.float32) / np.iinfo(sound.dtype).max
        else:
            sound = sound.astype(np.float32)
    
        # Resample
        sound = signal.resample_poly(sound, self.sr, original_sr, axis=0)
    
        # RMS-normalize each channel BEFORE selecting mono
        if sound.ndim > 1:
            for c in range(sound.shape[1]):
                r = self.rms_np(sound[:, c])
                sound[:, c] /= r
    
            # Take first channel
            sound = sound[:, 0]
    
        # Final RMS normalization
        r = self.rms_np(sound)
        sound = self.rms_level * sound / r
    
        return sound

    def rms_np(self, x, eps=1e-8):
        """Numerically safe RMS (matches safe_process_debug)."""
        return np.sqrt(np.mean(x ** 2)) + eps

    def process_sound(self, filepath):
        sound = self._preprocess(filepath)
        sound = np.expand_dims(sound, 0)
        sound = torch.from_numpy(sound).float().cuda()
        return sound
    
    def extract_from_cochdnn_time_avgd(self, sound_list):
        n_sounds = len(sound_list)
    
        sound = self.process_sound(sound_list[0])
        (predictions, rep, layer_returns), orig_image = self.model(sound, with_latent=True)
        n_feature_dimensions = torch.sum(torch.tensor(layer_returns[self.layer].shape) > 1).item()
        if n_feature_dimensions > 1:
            n_features = len(layer_returns[self.layer].mean(-1).flatten().detach())
        else:
            n_features = len(layer_returns[self.layer].flatten().detach())
        features = torch.zeros(n_sounds, n_features)
    
        for i in range(n_sounds):
            sound = self.process_sound(sound_list[i])
            (predictions, rep, layer_returns), orig_image = self.model(sound, with_latent=True)
            if n_feature_dimensions > 1:
                features[i, :] = layer_returns[self.layer].mean(-1).flatten().detach() # time-average                                                                                                      
            else:
                features[i, :] = layer_returns[self.layer].flatten().detach()
    
        return features

    def extract_from_cochdnn_nontime_avgd(self, sound_list):
        n_sounds = len(sound_list)
    
        sound = self.process_sound(sound_list[0])
        (predictions, rep, layer_returns), orig_image = self.model(sound, with_latent=True)
        n_feature_dimensions = torch.sum(torch.tensor(layer_returns[self.layer].shape) > 1).item()
        n_features = len(layer_returns[self.layer].flatten().detach())
        features = torch.zeros(n_sounds, n_features)

        for i in range(n_sounds):
            sound = self.process_sound(sound_list[i])
            (predictions, rep, layer_returns), orig_image = self.model(sound, with_latent=True)

            features[i, :] = layer_returns[self.layer].flatten().detach()
    
        return features

    def forward(self, filepath):
        """
        Accepts a string path to a .wav file, trims or drops based on length,
        returns a neural-layer feature tensor on self.device.
        """
        try:
            # Preprocess to match your AudioTextureEncoder behavior
            y = self._preprocess(filepath)

            # Save to a temp wav so the TextureSimilarity API can ingest it
            tmp_path = self._write_temp_wav(y)

            try:
                # Extract features from the specified model layer
                feats = self.extract_from_cochdnn([tmp_path])

                # Normalize output shape and convert to torch
                if isinstance(feats, list):
                    if len(feats) != 1:
                        raise RuntimeError(f"Unexpected feature list length: {len(feats)} (expected 1).")
                    feats = feats[0]
                feats = np.asarray(feats)
                if feats.ndim == 2 and feats.shape[0] == 1:
                    feats = feats[0]
                if feats.ndim != 1:
                    # If API returns e.g. [D] already fine; otherwise flatten
                    feats = feats.reshape(-1)

                feat_tensor = torch.from_numpy(feats).float().to(self.device, non_blocking=True)
                return feat_tensor

            finally:
                # Always cleanup temp file
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

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
