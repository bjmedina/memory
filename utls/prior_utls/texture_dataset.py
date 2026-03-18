# creates custom pytorch dataloader
# loads subset of sounds from audioset that pass stationarity screen and are thus considered textures

import torch, torchaudio, os, math
import pandas as pd

from chexture_choolbox.auditorytexture import audio_input_transforms as at
from chexture_choolbox.auditorytexture import texture_datasets as td

from texture_prior.params import model_params
    
def torch_rms(sound):
    return torch.sqrt(torch.mean(torch.square(sound)))

def set_rms(sound, rms_level):
    return rms_level * sound / torch_rms(sound)

class AudioSetTextures(torch.utils.data.Dataset):    
    def __init__(self, stationarity_screen_list, audioset_path, duration, stationarity_cutoff=-0.6, num_clips_per_sound=12, rms_level=0.01): 
        stationarity_df = pd.read_csv(stationarity_screen_list)
        texture_indices = (stationarity_df['stationarity_score'] <= stationarity_cutoff) & (stationarity_df['within_sound_ranking'] <= num_clips_per_sound)
        self.texture_list = stationarity_df[texture_indices] # list of what would be considered "textures"
        self.audioset_path = audioset_path
        self.sample_rate = model_params.coch_params['rep_kwargs']['sr']
        self.duration = duration
        self.rms_level = rms_level
    
    def __getitem__(self, index):        
        file_path = os.path.join(self.audioset_path, self.texture_list.iloc[index].filepath)
        waveform, original_sr = torchaudio.load(file_path)
        start_sample = int(original_sr * self.texture_list.iloc[index].onset_time)
        end_sample = start_sample + int(original_sr * self.duration)
        waveform = waveform[:, start_sample:end_sample]
        if waveform.shape[0] > 1: # convert to mono if stereo
            waveform = waveform.mean(dim=0, keepdim=True)
        if original_sr != self.sample_rate: # resample if at wrong sampling rate
            waveform = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=self.sample_rate)(waveform)
        waveform = set_rms(waveform, self.rms_level)
        return waveform
    
    def __len__(self):
        return len(self.texture_list)
    
class AudioSetMixtures(torch.utils.data.Dataset):    
    def __init__(self, stationarity_screen_list, audioset_path, duration, stationarity_cutoff=-0.6, num_clips_per_sound=12, rms_level=0.01): 
        stationarity_df = pd.read_csv(stationarity_screen_list)
        texture_indices = (stationarity_df['stationarity_score'] <= stationarity_cutoff) & (stationarity_df['within_sound_ranking'] <= num_clips_per_sound)
        texture_list = stationarity_df[texture_indices]
        n_half = len(texture_list) - (len(texture_list) % 2)
        mixture_inds1 = torch.randperm(n_half).view((int(n_half/2), 2))
        mixture_inds2 = torch.randperm(n_half).view((int(n_half/2), 2))
        mixture_inds = torch.vstack((mixture_inds1, mixture_inds2))

        self.mixture_inds = mixture_inds
        self.texture_list = stationarity_df[texture_indices] # list of what would be considered "textures"
        self.audioset_path = audioset_path
        self.sample_rate = model_params.coch_params['rep_kwargs']['sr']
        self.duration = duration
        self.rms_level = rms_level
    
    def __getitem__(self, index):    
        texture_ind = int(self.mixture_inds[index, 0])
        file_path = os.path.join(self.audioset_path, 
                                    self.texture_list.iloc[texture_ind].filepath)
        waveform, original_sr = torchaudio.load(file_path)
        start_sample = int(original_sr * self.texture_list.iloc[texture_ind].onset_time)
        end_sample = start_sample + int(original_sr * self.duration)
        waveform = waveform[:, start_sample:end_sample]
        if waveform.shape[0] > 1: # convert to mono if stereo
            waveform = waveform.mean(dim=0, keepdim=True)
        if original_sr != self.sample_rate: # resample if at wrong sampling rate
            waveform = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=self.sample_rate)(waveform)
        waveform1 = set_rms(waveform, self.rms_level)
        
        texture_ind = int(self.mixture_inds[index, 1])
        file_path = os.path.join(self.audioset_path, 
                                    self.texture_list.iloc[texture_ind].filepath)
        waveform, original_sr = torchaudio.load(file_path)
        start_sample = int(original_sr * self.texture_list.iloc[texture_ind].onset_time)
        end_sample = start_sample + int(original_sr * self.duration)
        waveform = waveform[:, start_sample:end_sample]
        if waveform.shape[0] > 1: # convert to mono if stereo
            waveform = waveform.mean(dim=0, keepdim=True)
        if original_sr != self.sample_rate: # resample if at wrong sampling rate
            waveform = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=self.sample_rate)(waveform)
        waveform2 = set_rms(waveform, self.rms_level)

        mixture = waveform1 + waveform2
        mixture = set_rms(mixture, self.rms_level)
        return mixture
    
    def __len__(self):
        return len(self.mixture_inds)

def get_dataset(batch_size, num_workers, 
                stationarity_screen_list='/orcd/data/jhm/001/om2/jmhicks/projects/TextureStreaming/stimuli/OVERLAPPED_0.5s_all_4s_sound_list_with_stationarity_score_no_speech_no_music_audioset_matlab_coch_rms0p02.csv', 
                audioset_path='/om/data/public/audioset/wavs/unbalanced_train_segments_downloads/', 
                duration=4):
    dataset = AudioSetTextures(stationarity_screen_list, audioset_path, duration)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

def get_mixture_dataset(batch_size, num_workers, 
                        stationarity_screen_list='/orcd/data/jhm/001/om2/jmhicks/projects/TextureStreaming/stimuli/OVERLAPPED_0.5s_all_4s_sound_list_with_stationarity_score_no_speech_no_music_audioset_matlab_coch_rms0p02.csv', 
                        audioset_path='/om/data/public/audioset/wavs/unbalanced_train_segments_downloads/', 
                        duration=4):
    dataset = AudioSetMixtures(stationarity_screen_list, audioset_path, duration)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)