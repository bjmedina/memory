import torch
import torchaudio
import math
import IPython.display as ipd

from texture_prior.params import model_params

def rms(waveform):
    return torch.sqrt(torch.mean(torch.pow(waveform,2)))

def rms_normalize(waveform, rms_level=0.02):
    return torch.div(torch.mul(waveform, rms_level), rms(waveform))

def next_power_of_2(x):
    # computes next closest power of 2 for fft
    return int(torch.pow(2, torch.ceil(torch.log2(x))))

def spectrally_matched_noise(input_waveform, match_rms=True):
    # creates noise spectrally matched to input waveform
    # output noise will be on same device as input waveform
    # useful place for optimization to start
    n_samples = input_waveform.shape[-1]
    n_fft = next_power_of_2(torch.tensor(n_samples))
    input_fft = torch.fft.rfft(input_waveform, n_fft)
    noise_fft = torch.abs(input_fft) * torch.exp(2 * torch.tensor(math.pi) * 1j * torch.rand(input_fft.shape[-1]).to(input_waveform.device))
    noise_waveform = torch.real(torch.fft.irfft(noise_fft, n_fft))
    noise_waveform = noise_waveform[:n_samples]
    if match_rms==True:
        input_rms = rms(input_waveform)
        noise_waveform = rms_normalize(noise_waveform, input_rms)
    return noise_waveform

def display(waveform, sr=model_params.audio_sr, label=None):
    if label is not None:
        print(label)
    if torch.is_tensor(waveform):
        ipd.display(ipd.Audio(torch.squeeze(waveform.cpu()), rate=sr))
    else:
        ipd.display(ipd.Audio(waveform, rate=sr))

def load(filepath, transform=True, return_params=True):
    waveform, orig_sr = torchaudio.load(filepath)
    if return_params: # then also return param dict for texture model
        if transform: # then properly format audio sr and rms level
            if orig_sr != model_params.audio_sr:
                resample = torchaudio.transforms.Resample(orig_sr, model_params.audio_sr, dtype=waveform.dtype)
                waveform = resample(waveform)            
            waveform = rms_normalize(waveform)
            sr = model_params.audio_sr
        else:
            sr = orig_sr
        duration = waveform.shape[-1] / sr
        coch_params, mod_params, octmod_params = model_params.update_param_dicts(duration)
        return waveform[None,:], sr, coch_params, mod_params, octmod_params
    else:
        return waveform[None,:], sr

def save(filepath, waveform, sr=model_params.audio_sr):
    torchaudio.save(filepath, waveform, sr)