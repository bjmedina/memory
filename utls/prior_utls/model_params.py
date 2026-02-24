# texture model parameters
# parameters roughly match that used by McDermott & Simoncelli 2011 texture model implemented in MATLAB
# one exception is that more cochlear filters are used

import chcochleagram
from chexture_choolbox.auditorytexture import custom_chcochleagram_ops

# parameters for cochlear filters
coch_params = {'rep_type': 'cochleagram',
               'rep_kwargs': {'signal_size':200000,
                              'sr':20000,
                              'env_sr': 400,
                              'pad_factor':None,
                              'use_rfft':True,
                              'coch_filter_type': chcochleagram.cochlear_filters.ERBCosFilters,
                              'coch_filter_kwargs': {'n':48,
                                                     'low_lim':20,
                                                     'high_lim':10000,
                                                     'sample_factor':1,
                                                     'no_highpass':False,
                                                     'no_lowpass':False,
                                                     'full_filter':False},
                              'env_extraction_type': custom_chcochleagram_ops.HilbertEnvelopeExtractionMatlab,
                              'env_extraction_kwargs': {'offset_abs':1e-16},
                              'downsampling_type': custom_chcochleagram_ops.SincWithKaiserWindowCustom,
                              'downsampling_kwargs': {'window_size':1001,
                                                      'padding':[500,500],
                                                      'pad_value':1e-8**0.3} 
                             },
               'compression_type': chcochleagram.compression.ClippedGradPowerCompression,
               'compression_kwargs': {'scale': 1,
                                      'offset':1e-16,
                                      'clip_value': 5,
                                      'power': 0.3},
               'cochleagram_kwargs': {'downsample_then_compress':False}
                }

# parameters for constant Q modulation filters
mod_params = {'sr':400,
              'low_lim':0.5,
              'high_lim':200,
              'n':20,
              'q':2,
              'coch_signal_length':4000,
              'use_rfft':False}

# parameters for octave spaced modulation filters
octmod_params = {'sr':400,
                 'low_lim':1,
                 'high_lim':200,
                 'n':20,
                 'coch_signal_length':4000,
                 'use_rfft' : False}

# get useful audio info from model params to be used later
audio_sr = coch_params['rep_kwargs']['sr']
audio_length = coch_params['rep_kwargs']['signal_size'] / audio_sr # in seconds

def update_param_dicts(duration):
    # duration is in seconds
    signal_length = int(duration * audio_sr)
    coch_signal_length = int(signal_length * coch_params['rep_kwargs']['env_sr'] / audio_sr)
    coch_params['rep_kwargs']['signal_size'] = signal_length
    mod_params['coch_signal_length'] = coch_signal_length
    octmod_params['coch_signal_length'] = coch_signal_length
    return coch_params, mod_params, octmod_params