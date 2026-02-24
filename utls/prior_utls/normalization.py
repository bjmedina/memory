from chexture_choolbox.auditorytexture.helpers import ZScoreFromNormalizationDict

def normalization_dict_to_device(normalization_dict, device):
    for norm_type_key in normalization_dict.keys():
        for layer_key in normalization_dict[norm_type_key].keys():
            for type_key in normalization_dict[norm_type_key][layer_key].keys():
                for stat_key in normalization_dict[norm_type_key][layer_key][type_key].keys():
                    normalization_dict[norm_type_key][layer_key][type_key][stat_key] = normalization_dict[norm_type_key][layer_key][type_key][stat_key].to(device)
                    return normalization_dict

# by default ZScoreFromNormalizationDict sends normalization info to cpu
# this function sends it to the specified device (e.g., gpu)
# see chexture-toolbox.auditorytexture.helpers.ZScoreFromNormalizationDict.register_dict_params
def norm_params_to_device(normalize, device, param_dict, param_name_base):
    for layer_key, layer_info in param_dict.items():
        for type_key, type_info in layer_info.items():
            for stat_key, stat_info in type_info.items():
                stat_info = stat_info.clone().detach().to(device)
                normalize.register_buffer('%s_%s_%s_%s'%(param_name_base, layer_key, type_key, stat_key), stat_info)

# creates a normalization function to be used on specified device (e.g., gpu)
# WARNING: this function will modify (i.e., normalize) a stat dict in place
# see chexture-toolbox.auditorytexture.helpers.ZScoreFromNormalizationDict
def get_normalization_function(normalization_dict, device):
    normalization_dict = normalization_dict_to_device(normalization_dict, device)
    normalize = ZScoreFromNormalizationDict(normalization_dict, stat_size_class_normalization='sqrt_num_stats')
    norm_params_to_device(normalize, device, normalization_dict['all_texture_mean'], param_name_base='normalization_mean')
    norm_params_to_device(normalize, device, normalization_dict['all_texture_std'], param_name_base='normalization_std')
    return normalize