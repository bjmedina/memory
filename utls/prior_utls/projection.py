import torch

from chexture_choolbox.auditorytexture.texture_model import TextureModel
from chexture_choolbox.auditorytexture.helpers import FlattenStats

from texture_prior.params import model_params, statistics_set
from texture_prior.utils import normalization, path

class statistics_projector:
    def __init__(self, type='texture', device='cpu'):
        # intialize flatstat function
        self.flatstat = FlattenStats(statistics_set.statistics)
        
        # note: must run a forward pass of flatstat to register expected shapes so undo method will work properly
        sample_rate = 20000
        duration = 4
        coch_params, mod_params, octmod_params = model_params.update_param_dicts(duration)
        model = TextureModel(coch_params, 
                            mod_params, 
                            octmod_params,
                            statistics_dict=statistics_set.statistics)
        example_input = torch.randn(1, 1, int(sample_rate*duration)) / 0.01
        example_stat_dict = model(example_input)
        example_stat_vec = self.flatstat(example_stat_dict)

        # get normalization function
        if type == 'texture':
            filepath = path.relative('../assets/normalization_dict.pt')
        elif type == 'mixture':
            filepath = path.relative('../assets/normalization_dict_mixtures.pt')
        normalization_dict = torch.load(filepath)
        self.normalize = normalization.get_normalization_function(normalization_dict, device)

        # load PCA info
        if type == 'texture':
            filepath = path.relative('../assets/principal_components.pt')
        elif type == 'mixture':
            filepath = path.relative('../assets/principal_components_mixtures.pt')
        pca = torch.load(filepath)
        self.PCs = pca['PCs'].to(device)

    def project(self, statistics, normalize=True, nPCs=10):  
        # stats below are for textures
        # 2698 PCs explain 95% of the variance
        # 509 PCs explain 80% of the variance
        # reduced to 50 PCs for faster computation (~59% of variance)   
        if not isinstance(statistics, dict): # assume vector and unflatten into dict
            statistics = self.flatstat.undo_flatten_stats(statistics)
        if normalize:
            statistics = self.normalize(statistics)
        return self.flatstat(statistics) @ self.PCs[:, :nPCs]

    def unproject(self, statistics, unnormalize=True, nPCs=10, return_dict=True):        
            statistics = statistics @ self.PCs[:, :nPCs].T
            statistics = self.flatstat.undo_flatten_stats(statistics)
            if unnormalize:
                statistics = self.normalize.undo_z_score(statistics)
            if not return_dict:
                return self.flatstat(statistics)
            else:
                return statistics