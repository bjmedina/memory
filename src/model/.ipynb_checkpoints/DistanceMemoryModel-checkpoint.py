import glob
import torch
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.io import wavfile
from scipy import signal
from sklearn.decomposition import PCA

sys.path.append('/orcd/data/jhm/001/om2/bjmedina/')

from chexture_choolbox.auditorytexture.statistics_sets import (
    STAT_SET_FULL_MCDERMOTTSIMONCELLI as statistics_dict
)
from chexture_choolbox.auditorytexture.texture_model import TextureModel
from chexture_choolbox.auditorytexture.helpers import FlattenStats
from texture_prior.params import model_params

import torch.nn as nn

class DistanceMemoryModel(nn.Module):
    def __init__(self, encoding_model, noise_variance=1.0, criterion=0.5, device='cpu'):
        super(DistanceMemoryModel, self).__init__()
        
        self.encoding_model = encoding_model
        self.noise_variance = noise_variance
        self.criterion = criterion
        self.device = device
        self.memory_bank = []
        self.debug_mode = False

        self.probe_reps = []
        self.memory_snapshots = []
        self.decisions = []
        self.trial_indices = []
        self.pca_result = None
        self.pca_ready = False

        self.filenames_seen = []
        self.probe_filenames = []

    def _toggle_debug(self):

        self.debug_mode = not self.debug_mode

        print(f"Debug flag is set to {self.debug_mode}")

    def clear_memory(self):
        """Clear all stored memory representations."""
        self.memory_bank = []
        self.memory_snapshots = []
        self.probe_reps = []
        self.decisions = []
        self.trial_indices = []
        self.filenames_seen = []
        self.probe_filenames = []


    def encode_sound(self, sound):
        """Encode a single sound into its representation (filepath expected)."""
        with torch.no_grad():
            rep = self.encoding_model(sound).squeeze(0)
        return rep

    def apply_noise_to_memory(self):
        """Simulate memory drift by applying noise to memory representations."""
        noisy_bank = []
        for rep in self.memory_bank:
            noise = torch.randn_like(rep) * self.noise_variance
            noisy_rep = rep + noise
            noisy_bank.append(noisy_rep)
        self.memory_bank = noisy_bank

    def forward(self, sound):
        """
        Process a single sound, decide recognition, store it in memory.
        
        Args:
            sound: raw audio tensor.
        
        Returns:
            decision: tensor([1]) if recognized, tensor([0]) otherwise.
        """
        current_rep = self.encode_sound(sound)

        if self.debug_mode:
            print(f"Current representation of {sound}: {current_rep[:2]}")

        # First presentation, no recognition possible
        if not self.memory_bank:
            decision = torch.tensor([0], device=self.device)
        else:
            # Compute distances
            memory_tensor = torch.stack(self.memory_bank)
            dists = torch.cdist(current_rep.unsqueeze(0), memory_tensor, p=2)

            if self.debug_mode:
                print(f"All distances are {dists}")
                
            min_dist = dists.min()

            if self.debug_mode:
                print(f"The minimum distance is {min_dist}\n")
                
            # Decision based on criterion
            decision = (min_dist <= self.criterion).float().unsqueeze(0)

        # Apply noise (drift) to existing memories
        self.apply_noise_to_memory()

        # Store current representation into memory
        self.memory_bank.append(current_rep)

        # Save internal state for visualization
        self.probe_reps.append(current_rep.detach().cpu())
        self.memory_snapshots.append(torch.stack(self.memory_bank).detach().cpu())
        self.decisions.append(decision.item())
        self.trial_indices.append(len(self.decisions) - 1)
        self.probe_filenames.append(sound)
        
        # Store ground-truth filename
        self.filenames_seen.append(sound)

        def compute_pca_projection(self):
            from sklearn.decomposition import PCA
            all_reps = torch.vstack(self.memory_snapshots + self.probe_reps)
            pca = PCA(n_components=2)
            self.pca_result = pca.fit_transform(all_reps.numpy())
            self.pca_ready = True
        
            # Compute indices for later slicing
            self._mem_lens = [m.shape[0] for m in self.memory_snapshots]
            self._mem_offsets = [sum(self._mem_lens[:i]) for i in range(len(self._mem_lens))]
            self._probe_indices = [
                offset + self._mem_lens[i] for i, offset in enumerate(self._mem_offsets)
            ]

        return decision

    def animate_trials(self, save_path=None):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import numpy as np
    
        fig, ax = plt.subplots(figsize=(6, 6))
        sc_mem = ax.scatter([], [], c='blue', label='Memory (noisy)', alpha=0.6)
        sc_probe = ax.scatter([], [], c='red', marker='X', s=100, label='Probe')
        text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, va='top')
    
        # Determine axis limits using all memory + probe representations
        all_points = torch.cat([torch.cat(self.memory_snapshots), torch.stack(self.probe_reps)], dim=0)[:, :2]
        ax.set_xlim(all_points[:, 0].min() - 1, all_points[:, 0].max() + 1)
        ax.set_ylim(all_points[:, 1].min() - 1, all_points[:, 1].max() + 1)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_title("Recognition Decisions Over Time")
        ax.legend(loc='lower right')
        ax.grid(True)
    
        # Storage for fading trail scatter collections
        memory_trails = []
    
        def update(frame):
            # Clear previous trail collections from plot
            for coll in memory_trails:
                coll.remove()
            memory_trails.clear()
    
            # --- Draw fading memory trail ---
            num_past = frame
            if num_past > 0:
                alphas = np.linspace(0.05, 0.4, num_past)  # older = more transparent
                for i in range(num_past):
                    mem_2d = self.memory_snapshots[i][:, :2].numpy()
                    trail = ax.scatter(mem_2d[:, 0], mem_2d[:, 1], c='gray', alpha=alphas[i], s=15, label='_nolegend_')
                    memory_trails.append(trail)
    
            # --- Current memory and probe ---
            mem_2d = self.memory_snapshots[frame][:, :2].numpy()
            probe_2d = self.probe_reps[frame][:2].numpy()
    
            sc_mem.set_offsets(mem_2d)
            sc_probe.set_offsets(probe_2d.reshape(1, -1))
    
            # --- Trial info ---
            filename = self.probe_filenames[frame]
            model_said = self.decisions[frame]
            ground_truth = filename in self.filenames_seen[:frame]
            correctness = 'correct' if model_said == ground_truth else 'incorrect'
    
            text.set_text(
                f"Trial {frame+1}: "
                f"{'YES' if model_said else 'NO'} (model) | "
                f"{'YES' if ground_truth else 'NO'} (truth) {correctness}\n"
                f"{filename.split('/')[-1]}"
            )
    
            return [sc_mem, sc_probe, text] + memory_trails
    
        ani = animation.FuncAnimation(
            fig, update, frames=len(self.trial_indices), interval=1000, blit=True
        )
    
        if save_path:
            ani.save(save_path, dpi=150, fps=1.0)
            print(f"Animation saved to {save_path}")
        else:
            from IPython.display import HTML
            return HTML(ani.to_jshtml())
