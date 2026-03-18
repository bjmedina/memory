import glob
import torch
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.io import wavfile
from scipy import signal
from sklearn.decomposition import PCA

sys.path.append('/om2/user/bjmedina/')

from chexture_choolbox.auditorytexture.statistics_sets import (
    STAT_SET_FULL_MCDERMOTTSIMONCELLI as statistics_dict
)
from chexture_choolbox.auditorytexture.texture_model import TextureModel
from chexture_choolbox.auditorytexture.helpers import FlattenStats
from texture_prior.params import model_params

import torch.nn as nn

class DistanceMemoryPriorModel(nn.Module):
    def __init__(self, encoding_model, score_model, drift_step_size=0.001, noise_variance=1.0, criterion=0.5, device='cpu'):
        super(DistanceMemoryPriorModel, self).__init__()
        
        self.encoding_model = encoding_model
        self.noise_variance = noise_variance
        self.criterion = criterion
        self.device = device
        self.memory_bank = []
        self.debug_mode = False
        self.score_model = score_model # this is in essence the prior (gradient of the log prior evaluated at x)

        self.drift_step_size = drift_step_size
        self.probe_reps = []
        self.memory_snapshots = []
        self.decisions = []
        self.trial_indices = []
        self.pca_result = None
        self.pca_ready = False
        self.filenames_seen = []
        self.probe_filenames = []

        self.output = []

    def _toggle_debug(self):

        self.debug_mode = not self.debug_mode

        print(f"Debug flag is set to {self.debug_mode}")

    def _fill_memory_bank(self, sound_list):
        # TODO
        self.memory_bank = []
        for sound in sound_list:
            rep = self.encode_sound(sound)
            self.memory_bank.append(rep)
    
    def clear_memory(self):
        """Clear all stored memory representations."""
        self.memory_bank = []
        self.memory_snapshots = []
        self.probe_reps = []
        self.decisions = []
        self.trial_indices = []
        self.filenames_seen = []
        self.probe_filenames = []
        self.output = []
    
    def encode_sound(self, sound):
        """Encode a single sound into its representation (filepath expected)."""
        with torch.no_grad():
            rep = self.encoding_model(sound).squeeze(0)
        return rep

    def apply_noise_to_memory_less_fast(self):
        """Simulate memory drift: Gaussian noise + small score-driven drift."""
        noisy_bank = []
        for rep in self.memory_bank:
            t = torch.tensor([0.125], device=rep.device, dtype=rep.dtype)
            noise = torch.randn_like(rep) * self.noise_variance
            with torch.no_grad():
                drift = self.score_model.score_model(rep.view(1, 1, 1, -1), t).view_as(rep)
            noisy_rep = rep + noise + self.drift_step_size * drift
            noisy_bank.append(noisy_rep)
        self.memory_bank = noisy_bank
    
    def apply_noise_to_memory(self):
        """Simulate memory drift on the whole bank in one call."""
        reps = torch.stack(self.memory_bank)  # [N, D]
        noise = torch.randn_like(reps) * self.noise_variance
        t = torch.full((reps.shape[0],), 0.125, device=reps.device, dtype=reps.dtype)
        with torch.no_grad():
            drift = self.score_model.score_model(reps.view(reps.shape[0], 1, 1, -1), t).view_as(reps)
        noisy_reps = reps + noise + self.drift_step_size * drift
        self.memory_bank = list(noisy_reps)

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
            self.output.append(np.inf)
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

            self.output.append(min_dist)
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
            output = self.output[frame]
    
            text.set_text(
                f"Trial {frame+1}: "
                f"{'YES' if model_said else 'NO'} (model) | "
                f"{'YES' if ground_truth else 'NO'} (truth) {correctness}\n"
                f"DISTANCE: {output:.2f}\n"
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

    def do_experiment(self, sound_list, yt_ids=None, verbose=False):
        """
        Run a sequence of sound file paths through the memory model,
        and return trial-wise info for better downstream evaluation.
    
        Args:
            sound_list (list of str): List of stimulus paths.
            yt_ids (list of str or None): List of same length as sound_list indicating YT IDs or unique trial IDs.
            verbose (bool): Print trial-by-trial info.
    
        Returns:
            pd.DataFrame: Trial-level model output.
        """
        import pandas as pd
    
        self.clear_memory()
    
        seen_yt = {}
        rows = []
    
        if yt_ids is None:
            yt_ids = sound_list  # fallback: use filename as yt_id
    
        for i, (stim_path, yt_id) in enumerate(zip(sound_list, yt_ids)):
            decision = self(stim_path).item()
            
            if yt_id in seen_yt:
                repeat = 'true'
                isi = i - seen_yt[yt_id] - 1
            else:
                repeat = 'false'
                isi = -1
                seen_yt[yt_id] = i
    
            row = {
                'trial': i,
                'stimulus': stim_path,
                'yt_id': yt_id,
                'response': int(decision),
                'repeat': repeat,
                'isi': isi
            }
            rows.append(row)
    
            if verbose:
                print(f"{stim_path.split('/')[-1]} => Model: {'YES' if decision else 'NO'}, Truth: {'YES' if repeat == 'true' else 'NO'}, ISI={isi}")
    
        return pd.DataFrame(rows)

    def export_responses_to_dataframe(self, metadata_df=None, additional_outputs=None):
        """
        Export trial-by-trial model responses and internal metrics to a Pandas DataFrame.
        
        Args:
            metadata_df (pd.DataFrame or None): Optional DataFrame with metadata per trial.
                Must include a 'stimulus' column that matches self.probe_filenames.
            additional_outputs (dict or None): Optional dictionary of additional trial-wise metrics.
                Example: {'min_dist': [...], 'likelihood': [...], etc.}
        
        Returns:
            pd.DataFrame: Trial-level DataFrame with responses, correctness, and metadata.
        """
        import pandas as pd
        
        n_trials = len(self.decisions)
        
        df = pd.DataFrame({
            'trial': self.trial_indices,
            'stimulus': self.probe_filenames,
            'model_response': self.decisions,
            'true_repeat': [1 if fname in self.filenames_seen[:i] else 0 for i, fname in enumerate(self.probe_filenames)],
        })
        
        # Add any additional per-trial metrics (e.g. distances, log-likelihoods)
        if additional_outputs:
            for key, values in additional_outputs.items():
                assert len(values) == n_trials, f"{key} must have length {n_trials}"
                df[key] = values
        
        # Merge with external metadata if provided
        if metadata_df is not None:
            assert 'stimulus' in metadata_df.columns, "'metadata_df' must contain a 'stimulus' column"
            df = df.merge(metadata_df, on='stimulus', how='left')
        
        df['correct'] = (df['model_response'] == df['true_repeat']).astype(int)
        
        return df