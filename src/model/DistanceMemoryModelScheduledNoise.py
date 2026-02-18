import torch
import torch.nn as nn
import numpy as np

class DistanceMemoryModelScheduledNoise(nn.Module):
    def __init__(self, encoding_model, 
                 max_variance=1.0, min_variance=0.0, total_steps=100,
                 criterion=0.5, device='cpu'):
        """
        Memory model with linearly scheduled noise variance.

        Args:
            encoding_model: encoder for sound stimuli
            max_variance (float): starting noise variance
            min_variance (float): ending noise variance
            total_steps (int): steps to decay from max -> min
            criterion (float): distance threshold for recognition
            device (str): torch device
        """
        super().__init__()
        self.encoding_model = encoding_model
        self.max_variance = max_variance
        self.min_variance = min_variance
        self.total_steps = total_steps
        self.device = device
        self.criterion = criterion

        # Internal trackers
        self.memory_bank = []
        self.current_step = 0
        self.ts = [] # age of each trace

        # Logging
        self.decisions = []
        self.probe_reps = []
        self.memory_snapshots = []
        self.trial_indices = []
        self.probe_filenames = []
        self.filenames_seen = []
        self.output = []

        self.debug_mode = False

    
    def _toggle_debug(self):
        self.debug_mode = not self.debug_mode
        print(f"Debug flag is set to {self.debug_mode}")
    
    def _fill_memory_bank(self, sound_list):
        # TODO
        self.memory_bank = []
        for sound in sound_list:
            rep = self.encode_sound(sound)
            self.memory_bank.append(rep)

    def _variance_for_age(self, age):
        """Linearly interpolate variance for a given age."""
        step = min(age, self.total_steps)
        frac = step / self.total_steps
        return self.max_variance - frac * (self.max_variance - self.min_variance)

    def encode_sound(self, sound):
        """Encode one sound into a representation (expects filepath)."""
        with torch.no_grad():
            rep = self.encoding_model(sound).squeeze(0)
        return rep

    def apply_noise_to_memory(self):
        """Apply drift with independent schedules per memory."""
        noisy_bank = []

        for rep, age in zip(self.memory_bank, self.ts):
            variance = self._variance_for_age(age)

            if self.debug_mode:
                print(f"Variance at age {age}: {variance}")
            noise = torch.randn_like(rep) * variance
            noisy_bank.append(rep + noise)
        self.memory_bank = noisy_bank

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
        self.ts = []

    def forward(self, sound):
        """Process one sound, return decision."""
        current_rep = self.encode_sound(sound)

        if not self.memory_bank:
            self.output.append(np.inf)
            decision = torch.tensor([0], device=self.device)
        else:
            memory_tensor = torch.stack(self.memory_bank)
            dists = torch.cdist(current_rep.unsqueeze(0), memory_tensor, p=2)
            min_dist = dists.min()
            self.output.append(min_dist)
            decision = (min_dist <= self.criterion).float().unsqueeze(0)

        self.apply_noise_to_memory()
        self.memory_bank.append(current_rep)
        self.ts.append(0)
        self.ts = [t + 1 for t in self.ts]

        # Save trial data
        self.decisions.append(decision.item())
        self.probe_reps.append(current_rep.detach().cpu())
        self.memory_snapshots.append(torch.stack(self.memory_bank).detach().cpu())
        self.trial_indices.append(len(self.decisions) - 1)
        self.probe_filenames.append(sound)
        self.filenames_seen.append(sound)

        return decision