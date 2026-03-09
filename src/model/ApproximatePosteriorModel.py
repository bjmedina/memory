import torch
import torch.nn as nn
import numpy as np


class ApproximatePosteriorModel(nn.Module):
    """
    Recognition memory model with prior-driven drift (Hypothesis 2).

    Memory traces evolve via:
        trace_new = trace + noise + drift_step_size * score(trace)

    where score(trace) = ∇ log π(trace) is the unit-norm gradient of the
    learned prior, provided by a ScoreFunction instance.

    Decision rule: minimum L2 distance to stored memories ≤ criterion.

    Parameters
    ----------
    encoding_model : callable
        Maps a sound file path to a representation tensor.
    score_model : ScoreFunction
        Provides .forward(x) returning unit-norm ∇ log π(x).
    noise_variance : float
        Standard deviation of Gaussian noise applied to traces each trial.
    drift_step_size : float
        Magnitude of prior-driven drift per trial.
    criterion : float
        Distance threshold for "heard before" decisions.
    device : str
        'cpu' or 'cuda'.
    """

    def __init__(self, encoding_model, score_model, noise_variance=1.0,
                 drift_step_size=0.001, criterion=0.5, device='cpu'):
        super(ApproximatePosteriorModel, self).__init__()

        self.encoding_model = encoding_model
        self.score_model = score_model
        self.noise_variance = noise_variance
        self.drift_step_size = drift_step_size
        self.criterion = criterion
        self.device = device

        self.memory_bank = []
        self.debug_mode = False

        self.probe_reps = []
        self.memory_snapshots = []
        self.decisions = []
        self.trial_indices = []
        self.filenames_seen = []
        self.probe_filenames = []
        self.output = []

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
        self.output = []

    def encode_sound(self, sound):
        """Encode a single sound into its representation (filepath expected)."""
        with torch.no_grad():
            rep = self.encoding_model(sound).squeeze(0)
        return rep

    def apply_noise_to_memory(self):
        """Apply Gaussian noise + prior-driven drift to all stored traces."""
        if not self.memory_bank:
            return

        reps = torch.stack(self.memory_bank)  # [N, D]

        # Gaussian noise
        noise = torch.randn_like(reps) * self.noise_variance

        # Prior-driven drift via score function
        with torch.no_grad():
            drift = self.score_model.forward(reps)  # [N, 1, 1, D] or similar

        noisy_reps = reps + noise + self.drift_step_size * drift.view_as(reps)
        self.memory_bank = list(noisy_reps)

    def forward(self, sound):
        """
        Process a single sound: decide recognition, then store in memory.

        Returns
        -------
        decision : tensor([1]) if recognized, tensor([0]) otherwise.
        """
        current_rep = self.encode_sound(sound)

        if self.debug_mode:
            print(f"Current representation of {sound}: {current_rep[:2]}")

        # First presentation — no recognition possible
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
            decision = (min_dist <= self.criterion).float().unsqueeze(0)

        # Apply noise + drift to existing memories
        self.apply_noise_to_memory()

        # Store current representation into memory
        self.memory_bank.append(current_rep)

        # Save internal state
        self.probe_reps.append(current_rep.detach().cpu())
        self.memory_snapshots.append(
            torch.stack(self.memory_bank).detach().cpu()
        )
        self.decisions.append(decision.item())
        self.trial_indices.append(len(self.decisions) - 1)
        self.probe_filenames.append(sound)
        self.filenames_seen.append(sound)

        return decision

    def do_experiment(self, sound_list, yt_ids=None, verbose=False):
        """
        Run a sequence of sound file paths through the memory model.

        Parameters
        ----------
        sound_list : list of str
            Stimulus paths.
        yt_ids : list of str or None
            Unique trial IDs (defaults to sound_list).
        verbose : bool
            Print trial-by-trial info.

        Returns
        -------
        pd.DataFrame
            Trial-level model output.
        """
        import pandas as pd

        self.clear_memory()

        seen_yt = {}
        rows = []

        if yt_ids is None:
            yt_ids = sound_list

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
                print(
                    f"{stim_path.split('/')[-1]} => "
                    f"Model: {'YES' if decision else 'NO'}, "
                    f"Truth: {'YES' if repeat == 'true' else 'NO'}, "
                    f"ISI={isi}"
                )

        return pd.DataFrame(rows)

    def animate_trials(self, save_path=None):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import numpy as np

        fig, ax = plt.subplots(figsize=(6, 6))
        sc_mem = ax.scatter([], [], c='blue', label='Memory (noisy)', alpha=0.6)
        sc_probe = ax.scatter(
            [], [], c='red', marker='X', s=100, label='Probe'
        )
        text = ax.text(
            0.05, 0.95, '', transform=ax.transAxes, fontsize=12, va='top'
        )

        all_points = torch.cat(
            [torch.cat(self.memory_snapshots),
             torch.stack(self.probe_reps)], dim=0
        )[:, :2]
        ax.set_xlim(all_points[:, 0].min() - 1, all_points[:, 0].max() + 1)
        ax.set_ylim(all_points[:, 1].min() - 1, all_points[:, 1].max() + 1)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_title("Recognition Decisions Over Time")
        ax.legend(loc='lower right')
        ax.grid(True)

        memory_trails = []

        def update(frame):
            for coll in memory_trails:
                coll.remove()
            memory_trails.clear()

            num_past = frame
            if num_past > 0:
                alphas = np.linspace(0.05, 0.4, num_past)
                for i in range(num_past):
                    mem_2d = self.memory_snapshots[i][:, :2].numpy()
                    trail = ax.scatter(
                        mem_2d[:, 0], mem_2d[:, 1],
                        c='gray', alpha=alphas[i], s=15, label='_nolegend_'
                    )
                    memory_trails.append(trail)

            mem_2d = self.memory_snapshots[frame][:, :2].numpy()
            probe_2d = self.probe_reps[frame][:2].numpy()

            sc_mem.set_offsets(mem_2d)
            sc_probe.set_offsets(probe_2d.reshape(1, -1))

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
            fig, update, frames=len(self.trial_indices),
            interval=1000, blit=True
        )

        if save_path:
            ani.save(save_path, dpi=150, fps=1.0)
            print(f"Animation saved to {save_path}")
        else:
            from IPython.display import HTML
            return HTML(ani.to_jshtml())
