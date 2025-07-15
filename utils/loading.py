import os
import json
import pandas as pd

'''
functions to help load human results
'''

def load_results(results_dir, isi_pow=2, min_trials=120, skip_len60=True):
    """Load and filter experiment result CSVs."""
    files = sorted(
        [f for f in os.listdir(results_dir) if f.endswith(".csv")],
        key=lambda fn: os.path.getctime(os.path.join(results_dir, fn))
    )
    exps, seqs, fnames = [], [], []
    for fn in files:
        df = pd.read_csv(os.path.join(results_dir, fn))
        main = df[df.stim_type == "main"]
        seq_file = main.sequence_file.iloc[0].split("/")[-1]
        if len(main) < min_trials: continue
        if "tol0" in seq_file: continue
        exps.append(main); seqs.append(seq_file); fnames.append(fn)
    return exps, seqs, fnames

def remove_sequences_with_len60(seq_dir):
    """Remove entries containing 'len60' from unused.json and used.json."""
    for sub in ("unused", "used"):
        path = os.path.join(seq_dir, sub, f"{sub}.json")
        data = json.load(open(path))
        filtered = [f for f in data if "len60" not in f]
        json.dump(sorted(filtered), open(path, "w"), indent=2)

def move_sequences_to_used(seq_dir, seqs_used):
    """Move used sequence filenames from unused.json to used.json."""
    u_path = os.path.join(seq_dir, "unused", "unused.json")
    z_path = os.path.join(seq_dir, "used",   "used.json")
    unused = json.load(open(u_path)); used = json.load(open(z_path))
    seqs = [os.path.basename(s) for s in seqs_used]
    new_unused = [s for s in unused if s not in seqs]
    new_used = sorted(set(used + seqs))
    json.dump(sorted(new_unused), open(u_path, "w"), indent=2)
    json.dump(new_used,         open(z_path, "w"), indent=2)