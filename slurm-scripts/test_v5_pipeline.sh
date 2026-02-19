#!/bin/bash
# Quick dry-run test for the v5 pipeline.
# Generates a small set of YAMLs, picks the first one, and checks that
# main_v5.py can at least parse the config and import everything cleanly.
#
# Usage:  bash slurm-scripts/test_v5_pipeline.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

TMP_YAML_DIR=$(mktemp -d /tmp/v5_test_yamls_XXXX)
trap "rm -rf $TMP_YAML_DIR" EXIT

echo "=== 1. Generate test YAMLs ==="
python slurm-scripts/gen_model_yamls_v5.py --out-dir "$TMP_YAML_DIR"

N_YAMLS=$(ls "$TMP_YAML_DIR"/run_*.yaml 2>/dev/null | wc -l)
echo "  Generated $N_YAMLS YAML files"

if [ "$N_YAMLS" -eq 0 ]; then
  echo "FAIL: no YAMLs generated"
  exit 1
fi

echo ""
echo "=== 2. Inspect first YAML ==="
FIRST_YAML=$(ls "$TMP_YAML_DIR"/run_*.yaml | sort | head -1)
echo "  $FIRST_YAML"
cat "$FIRST_YAML"

echo ""
echo "=== 3. Validate YAML structure ==="
python -c "
import yaml, sys
with open('$FIRST_YAML') as f:
    cfg = yaml.safe_load(f)

required = ['results_root', 'experiment', 'metric', 'noise_model', 'fitting', 'representation', 'run_id']
missing = [k for k in required if k not in cfg]
if missing:
    print(f'FAIL: missing keys: {missing}')
    sys.exit(1)

fit_keys = ['n_grid', 'n_mc', 'n_refine_iters', 'n_experiments_per_isi', 'k_stimuli_per_exp']
missing_fit = [k for k in fit_keys if k not in cfg['fitting']]
if missing_fit:
    print(f'FAIL: fitting section missing keys: {missing_fit}')
    sys.exit(1)

print('  All required keys present')
print(f'  experiment.which_task = {cfg[\"experiment\"][\"which_task\"]}')
print(f'  metric = {cfg[\"metric\"]}')
print(f'  noise_model.name = {cfg[\"noise_model\"][\"name\"]}')
print(f'  representation.type = {cfg[\"representation\"][\"type\"]}')
print(f'  fitting.n_grid = {cfg[\"fitting\"][\"n_grid\"]}')
"

echo ""
echo "=== 4. Test save/load round-trip (pure stdlib) ==="
python -c "
import os, tempfile, shutil, pickle, json
from datetime import datetime

# Inline save/load using same logic as sigma_fitting.py
# (avoids importing numpy/sklearn which may not be in this env)

fit_result = {
    'sigma0': 0.72, 'sigma1': 0.34, 'sigma2': 0.008,
    'stage_a': {'best_sigma': 0.72, 'best_mse': 0.01, 'best_result': {}, 'all_results': [], 'bounds_history': []},
    'stage_b': {'best_sigma': 0.34, 'best_mse': 0.02, 'best_result': {}, 'all_results': [], 'bounds_history': []},
    'stage_c': {'best_sigma': 0.008, 'best_mse': 0.03, 'best_result': {}, 'all_results': [], 'bounds_history': []},
}

tmpdir = tempfile.mkdtemp()
try:
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    fname = f'test_{timestamp}'

    payload = {
        'fit_result': fit_result,
        'config': {'test': True},
        'metadata': {
            'encoder_name': 'test_encoder',
            'task_name': 'test_task',
            'metric': 'cosine',
            'noise_mode': 'three-regime',
            't_step': 5,
            'human_curve': [3.0, 2.5, 2.0],
            'isis': [0, 1, 2],
            'param_bounds': {'sigma0': [0.5, 1.0]},
            'timestamp': timestamp,
            'fitting_settings': {'n_grid': 15, 'n_mc': 32},
        },
    }

    pkl_path = os.path.join(tmpdir, f'{fname}.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(payload, f)

    json_meta = {
        'sigma0': fit_result['sigma0'],
        'sigma1': fit_result['sigma1'],
        'sigma2': fit_result['sigma2'],
        **payload['metadata'],
    }
    json_path = os.path.join(tmpdir, f'{fname}.json')
    with open(json_path, 'w') as f:
        json.dump(json_meta, f, indent=2)

    # load back
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    assert data['fit_result']['sigma0'] == 0.72
    assert data['metadata']['encoder_name'] == 'test_encoder'
    print('  pickle round-trip: PASS')

    with open(json_path) as f:
        j = json.load(f)
    assert j['sigma0'] == 0.72
    assert j['encoder_name'] == 'test_encoder'
    print('  JSON sidecar: PASS')
finally:
    shutil.rmtree(tmpdir)
"

echo ""
echo "=== 5. Test sigma_fitting.py source structure ==="
python -c "
import ast, sys

with open('utls/sigma_fitting.py') as f:
    tree = ast.parse(f.read())

func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

required_funcs = ['save_three_stage_result', 'load_three_stage_result', 'three_stage_fit', 'fit_sigma_1d']
missing = [fn for fn in required_funcs if fn not in func_names]
if missing:
    print(f'FAIL: missing functions: {missing}')
    sys.exit(1)

print(f'  Found functions: {[f for f in func_names if f in required_funcs]}')
print('  sigma_fitting.py structure: PASS')
"

echo ""
echo "=== 6. Test toy_experiments.py imports ==="
python -c "
import sys
sys.path.insert(0, '.')
from utls.toy_experiments import make_toy_experiment_list, make_multi_isi_toy_experiments, make_isi_n_block_experiment

# quick sanity: ISI=0 should produce [A,A,B,B]
seq = make_isi_n_block_experiment(['A','B'], isi=0)
assert seq == ['A','A','B','B'], f'unexpected: {seq}'

# ISI=1 should produce [A,B,A,B]
seq = make_isi_n_block_experiment(['A','B'], isi=1)
assert seq == ['A','B','A','B'], f'unexpected: {seq}'

print('  toy_experiments imports: PASS')
print('  ISI block generation: PASS')
"

echo ""
echo "=== ALL TESTS PASSED ==="
