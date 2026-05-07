# Experiments

Scripts for reproducing the experiments in the paper.

| Script | Description |
|--------|-------------|
| `run_test.py` | Main evaluation with diagnostic retry |
| `run_tell_verifier_posthoc.py` | Post-hoc TellVerifier analysis on Round 1 results |

## Usage

```bash
# Main experiment
bash scripts/reproduce.sh

# Or run directly:
python experiments/run_test.py --config configs/run_config.yaml

# Post-hoc TellVerifier
python experiments/run_tell_verifier_posthoc.py \
    --work_dir work_dirs/<round1_dir> \
    --config configs/run_config.yaml \
    --workers 8
```
