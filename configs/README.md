# Configuration

## Quick Start

```bash
# 1. Copy the example config
cp configs/config.yaml.example configs/config.yaml

# 2. Edit configs/config.yaml — fill in your API key and base_url
#    The framework uses OpenAI-compatible endpoints.
#    Recommended: claude-3-5-sonnet-v2, gpt-4o, gemini-2.0-flash

# 3. (Optional) Copy and customize the run config
cp configs/run_config.yaml.example configs/run_config.yaml
```

## Files

| File | Purpose |
|------|---------|
| `config.yaml.example` | LLM API credentials template (model, API key, base URL) |
| `run_config.yaml.example` | Experiment parameters (workers, branching settings, data paths) |

## Key Parameters (run_config.yaml)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `branching_n_candidates` | 5 | Number of candidate diagnostic plans to generate (0 = disable branching) |
| `branching_k` | 3 | Top-K plans to execute per failed case |
| `branching_env_fail_threshold` | 0.7 | P(EnvFail) threshold for early stopping |
| `workers` | 10 | Parallel evaluation workers |
| `model` | remote | `local` \| `remote` \| `text` |

## Environment Variables

You can use environment variables in YAML configs:

```yaml
api_key: "${OPENAI_API_KEY}"
```

Set them before running:
```bash
export OPENAI_API_KEY="sk-your-key-here"
```
