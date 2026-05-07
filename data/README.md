# Data

## Test Data

Test datasets are not included in this repository due to size constraints.

### Download

```bash
bash scripts/download_data.sh
```

### Format

Each test case is a JSON object:

```json
{
  "task_name": "web_task_001",
  "url": "https://target-app.example.com",
  "case_desc": "Description of what to test",
  "expected_result": "Pass",
  "label": 1
}
```

### Example

A minimal example is provided in `data/example/sample_case.json`.

## Model Weights

- `omniparser_icon_detect.pt` — Icon detection model (required only for `[ultra]` mode)
  - Download via `scripts/download_data.sh`
