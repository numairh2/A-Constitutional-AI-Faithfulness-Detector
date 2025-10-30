# Experiment Tracking Setup

This project uses Weights & Biases (wandb) for experiment tracking.

## Quick Start

### 1. Install wandb
```bash
pip install wandb
```

### 2. Login
```bash
wandb login
```
Or run: `python setup_tracking.py`

### 3. Use in your code
```python
from src.tracking_utils import init_experiment, log_metrics, finish_experiment

# Initialize
run = init_experiment(
    experiment_name="my_experiment",
    config={"model": "llama-3-8b", "batch_size": 4}
)

# Log metrics
log_metrics({"loss": 0.5, "accuracy": 0.85})

# Finish
finish_experiment()
```

## Features

- **Automatic logging**: Metrics, configs, and artifacts
- **Example tracking**: Log reasoning examples with faithfulness scores
- **Visualization**: Automatic plots and tables in wandb dashboard
- **Comparison**: Compare experiments side-by-side
- **Artifacts**: Version models and datasets

## Utilities

### `init_experiment()`
Initialize experiment tracking with name, config, and tags.

### `log_metrics()`
Log numerical metrics (loss, accuracy, faithfulness scores, etc).

### `log_example()`
Log individual examples with questions, reasoning, and scores.

### `log_confusion_matrix()`
Log confusion matrices for detection evaluation.

### `save_artifact()`
Version and save models, datasets, or other files.

### `finish_experiment()`
Cleanly finish and sync experiment.

## Example Experiments

See `examples/example_tracking.py` for a complete example.

## Configuration

Edit `configs/wandb_config.json` to change:
- Project name
- Default entity
- Default tags
- Default config values

## Without wandb

If wandb is not installed, the utilities will fall back to console logging.
You can still use all the same functions - they just won't upload to wandb.

## Dashboard

View your experiments at: https://wandb.ai/YOUR_USERNAME/constitutional-ai-faithfulness

## Tips

1. **Tag experiments** for easy filtering: `tags=["baseline", "iphr"]`
2. **Add notes** to remember what you tried: `notes="Testing new prompt format"`
3. **Log examples** to debug faithfulness issues
4. **Save artifacts** to version your models and datasets
5. **Compare runs** side-by-side in the wandb UI

## Troubleshooting

**Can't login?**
- Check you have a wandb account
- Try: `wandb login --relogin`

**Slow upload?**
- Reduce logging frequency
- Use `log_metrics(..., step=step)` only every N steps

**Don't want to use wandb?**
- Set `use_wandb=False` in `init_experiment()`
- Metrics will print to console instead
