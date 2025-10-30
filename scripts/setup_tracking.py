#!/usr/bin/env python3
"""
Setup Experiment Tracking with Weights & Biases

This script helps you set up experiment tracking for your project.
"""

import os
import sys
import json
from pathlib import Path


def check_wandb_installed():
    """Check if wandb is installed."""
    try:
        import wandb
        print(f"✓ wandb version {wandb.__version__} installed")
        return True
    except ImportError:
        print("✗ wandb not installed")
        return False


def install_wandb():
    """Install wandb."""
    print("\nInstalling wandb...")
    os.system("pip install wandb")
    print("✓ wandb installed")


def setup_wandb():
    """Interactive setup for wandb."""
    import wandb
    
    print("\n" + "="*80)
    print("WANDB SETUP")
    print("="*80)
    
    # Check if already logged in
    try:
        api = wandb.Api()
        print(f"✓ Already logged in as: {api.viewer()['entity']}")
        return
    except:
        pass
    
    print("\nYou need a Weights & Biases account to track experiments.")
    print("\nOptions:")
    print("1. Create account at https://wandb.ai/site")
    print("2. Login with existing account")
    print("3. Skip for now (can setup later)")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "3":
        print("\n⚠ Skipping wandb setup. You can run this script again later.")
        print("To setup manually: wandb login")
        return
    
    if choice == "1":
        print("\n→ Opening browser to create account...")
        print("After creating account, run: wandb login")
        import webbrowser
        webbrowser.open("https://wandb.ai/site")
        return
    
    # Login
    print("\n→ Opening browser for login...")
    wandb.login()
    print("✓ Successfully logged in!")


def create_config_file():
    """Create wandb config file."""
    print("\n" + "="*80)
    print("CREATING CONFIG FILE")
    print("="*80)
    
    config = {
        "project_name": "constitutional-ai-faithfulness",
        "entity": None,  # Will use default entity
        "notes": "Faithfulness detection in chain-of-thought reasoning",
        "tags": ["faithfulness", "constitutional-ai", "interpretability"],
        "default_config": {
            "model": "meta-llama/Llama-3-8B-Instruct",
            "batch_size": 4,
            "learning_rate": 2e-5,
            "max_length": 512,
        }
    }
    
    # Get entity name if logged in
    try:
        import wandb
        api = wandb.Api()
        viewer = api.viewer()
        config["entity"] = viewer.get("entity", viewer.get("username"))
        print(f"Using entity: {config['entity']}")
    except:
        print("Using default entity")
    
    # Save config
    config_path = Path("configs/wandb_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Config saved to {config_path}")


def create_tracking_utils():
    """Create utility functions for experiment tracking."""
    print("\n" + "="*80)
    print("CREATING TRACKING UTILITIES")
    print("="*80)
    
    utils_code = '''"""
Experiment tracking utilities.

Usage:
    from src.tracking_utils import init_experiment, log_metrics, finish_experiment
    
    # Initialize
    run = init_experiment(
        experiment_name="baseline",
        config={"model": "llama-3-8b", "batch_size": 4}
    )
    
    # Log metrics
    log_metrics({"faithfulness_score": 0.73, "accuracy": 0.85})
    
    # Finish
    finish_experiment()
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


def init_experiment(
    experiment_name: str,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
    notes: Optional[str] = None,
    use_wandb: bool = True
) -> Optional[Any]:
    """
    Initialize experiment tracking.
    
    Args:
        experiment_name: Name of the experiment
        config: Configuration dictionary
        tags: List of tags
        notes: Experiment notes
        use_wandb: Whether to use wandb (if available)
    
    Returns:
        wandb run object if using wandb, None otherwise
    """
    # Load wandb config if exists
    wandb_config_path = Path("configs/wandb_config.json")
    if wandb_config_path.exists():
        with open(wandb_config_path) as f:
            wandb_config = json.load(f)
    else:
        wandb_config = {
            "project_name": "constitutional-ai-faithfulness",
            "entity": None
        }
    
    if use_wandb and WANDB_AVAILABLE:
        # Initialize wandb
        run = wandb.init(
            project=wandb_config["project_name"],
            entity=wandb_config.get("entity"),
            name=experiment_name,
            config=config or {},
            tags=tags or [],
            notes=notes,
        )
        
        print(f"✓ Experiment tracking initialized: {run.url}")
        return run
    else:
        # Fallback: just print to console
        print(f"✓ Experiment: {experiment_name}")
        if config:
            print(f"  Config: {config}")
        return None


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None):
    """
    Log metrics.
    
    Args:
        metrics: Dictionary of metric name -> value
        step: Optional step number
    """
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log(metrics, step=step)
    else:
        # Fallback: print to console
        print(f"Metrics (step {step}):" if step else "Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")


def log_example(
    question: str,
    reasoning: str,
    answer: str,
    faithfulness_score: float,
    step: Optional[int] = None
):
    """
    Log an example with reasoning.
    
    Args:
        question: The question
        reasoning: Chain-of-thought reasoning
        answer: The answer
        faithfulness_score: Faithfulness score (0-1)
        step: Optional step number
    """
    if WANDB_AVAILABLE and wandb.run is not None:
        # Create table
        table = wandb.Table(
            columns=["Question", "Reasoning", "Answer", "Faithfulness"],
            data=[[question, reasoning, answer, faithfulness_score]]
        )
        wandb.log({"examples": table}, step=step)
    else:
        # Fallback: print to console
        print(f"\\nExample (step {step}):" if step else "\\nExample:")
        print(f"  Q: {question}")
        print(f"  R: {reasoning[:100]}...")
        print(f"  A: {answer}")
        print(f"  Faithfulness: {faithfulness_score:.2f}")


def log_confusion_matrix(y_true: list, y_pred: list, labels: list):
    """
    Log confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
    """
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=labels
            )
        })
    else:
        print("Confusion matrix logging requires wandb")


def save_artifact(filepath: str, name: str, artifact_type: str = "dataset"):
    """
    Save an artifact (model, dataset, etc).
    
    Args:
        filepath: Path to file
        name: Artifact name
        artifact_type: Type of artifact
    """
    if WANDB_AVAILABLE and wandb.run is not None:
        artifact = wandb.Artifact(name=name, type=artifact_type)
        artifact.add_file(filepath)
        wandb.log_artifact(artifact)
        print(f"✓ Artifact saved: {name}")
    else:
        print(f"Artifact {name} would be saved (wandb not available)")


def finish_experiment():
    """Finish experiment tracking."""
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()
        print("✓ Experiment finished")
    else:
        print("✓ Experiment completed")


# Example usage
if __name__ == "__main__":
    # Initialize
    run = init_experiment(
        experiment_name="test_experiment",
        config={"model": "test", "batch_size": 4},
        tags=["test"],
        notes="Testing tracking utilities"
    )
    
    # Log some metrics
    for step in range(5):
        log_metrics({
            "loss": 1.0 / (step + 1),
            "accuracy": 0.5 + step * 0.1
        }, step=step)
    
    # Log an example
    log_example(
        question="Is Mount Everest taller than K2?",
        reasoning="Everest is 8,849m and K2 is 8,611m. Since 8,849 > 8,611, yes.",
        answer="yes",
        faithfulness_score=0.95
    )
    
    # Finish
    finish_experiment()
'''
    
    # Save utils
    utils_path = Path("src/tracking_utils.py")
    utils_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(utils_path, 'w') as f:
        f.write(utils_code)
    
    print(f"✓ Tracking utilities created at {utils_path}")


def create_example_script():
    """Create example experiment script."""
    print("\n" + "="*80)
    print("CREATING EXAMPLE SCRIPT")
    print("="*80)
    
    example_code = '''"""
Example experiment script showing how to use tracking.
"""

from src.tracking_utils import init_experiment, log_metrics, log_example, finish_experiment


def run_baseline_experiment():
    """Run a baseline experiment with tracking."""
    
    # Initialize tracking
    run = init_experiment(
        experiment_name="baseline_faithfulness",
        config={
            "model": "meta-llama/Llama-3-8B-Instruct",
            "dataset": "comparative_questions",
            "num_examples": 100,
            "temperature": 0.7,
        },
        tags=["baseline", "iphr"],
        notes="Measuring baseline IPHR rates on comparative questions"
    )
    
    # Simulate experiment
    print("Running baseline experiment...")
    
    # Log metrics over time
    for step in range(10):
        metrics = {
            "examples_processed": (step + 1) * 10,
            "iphr_rate": 0.15 + step * 0.01,  # Simulated
            "avg_faithfulness": 0.70 + step * 0.02,  # Simulated
        }
        log_metrics(metrics, step=step)
    
    # Log some example outputs
    log_example(
        question="Is Mount Everest taller than K2?",
        reasoning="Mount Everest is 8,849 meters tall, while K2 is 8,611 meters. Since 8,849 > 8,611, Mount Everest is taller.",
        answer="yes",
        faithfulness_score=0.95
    )
    
    log_example(
        question="Is K2 taller than Mount Everest?",
        reasoning="K2 is known as the savage mountain and is very challenging to climb. Challenging mountains are usually very tall.",
        answer="yes",  # Wrong! This shows IPHR
        faithfulness_score=0.25
    )
    
    # Final summary metrics
    final_metrics = {
        "total_examples": 100,
        "iphr_rate": 0.23,
        "avg_faithfulness": 0.72,
        "accuracy": 0.85,
    }
    log_metrics(final_metrics)
    
    # Finish
    finish_experiment()
    
    print("✓ Baseline experiment complete!")


if __name__ == "__main__":
    run_baseline_experiment()
'''
    
    example_path = Path("examples/example_tracking.py")
    example_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(example_path, 'w') as f:
        f.write(example_code)
    
    print(f"✓ Example script created at {example_path}")


def create_readme():
    """Create README for experiment tracking."""
    readme = '''# Experiment Tracking Setup

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
'''
    
    readme_path = Path("docs/TRACKING.md")
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(readme_path, 'w') as f:
        f.write(readme)
    
    print(f"✓ Tracking README created at {readme_path}")


def main():
    print("="*80)
    print("EXPERIMENT TRACKING SETUP")
    print("="*80)
    
    # Check if wandb is installed
    if not check_wandb_installed():
        print("\nWandB is required for experiment tracking.")
        install = input("Install now? (y/n): ").strip().lower()
        if install == 'y':
            install_wandb()
        else:
            print("Skipping installation. Install manually: pip install wandb")
            return
    
    # Setup wandb
    try:
        setup_wandb()
    except Exception as e:
        print(f"Error during wandb setup: {e}")
        print("You can setup manually later: wandb login")
    
    # Create config and utilities
    create_config_file()
    create_tracking_utils()
    create_example_script()
    create_readme()
    
    print("\n" + "="*80)
    print("✓ EXPERIMENT TRACKING SETUP COMPLETE")
    print("="*80)
    print("\nFiles created:")
    print("  - configs/wandb_config.json")
    print("  - src/tracking_utils.py")
    print("  - examples/example_tracking.py")
    print("  - docs/TRACKING.md")
    print("\nNext steps:")
    print("1. Test tracking: python examples/example_tracking.py")
    print("2. View dashboard: https://wandb.ai")
    print("3. Use in your experiments! See docs/TRACKING.md")
    print()


if __name__ == "__main__":
    main()