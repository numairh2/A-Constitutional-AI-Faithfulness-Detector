"""
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
        print(f"\nExample (step {step}):" if step else "\nExample:")
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
