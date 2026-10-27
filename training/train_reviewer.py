"""QLoRA fine-tuning for the review (generative comment) task.

Same trainer as train_verdict with review-task defaults (longer sequences,
more epochs) — kept as its own entrypoint to match the project plan.

    python -m training.train_reviewer --data-dir data/processed/review --epochs 3
"""

from __future__ import annotations

from training import train_verdict


def main() -> None:
    import sys

    sys.argv = [sys.argv[0], "--task", "review", *sys.argv[1:]]
    train_verdict.main()


if __name__ == "__main__":
    main()
