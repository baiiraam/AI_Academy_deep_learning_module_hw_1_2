"""
run_all.py

Main entry point for the assignment.
Run with: STUDENT_ID=<your_id> python run_all.py
"""

import os
import subprocess
import sys


def main():
    """Run the entire pipeline: tests, training, and comparison."""
    student_id = os.environ.get("STUDENT_ID")

    if student_id is None:
        print("ERROR: STUDENT_ID environment variable not set.")
        print("Usage: STUDENT_ID=<your_id> python run_all.py")
        sys.exit(1)

    try:
        student_id = int(student_id)
    except ValueError:
        print(f"ERROR: STUDENT_ID must be an integer, got '{student_id}'")
        sys.exit(1)

    print("=" * 70)
    print(f"RUNNING PIPELINE FOR STUDENT ID: {student_id}")
    print("=" * 70)

    # Step 1: Run tests
    print("\n[1/4] Running tests...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v"],
        capture_output=False,
        check=False,
    )

    if result.returncode != 0:
        print("\n❌ Tests failed! Fix errors before proceeding.")
        sys.exit(1)

    print("\n✅ All tests passed!")

    # Step 2: Train best model
    print("\n[2/4] Training best model...")
    result = subprocess.run(
        [sys.executable, "experiments/train_best.py"],
        env={**os.environ, "STUDENT_ID": str(student_id)},
        check=False,
    )

    if result.returncode != 0:
        print("\n❌ Training failed!")
        sys.exit(1)

    print("\n✅ Training complete!")

    # Step 3: Compare optimizers (now includes RMSProp)
    print("\n[3/4] Comparing optimizers (including RMSProp)...")
    result = subprocess.run(
        [sys.executable, "experiments/compare_optimizers.py"],
        env={**os.environ, "STUDENT_ID": str(student_id)},
        check=False,
    )

    if result.returncode != 0:
        print("\n❌ Optimizer comparison failed!")
        sys.exit(1)

    print("\n✅ Optimizer comparison complete!")

    # Step 4: Verify figures
    print("\n[4/4] Verifying output...")
    figures = ["figures/training_curves.png", "figures/optimizer_comparison.png"]

    for fig in figures:
        if os.path.exists(fig):
            print(f"✅ {fig} generated")
        else:
            print(f"❌ {fig} NOT found")

    print("\n" + "=" * 70)
    print("🎉 PIPELINE COMPLETE!")
    print("Figures saved in: figures/")
    print("=" * 70)


if __name__ == "__main__":
    main()
