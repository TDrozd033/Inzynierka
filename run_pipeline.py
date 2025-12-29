# run_pipeline.py
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

def run(script):
    print(f"\n=== RUN: {script} ===")
    subprocess.run(
        [PYTHON, "-m", script],
        cwd=PROJECT_ROOT,
        check=True
    )

if __name__ == "__main__":
    print("=== FOOTBALL PREDICTION PIPELINE ===")

  
    run("src.fetch_league_tables")

    
    run("src.predict_matches")

    print("\n=== PIPELINE FINISHED SUCCESSFULLY ===")
