import numpy as np
import pandas as pd

fs = 125           # sampling frequencypython -m pip uninstall numpy

duration = 10      # seconds
t = np.arange(0, duration, 1/fs)
ppg = 0.5 * np.sin(2 * np.pi * 1.2 * t) + 0.5  # simple sinusoidal PPG

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
df = pd.DataFrame({'ppg': ppg})
output_path = PROJECT_ROOT / "data" / "processed" / "example_ppg.csv"
df.to_csv(output_path, index=False)
print(f"CSV saved: {output_path}")
