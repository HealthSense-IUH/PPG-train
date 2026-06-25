import numpy as np
import pandas as pd

fs = 125           # sampling frequencypython -m pip uninstall numpy

duration = 10      # seconds
t = np.arange(0, duration, 1/fs)
ppg = 0.5 * np.sin(2 * np.pi * 1.2 * t) + 0.5  # simple sinusoidal PPG

df = pd.DataFrame({'ppg': ppg})
df.to_csv('example_ppg.csv', index=False)
print("CSV saved: example_ppg.csv")
