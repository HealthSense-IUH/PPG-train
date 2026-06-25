"""Extract a real wrist PPG AF session from Vilnius dataset for testing.

This script extracts Session 7 of Patient 001 (which is validated as AF)
and saves it as a CSV file (outputs/vilnius_wrist_af_sample.csv)
with Huywatch-like format (device_millis, red, ir) for easy testing in the app.
"""

import h5py
import pandas as pd
import numpy as np

ppg_mat = "D:/DATT/PPG-Arrhythmia-Detection/data/001_PPG.mat"
output_csv = "D:/DATT/PPG-Arrhythmia-Detection/outputs/vilnius_wrist_af_sample.csv"

def get_hdf5_text(f, ref):
    obj = f[ref]
    arr = obj[:].flatten().astype(int)
    return "".join(chr(c) for c in arr if c != 0)

def main():
    try:
        f = h5py.File(ppg_mat, 'r')
    except FileNotFoundError:
        print(f"Error: 001_PPG.mat not found at {ppg_mat}.")
        print("Please download it again and place it in the data/ folder to extract the AF sample.")
        return

    print("Extracting Session 7 (AF session) from 001_PPG.mat...")
    
    # Session 7 contains highly validated AF intervals
    ref_ppg = f["PPG_GREEN"][7, 0]
    ppg_raw = f[ref_ppg][0, :].astype(float)
    
    # We will take 12,000 samples (2 minutes of recording at 100 Hz)
    n_samples = min(12000, len(ppg_raw))
    ppg_slice = ppg_raw[:n_samples]
    
    # Generate device_millis (10 ms intervals for 100 Hz)
    device_millis = np.arange(n_samples) * 10
    
    # Format to look like Huywatch (with inverted polarity so we can load it)
    # predict_huywatch.py and ppg_app.py multiply ir by -1, so we write -ppg_slice as ir
    df = pd.DataFrame({
        "device_millis": device_millis,
        "red": np.zeros(n_samples, dtype=int), # dummy red channel
        "ir": -ppg_slice.astype(int)           # actual PPG signal inverted
    })
    
    df.to_csv(output_csv, index=False)
    print(f"Successfully extracted {n_samples} samples of wrist PPG AF.")
    print(f"Saved to: {output_csv}")
    print("You can now upload this file to the Streamlit app to test AF detection!")

if __name__ == "__main__":
    main()
