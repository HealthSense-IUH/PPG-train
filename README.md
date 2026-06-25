# 💓 PPG Atrial Fibrillation Detection with Explainable Machine Learning

An **end-to-end PPG-based atrial fibrillation (AF) detection system** combining **signal preprocessing**, **window-based classification**, **SHAP explainability**, and an interactive **Streamlit dashboard**.

Detect **AF vs Non-AF rhythm** from photoplethysmography (PPG) recordings with **feature-level interpretability**.

---

## 🚀 Key Features

* **Binary rhythm classification:** AF vs Non-AF
* Shared preprocessing pipeline for notebook & app
* PPG cleaning: **interpolation**, **band-pass filtering**, **z-score normalization**
* **Window-based analysis:** 5-second overlapping segments
* **LSTM baseline** for sequence learning
* **Random Forest deployment model** on engineered features
* **SHAP explainability** for sequence & feature-based analysis
* Interactive **Streamlit dashboard**: upload, visualize, predict, and export

---

## 🛠 Project Workflow

**Deployment model flow:**

```text
Raw PPG -> Interpolation -> Band-pass Filter -> Normalization -> Beat Detection -> Windowing -> Feature Extraction -> Random Forest -> AF Prediction
```

**Notebook workflow (LSTM baseline + explainability):**

```text
Windowed PPG -> LSTM Baseline -> Evaluation -> SHAP Explainability
```

---

## 📊 Dataset

**MIMIC PERform AF dataset** (CSV format in `data/`)

* Sampling rate: ~125 Hz
* Labels: AF = `1`, Non-AF = `0`
* Folder structure expected:

```text
data/mimic_perform_af_csv/mimic_perform_af_csv/
data/mimic_perform_non_af_csv/mimic_perform_non_af_csv/
```

> ⚠️ Raw data is not included due to redistribution restrictions.

---

## 🧹 Data Preprocessing (`code/ppg_pipeline.py`)

* Linear interpolation for invalid values (`NaN`, `Inf`)
* Band-pass filter: 0.5–8.0 Hz
* Z-score normalization
* Peak detection & inter-beat interval (IBI) computation
* Windowing: 5.0-second windows, 2.5-second overlap
* At 125 Hz → 625 samples per window

---

## ✨ Engineered Features (Random Forest)

Designed to capture **pulse morphology & rhythm irregularity**:

* `signal_mean`
* `signal_std`
* `signal_range`
* `signal_energy`
* `peak_count`
* `ibi_mean`
* `ibi_std`
* `ibi_rmssd`
* `ibi_cv`

---

## 🧠 Model Architecture

### 1️⃣ LSTM Baseline

* **Input → LSTM → Dense → Output**
* Notebook reference only (not used in app)

### 2️⃣ Random Forest Deployment Model

* Trained on **engineered window features**
* Exported: `models/ppg_af_rf.joblib`
* Used in Streamlit app for real-time predictions

---

## 🔍 Explainability

**SHAP in `code/af_rnn.ipynb`:**

* GradientExplainer: LSTM sequence-level
* TreeExplainer: Random Forest
* Visualizations:

  * Global feature importance
  * Per-sample positive/negative contributions

> Rhythm irregularity features like `ibi_cv`, `ibi_std`, `ibi_mean`, `ibi_rmssd` strongly predict AF.

---

## 📈 Performance

### LSTM Baseline

| Metric                    | Value       |
| ------------------------- | ----------- |
| Accuracy                  | 0.429       |
| Non-AF Precision / Recall | 0.43 / 1.00 |
| AF Precision / Recall     | 0.60 / 0.00 |

### Random Forest (Deployment)

| Metric            | Value |
| ----------------- | ----- |
| Accuracy          | 0.90  |
| Macro F1-score    | 0.90  |
| Weighted F1-score | 0.90  |

| Class  | Precision | Recall | F1-score | Support |
| ------ | --------- | ------ | -------- | ------- |
| Non-AF | 0.93      | 0.84   | 0.88     | 1434    |
| AF     | 0.89      | 0.95   | 0.92     | 1912    |

> 🌟 Random Forest clearly outperforms the LSTM baseline.

---

## 🖥 Streamlit Dashboard (`code/ppg_app.py`)

**Capabilities:**

* Upload PPG `.csv`
* Plot **raw** & **processed signals** with detected peaks
* Segment into windows & extract features
* Predict AF probability per window
* Summarize AF percentage across recording
* Export predictions to CSV

> SHAP explanations are **not displayed in-app** yet.

---

## 📁 Repository Structure

```text
PPG-Arrhythmia-Detection/
|-- code/
|   |-- af_rnn.ipynb
|   |-- ppg_app.py
|   |-- ppg_pipeline.py
|   `-- example_generate.py
|-- requirements.txt
`-- README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Sanugiw/PPG-Arrhythmia-Detection.git
cd PPG-Arrhythmia-Detection

python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

---

## 🏃 Usage

### 1️⃣ Notebook

Open `code/af_rnn.ipynb` → run all cells to:

* Load & preprocess PPG data
* Window the signals
* Train LSTM baseline (optional)
* Train & evaluate Random Forest
* Generate SHAP explanations
* Save trained Random Forest

### 2️⃣ Streamlit App

```bash
cd code
streamlit run ppg_app.py
```

* Upload PPG CSV
* View raw & processed signals
* Get AF predictions
* Download CSV

---

## 📄 CSV Input Format

```csv
PPG
0.51
0.52
0.49
...
```

* If `PPG` column missing → **first column** is used

---

## 🛠 Tech Stack

* Python
* TensorFlow / Keras
* scikit-learn
* SHAP
* NumPy, Pandas, SciPy, Matplotlib
* Streamlit
* joblib

---

## 🔁 Reproducibility

* Shared preprocessing pipeline for training & inference
* Fixed windowing logic for notebook & app
* Serialized Random Forest ensures consistent predictions
* Notebook evaluation matches saved train/test split

---

## ✅ Summary

This project delivers a **practical, interpretable AF detection workflow**:

* Signal processing → Feature extraction → Machine learning
* Feature-level explainability using SHAP
* Interactive deployment through Streamlit

Ideal for **research, prototyping, and wearable AF screening experiments**.

---
