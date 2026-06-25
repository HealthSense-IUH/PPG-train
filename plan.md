# Ke hoach train model phat hien rung tam nhi tu PPG

## 1. Muc tieu

Xay dung quy trinh train model de phan loai tin hieu PPG thanh 2 nhom:

- `AF = 1`: co rung tam nhi
- `Non-AF = 0`: khong rung tam nhi

Sau khi train thanh cong, model se duoc dung de du doan tren file PPG rieng:

- `archive/huywatch-ppg-20260623-132651 (1).csv`

File Huywatch chi dung de test/inference sau khi model da duoc train, khong dua vao tap train.

## 2. Hien trang du lieu

Thu muc `archive` dang co cac nguon du lieu sau:

```text
archive/
|-- af/
|   |-- mimic_perform_af_001_data.csv
|   |-- ...
|   `-- mimic_perform_af_019_data.csv
|-- non-af/
|   |-- mimic_perform_non_af_001_data.csv
|   |-- ...
|   `-- mimic_perform_non_af_016_data.csv
|-- ppg_af_dataset.csv
`-- huywatch-ppg-20260623-132651 (1).csv
```

### 2.1 Du lieu AF

Thu muc:

```text
archive/af/
```

So file data CSV: 19.

Format chinh:

```csv
Time,PPG,resp
0,0.537634408602151,-0.0293398533007335
0.008,0.534701857282502,-0.0366748166259169
...
```

Nhan gan cho tat ca file trong thu muc nay:

```text
label = 1
```

### 2.2 Du lieu Non-AF

Thu muc:

```text
archive/non-af/
```

So file data CSV: 16.

Format chinh:

```csv
Time,PPG,resp
0,0.410557184750733,0.71709717097171
0.008,0.400782013685239,0.720787207872079
...
```

Nhan gan cho tat ca file trong thu muc nay:

```text
label = 0
```

### 2.3 File da gop san

File:

```text
archive/ppg_af_dataset.csv
```

Format:

```csv
time,ppg,resp,status
0.0,0.410557184750733,0.71709717097171,0
0.008,0.400782013685239,0.720787207872079,0
...
```

File nay co the dung de tham khao hoac train nhanh. Tuy nhien, nen uu tien train tu `archive/af` va `archive/non-af` de giu duoc thong tin moi file/recording, giup chia train/test dung hon.

### 2.4 File Huywatch de test

File:

```text
archive/huywatch-ppg-20260623-132651 (1).csv
```

Format:

```csv
device_millis,red,ir
106984,33370,191355
106993,33353,191353
...
```

Dac diem:

- Khong co cot `PPG`
- Co 2 kenh quang hoc: `red` va `ir`
- Co cot thoi gian `device_millis`
- Thoi luong xap xi 70 giay
- Sampling rate uoc tinh khoang 100 Hz

De test model, can chuyen file nay ve format signal PPG bang cach chon mot cot tin hieu, uu tien `ir`, hoac cho phep so sanh ca `red` va `ir`.

## 3. Van de can sua trong code hien tai

### 3.1 Loader trong pipeline chua khop voi folder archive

File:

```text
code/ppg_pipeline.py
```

Ham hien tai:

```python
load_mimic_perform_csv_dataset(...)
```

dang ky vong cau truc:

```text
data/mimic_perform_af_csv/mimic_perform_af_csv/
data/mimic_perform_non_af_csv/mimic_perform_non_af_csv/
```

Trong khi du lieu thuc te dang nam o:

```text
archive/af/
archive/non-af/
```

Can them loader moi hoac sua loader hien tai de doc dung cau truc nay.

### 3.2 App Streamlit chua doc tot file Huywatch

File:

```text
code/ppg_app.py
```

App hien tai doc:

```python
ppg_raw = df["PPG"].values if "PPG" in df.columns else df.iloc[:, 0].values
```

Voi file Huywatch, cot dau tien la `device_millis`, khong phai tin hieu PPG. Neu giu logic nay, app se lay sai tin hieu.

Can sua logic:

- Neu co cot `PPG`: dung `PPG`
- Neu co cot `ppg`: dung `ppg`
- Neu co cot `ir`: dung `ir`
- Neu co cot `red`: dung `red`
- Neu co `device_millis`: tinh sampling rate tu thoi gian

### 3.3 Chua co model da train

App dang tim:

```text
models/ppg_af_rf.joblib
```

Neu file nay chua ton tai, can train va save model truoc.

## 4. Pipeline xu ly du lieu

Quy trinh de train:

```text
CSV recordings
-> load signal va label
-> tinh sampling rate
-> preprocess PPG
-> segment thanh window
-> extract features
-> train Random Forest
-> evaluate
-> save model
```

### 4.1 Load signal

Voi file MIMIC trong `archive/af` va `archive/non-af`:

- Tin hieu: cot `PPG`
- Thoi gian: cot `Time`
- Sampling rate: tinh tu `Time`

Cong thuc:

```text
dt = Time[1] - Time[0]
fs = round(1 / dt)
```

Voi du lieu hien tai, `dt = 0.008`, nen:

```text
fs = 125 Hz
```

### 4.2 Preprocess

Dung cac buoc trong `ppg_pipeline.py`:

1. Noi suy gia tri khong hop le:

```text
NaN/Inf -> linear interpolation
```

2. Loc band-pass:

```text
0.5 Hz den 8.0 Hz
```

3. Chuan hoa z-score:

```text
mean = 0
std = 1
```

### 4.3 Chia window

Tham so mac dinh:

```text
window_sec = 5.0
overlap_sec = 2.5
```

Voi sampling rate 125 Hz:

```text
window length = 625 samples
step = 312 hoac 313 samples
```

Moi recording se duoc tach thanh nhieu window. Moi window lay cung label voi recording goc.

### 4.4 Trich xuat dac trung

Dung cac feature hien co:

```text
signal_mean
signal_std
signal_range
signal_energy
peak_count
ibi_mean
ibi_std
ibi_rmssd
ibi_cv
```

Y nghia:

- `signal_*`: dac trung bien do va nang luong tin hieu
- `peak_count`: so nhip/phach trong window
- `ibi_*`: do bat thuong cua khoang cach giua cac beat
- `ibi_rmssd`, `ibi_cv`: dac trung quan trong cho tinh bat thuong nhip, co lien quan den AF

## 5. Chien luoc train model

### 5.1 Model de xuat

Dung model chinh:

```text
RandomForestClassifier
```

Ly do:

- Phu hop voi feature dang bang
- Train nhanh
- It can GPU
- De save/load bang `joblib`
- Co the giai thich bang feature importance hoac SHAP
- Dang khop voi Streamlit app hien tai

Thong so ban dau de thu:

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
```

### 5.2 Chia train/test dung cach

Khong nen chia random tat ca window ngay tu dau, vi cac window tu cung mot recording rat giong nhau. Neu mot recording bi cat thanh window roi vua nam trong train vua nam trong test, ket qua co the bi ao.

Nen chia theo file/recording:

```text
recording-level split
```

Vi du:

```text
80% recording train
20% recording test
```

Can stratify theo label de train/test deu co AF va Non-AF.

### 5.3 Output model

Sau train, luu:

```text
models/ppg_af_rf.joblib
```

Nen luu them metadata:

```text
models/ppg_af_rf_metadata.json
```

Noi dung metadata:

```json
{
  "model_type": "RandomForestClassifier",
  "labels": {
    "0": "Non-AF",
    "1": "AF"
  },
  "window_sec": 5.0,
  "overlap_sec": 2.5,
  "bandpass": [0.5, 8.0],
  "features": [
    "signal_mean",
    "signal_std",
    "signal_range",
    "signal_energy",
    "peak_count",
    "ibi_mean",
    "ibi_std",
    "ibi_rmssd",
    "ibi_cv"
  ]
}
```

## 6. Danh gia model

Can bao cao cac metric:

```text
accuracy
precision
recall
f1-score
confusion matrix
ROC-AUC
```

Uu tien theo doi:

- Recall cua class AF: model co bo sot AF khong
- Precision cua class AF: model co bao dong gia nhieu khong
- F1-score macro: can bang giua 2 class

Nen luu report:

```text
reports/rf_evaluation.txt
reports/rf_confusion_matrix.csv
reports/rf_feature_importance.csv
```

## 7. Test voi file Huywatch

### 7.1 Doc file Huywatch

File co format:

```csv
device_millis,red,ir
```

Can xu ly:

```text
time_sec = (device_millis - device_millis[0]) / 1000
fs = 1 / median(diff(time_sec))
signal = ir
```

Neu `ir` bi loi hoac chat luong kem, thu `red`.

### 7.2 Chay inference

Quy trinh:

```text
Huywatch CSV
-> lay cot ir
-> tinh fs tu device_millis
-> preprocess
-> segment
-> extract features
-> model.predict_proba
-> tong hop ket qua
```

Output mong muon:

```text
Window_Index
Start_Time_Sec
End_Time_Sec
AF_Prediction
AF_Probability
```

Tong hop:

```text
AF windows percentage
Mean AF probability
Max AF probability
```

Can luu file:

```text
outputs/huywatch_predictions.csv
```

## 8. Cap nhat Streamlit app

Sau khi train xong, cap nhat `code/ppg_app.py` de:

1. Load model tu:

```text
models/ppg_af_rf.joblib
```

2. Cho upload file CSV.

3. Tu nhan dien format:

```text
MIMIC: Time, PPG, resp
Merged dataset: time, ppg, resp, status
Huywatch: device_millis, red, ir
```

4. Neu la Huywatch:

- Cho nguoi dung chon `ir` hoac `red`
- Mac dinh chon `ir`
- Tinh sampling rate tu `device_millis`

5. Hien thi:

- Raw signal
- Processed signal
- Peaks detected
- Feature table
- Prediction tung window
- Tong hop AF probability

## 9. Cac file nen tao hoac sua

### 9.1 Tao script train

File moi:

```text
code/train_af_model.py
```

Nhiem vu:

- Doc `archive/af` va `archive/non-af`
- Chia train/test theo recording
- Preprocess va extract feature
- Train Random Forest
- Evaluate
- Save model vao `models/ppg_af_rf.joblib`
- Save report vao `reports/`

### 9.2 Tao script test Huywatch

File moi:

```text
code/predict_huywatch.py
```

Nhiem vu:

- Doc file Huywatch
- Chon cot `ir`
- Tinh sampling rate
- Chay model
- Luu ket qua vao `outputs/huywatch_predictions.csv`

### 9.3 Sua pipeline

File:

```text
code/ppg_pipeline.py
```

Can them:

- Loader cho `archive/af` va `archive/non-af`
- Ham doc CSV linh hoat cho nhieu format
- Ham tinh sampling rate tu `Time`, `time`, hoac `device_millis`

### 9.4 Sua app

File:

```text
code/ppg_app.py
```

Can sua:

- Khong lay cot dau tien mot cach may moc
- Ho tro file Huywatch
- Dung sampling rate thuc te khi preprocess/windowing

## 10. Thu tu thuc hien de xuat

1. Cap nhat `ppg_pipeline.py` de doc dung du lieu.
2. Tao `code/train_af_model.py`.
3. Train Random Forest va luu model.
4. Kiem tra evaluation report.
5. Tao `code/predict_huywatch.py`.
6. Chay test tren file Huywatch.
7. Cap nhat `ppg_app.py` de upload/test truc quan.
8. Neu ket qua Huywatch bat thuong, kiem tra lai chat luong tin hieu `ir/red`, sampling rate, va tham so peak detection.

## 11. Rui ro va luu y

### 11.1 Khac biet domain giua MIMIC va Huywatch

Du lieu train MIMIC va du lieu Huywatch co the khac nhau ve:

- Cam bien
- Vi tri deo
- Cuong do anh sang
- Tan so lay mau
- Nhieu do chuyen dong
- Scale tin hieu

Model train tren MIMIC co the khong tong quat tot ngay tren Huywatch. Can xem day la prototype/research, chua phai cong cu chan doan y khoa.

### 11.2 Huywatch khong co label

File Huywatch khong co nhan that. Vi vay chi co the xem du doan, khong tinh accuracy tren file nay duoc.

### 11.3 Can tranh leakage

Neu chia train/test theo window random, ket qua co the cao gia. Phai chia theo recording/file.

### 11.4 Chat luong peak detection

Feature IBI phu thuoc vao viec detect peak. Neu tin hieu Huywatch nhieu hoac nguoc pha, can:

- Kiem tra plot raw/processed
- Thu `ir` va `red`
- Dieu chinh nguong detect peak
- Co the can invert signal neu peak bi dao

## 12. Ket qua mong doi

Sau khi hoan thanh:

```text
models/ppg_af_rf.joblib
reports/rf_evaluation.txt
outputs/huywatch_predictions.csv
```

Streamlit app co the upload file Huywatch va hien thi:

- Bieu do tin hieu
- Cac window duoc du doan AF/Non-AF
- Xac suat AF trung binh
- Ti le window nghi AF
- File CSV prediction de tai ve

