# ⚡ Electrical Fault Detection & Classification using DNN

This project implements an end-to-end Deep Learning pipeline for detecting and classifying electrical faults based on voltage and current sensor data. The goal is to build a professional and accurate model capable of identifying six types of electrical faults, as well as detecting when there is no fault.

---

## 📌 Problem Statement

Electrical faults in power systems can cause serious damage to infrastructure and affect stability. Quick and accurate fault classification is essential for timely maintenance and safety.

In this project, we classify:
- No Fault
- LG Fault (Line to Ground)
- LL Fault (Line to Line)
- LLG Fault (Line to Line to Ground)
- LLL Fault (Three Phase)
- LLLG Fault (Three Phase to Ground)

---

## 🧠 Model Used

A **Deep Neural Network (DNN)** with multiple hidden layers is built using **TensorFlow/Keras**. The architecture includes:

- Input Layer based on voltage and current values
- 2 Hidden Dense Layers with ReLU activation
- Dropout layers for regularization
- Output Layer with Softmax activation (6-class classification)

---

## 🧪 Dataset

- Source: [Kaggle] Voltage and Current Measurements
- Features: `Ia`, `Ib`, `Ic`, `Va`, `Vb`, `Vc` (line currents and voltages)
- Target labels constructed from binary columns `A`, `B`, `C`, `G`:
  - `[0 0 0 0]` → No Fault  
  - `[1 0 0 1]`, `[0 1 0 1]`, `[0 0 1 1]` → LG Fault  
  - `[1 1 0 0]`, etc. → LL Fault  
  - and so on...

---

## 📊 Data Preprocessing

- Removed irrelevant columns
- Mapped binary fault codes to categorical fault labels
- Performed stratified train-test split to maintain class balance
- Scaled features using StandardScaler
- Removed significant outliers using visual inspection via boxplots

---

## 🏗️ Model Architecture

```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(6, activation='softmax')
])
