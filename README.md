# Fraud_Detection_System

A machine learning based fraud detection system that identifies suspicious financial transactions using Random Forest and Isolation Forest algorithms.

## 🚀 Features
- Detects fraudulent transactions in real-time
- Uses both supervised and unsupervised ML models
- Analyzes transaction amount, location, time, and merchant behavior
- Classifies transactions as SAFE, FLAGGED, or BLOCKED
- Generates detailed risk factor reports

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Scikit-learn, Pandas, NumPy
- **Algorithms:** Random Forest Classifier, Isolation Forest

## ⚙️ How to Run

1. Clone the repository
   git clone https://github.com/TVARDHINI/Fraud_Detection_System.git

2. Install dependencies
   pip install -r requirements.txt

3. Run the system
   python fraud_detection.py

## 📊 How It Works
- **Random Forest** → Supervised classification to detect known fraud patterns
- **Isolation Forest** → Unsupervised anomaly detection for unusual transactions
- **Feature Engineering** → Extracts time, location, amount, and merchant-based features
