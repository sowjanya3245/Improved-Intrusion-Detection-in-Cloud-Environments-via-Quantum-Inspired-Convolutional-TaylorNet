# QuCFTnet: Quantum-Inspired Convolutional Feature Transformation Network
This repository provides the implementation of QuCFTnet, a quantum-inspired CNN for binary network intrusion detection using the CIC-IDS2018 dataset.

Pipeline

CIC-IDS2018 → Preprocessing → Normalization → Quantum-Inspired Features → Taylor Transformation → CNN → Softmax

Models
QuCFTnet
CNN
BiLSTM
QNN
Metrics

Accuracy, Precision, Recall, F1-score, MCC, Kappa, ROC-AUC, PR-AUC, Log Loss, and Confusion Matrix.

Run
pip install -r requirements.txt
python experiments/train_qucftnet.py --data_path "C:/path/to/CIC-IDS2018"

