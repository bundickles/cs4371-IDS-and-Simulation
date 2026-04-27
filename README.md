# CS4371-IDS-and-Simulation
# AI-Based IoT Intrusion Detection System and Real-Time Packet Simulation
# Author: Vanny Bundick


## Overview:
This project implements a hybrid IDS for IoT networks using machine learning and a deep neural network technique, and focuses on a real-time packet simulation. The actual model is based on the architecture layed out in “A Novel Deep Learning-Based Intrusion Detection System for IoT Networks” (Awajan, 2023). This implementation does a great job at showing how deep learning models can detect malicious traffic effectively, which was a main point in the original research.


## How to Run:
1.Clone repository: git clone https://github.com/bundickles/cs4371-IDS-and-Simulation.git -> cd cs4371-IDS-and-Simulation
2.Create virtual enviornment: python -m venv venv -> Windows: venv\Scripts\activate or Linux/Mac: source venv/bin/activate
3.Install dependencies: pip install -r requirements.txt
4.Run application: streamlit run main.py


### Project Structure:
CS4371-IDS-AND-SIMULATION/ │ ├── main.py # Streamlit UI and main driver ├── data_preprocessing.py # Data loading and preprocessing ├── model.py # ML + DL models ├── simulation.py # Real-time packet simulation logic ├── requirements.txt # Dependencies └── data/ └── nsl_kdd_train.csv ├── nsl_kdd_test.csv


### Dataset:
Used the standard NSL_KDD dataset and converty to binary classification:
-0 -> normal
-1 -> attack


### Tech Stack:
Language: Python 3.10
Libararies:
-pandas, numpy -> data processing
-scikit-learn -> SVM, preprocessing, evaluation
-Tensorfow/keras -> deep learning
- Streamlit -> interactive UI
