# CS4371-IDS-and-Simulation
# AI-Based IoT Intrusion Detection System and Real-Time Packet Simulation
# Author: Vanny Bundick


## Overview:
This project implements a hybrid IDS for IoT networks using machine learning and a deep neural network technique, and focuses on a real-time packet simulation. The actual model is based on the architecture layed out in “A Novel Deep Learning-Based Intrusion Detection System for IoT Networks” (Awajan, 2023). This implementation does a great job at showing how hybrid deep learning models with Support Vector Machine (SVM) and Deep Neural Network (DNN) can detect malicious traffic effectively, a main point in the original research.


## How to Run:
1. Clone repository: git clone https://github.com/bundickles/cs4371-IDS-and-Simulation.git -> cd cs4371-IDS-and-Simulation
2. Create virtual enviornment: python -m venv venv -> Windows: venv\Scripts\activate or Linux/Mac: source venv/bin/activate
3. Install dependencies: pip install -r requirements.txt
4. Run application: python -m src.main.py


### Project Structure:
CS4371-IDS-AND-SIMULATION/

│

└── src/

    ├──__init__.py
    ├── main.py # Simple UI and main driver 
    ├── preprocess_data.py # Data loading and preprocessing 
    ├── model.py # ML + DL models 
    ├── pipeline.py # Pipeline between SVM and DNN
    ├── packet_sim.py # Real-time packet simulation logic 
    
└── data/ 

    └── nsl_kdd_train.csv 
    ├── nsl_kdd_test.csv
    
└── expected_results_example.txt # Example of results

└── requirements.txt # Dependencies 

└── IoT Intrusion Detection System - Evaluation.txt

### Dataset:
Used the standard NSL_KDD training and testing dataset and converted to binary classification:
- 0 -> normal
- 1 -> attack


### Tech Stack:
Language: Python 3.10
Libararies:
- pandas, numpy -> data processing
- scikit-learn -> SVM, preprocessing, evaluation
- Tensorfow/keras -> deep learning
- matplotlib -> bar graph for packet simulation
