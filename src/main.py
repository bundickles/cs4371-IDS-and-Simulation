"""
File Name: main.py

Usage:
-Main driver: loads dataset -> train and evaluate models -> run packet simulation
-Output simple UI in terminal.

Contributor: Vanny Bundick
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'                # Hides warnings from tensorflow for demo
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from src.preprocess_data import load_and_preprocess
from src.model import train_svm, train_dnn, evaluate_model
from src.packet_sim import run_sim
from src.pipeline import classify_packet, evaluate_pipeline

def main():
    # Main title UI
    print("=" * 50)
    print(" IoT Intrusion Detection System")
    print(" Support Vector Machine and Deep Neural Network ")
    print("=" * 50)

    # Step 1: load dataset
    print("\nLoading Dataset...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess(
        "data/NSL_KDD_Train.csv",
        "data/NSL_KDD_Test.csv"
    )

    # Step 2: Train models and evaluate
    print("\nTraining SVM (Anomaly Detecion Layer)...")
    svm_model = train_svm(X_train, y_train)

    print("\nTraining DNN (Deep Classification Layer)...")
    dnn_model = train_dnn(X_train, y_train)

    svm_preds = svm_model.predict(X_test)
    evaluate_model("SVM", y_test, svm_preds)

    dnn_preds = (dnn_model.predict(X_test) > 0.5).astype(int)
    evaluate_model("DNN", y_test, dnn_preds)

    pipeline_preds = evaluate_pipeline(X_test, svm_model, dnn_model, scaler)
    evaluate_model("Hybrid Pipeline (SVM + DNN)", y_test, pipeline_preds)

    # Step 3: Run packet simulation
    print("\nStarting Real-Time Packet Simulation...\n")
    run_sim(
        svm_model,
        dnn_model,
        scaler,
        X_train.shape[1],
        classify_packet
    )

if __name__ == "__main__":
    main()

