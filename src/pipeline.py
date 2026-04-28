"""
File Name: pipeline.py

Usage: For pipelining the two models used for this research, the
SVM and DNN models.

Contributor: Vanny Bundick
"""
import numpy as np

"""
def classify_packet: Used to pipeline packet filtering.
1.SVM -> anomaly detection
2.Normal(SVM) -> break
3.If suspiciou -> pass to DNN for deeper filtering
"""
def classify_packet(packet, svm_model, dnn_model, scaler):
    # Reshape packet for model input
    packet = packet.reshape(1, -1)
    packet = scaler.transform(packet)

    # Step 1: SVM Filter
    svm_pred = svm_model.predict(packet)[0]

    # Step 2: if normal -> break
    if svm_pred == 0:
        return 0, None             # Normal

    # Step 3: DNN filtering
    dnn_pred = dnn_model.predict(packet, verbose = 0)[0][0]
    dnn_pred = 1 if dnn_pred > 0.5 else 0

    return svm_pred, dnn_pred

"""
def evaluate_pipeline: Evaluates fully ran pipeline on dataset
"""
def evaluate_pipeline(X_test, svm_model, dnn_model, scaler):
    preds = []

    for i in range(len(X_test)):
        packet = X_test[i].reshape(1, -1)

        svm_pred, dnn_pred = classify_packet(packet, svm_model, dnn_model, scaler)
        final_pred = svm_pred if svm_pred == 0 else dnn_pred
        preds.append(final_pred)

    return np.array(preds)