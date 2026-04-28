"""
File Name: packet_sim.py

Usage: Meant to simulate real-time network traffic processing in
model from research paper.

Contributor: Vanny Bundick
"""
import numpy as np
import time
import random

"""
def print_packet: output of the IoT traffic simulation.
"""
def print_packet(i, svm_pred, dnn_pred):
    if svm_pred == 0:
        label = "NORMAL (SVM)"
    else:
        label = "ATTACK (SVM)"
        if dnn_pred == 1:
            label += " - CONFIRMED (DNN)"
        else:
            label += " - REJECTED (DNN)"
        
    print(f"[Packet {i}] {label}")

"""
def generate_packet: simulates IoT network traffic.
- Mixed packet -> 80% normal, 20% anomalous
- Mimics real-world network to test efficacy of paper's IDS design.
"""
def generate_packet(num_features):
    # Normal traffic -> centered near 0
    if random.random() < 0.8:
        return np.random.normal(loc = 0.0, scale = 1.0, size = num_features)
    
    # Attack traffic -> deviates more
    return np.random.normal(loc = 3.0, scale = 2.0, size = num_features)

"""
def run_sim: Runs real-time packet simulation.
- Each data sample or "packet" is passed through the anomoly detection
  of SVM and classified by DNN in pipeline.
- Delay parameter simulates real time packet arrival.
"""
def run_sim(svm_model, dnn_model, scaler, feature_size, classify_packet):
    print("\n--- Real-Time IoT Traffic Simulation ---\n")

    # Counters
    total = 0
    normal = 0
    svm_attacks = 0
    dnn_attacks = 0

    # Simulate the first 20 packets
    for i in range(20):
        # Generate packet
        packet = generate_packet(feature_size)
        
        svm_pred, dnn_pred = classify_packet(packet, svm_model, dnn_model, scaler)
        total += 1

        if svm_pred == 0:
            normal += 1
        else:
            svm_attacks += 1
            if dnn_pred == 1:
                dnn_attacks += 1


        print_packet(i, svm_pred, dnn_pred)
        print(
            f"[STATS] Total: {total} | Normal: {normal} | "
            f"SVM: {svm_attacks} | DNN: {dnn_attacks}"
        )

        # Simulate network delay 
        time.sleep(0.2)
    