"""
File Name: packet_sim.py

Usage: Meant to simulate real-time network traffic processing in
model from research paper.

Contributor: Vanny Bundick
"""

import time

"""
def run_sim: Runs real-time packet simulation.
- Each data sample or "packet" is passed through the anomoly detection
  of SVM and classified by DNN. 
- Delay parameter simulates real time packet arrival.
"""
def run_sim(svm_model, dl_model, X_test, delay = 0.3):
    results = []

    # Simulate the first 20 packets
    for i in range(20):
        # Select 1 "packet"
        packet = X_test[i].reshape(1, -1)

        # SVM Filtering: packet considered normal -> output normal at SVM level
        svm_pred = svm_model.predict(packet)[0]
        if svm_pred == 0:
            result = f"Packet {i + 1}: NORMAL (SVM)"
        else:                                           # If not normal @ SVM -> DNN filtering
            dl_pred = dl_model.predict(packet)[0][0]

            if dl_pred > 0.5:
                result = f"Packet {i + 1}: ATTACK (DNN)"
            else:
                result = f"Packet {i + 1}: NORMAL (Filtered)"
        
        results.append(result)

        # Delay to simulate real time incoming packets
        time.sleep(delay)
    
    return results
    