"""
File Name: main.py

Usage: Driver for model and simulation and uses streamlit
for UI.

Contributor: Vanny Bundick
"""

import streamlit as st

from preprocess_data import load_and_preprocess
from model import train_model, evaluate_model
#from packet_sim import run_sim

# UI Title and description of program.
st.title("AI-Based IoT Intrusion Detection System")
st.write("Hybrid IDS using SVM and Deep Learning")

X_train, X_test, y_train, y_test = load_and_preprocess()

# UI Button for model training.
if st.button("Train Model"):
    st.write("Training model...")

    # Train both SVM and DNN
    svm_model, dl_model = train_model(X_train, y_train)

    st.success

    # Evaluate performance of model
    results = evaluate_model(dl_model, X_test, y_test)

    # Display metrics
    st.subheader("Evaluation Metrics")
    st.write(f"Accuracy: {results['accuracy']:.4f}")
    st.write(f"Precision: {results['precision']:.4f}")
    st.write(f"Recall: {results['recall']:.4f}")
    st.write(f"F1 Score: {results['f1']:.4f}")

    st.subheader("Confusion Matrix")
    st.write(results["confusion_matrix"])

    st.session_state["svm"] = svm_model
    st.session_state["dl"] = dl_model
    st.session_state["X_test"] = X_test

# packet sim section once file is done 