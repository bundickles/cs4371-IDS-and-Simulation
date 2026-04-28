"""
File Name: model.py

Usage: Trains and evaluates model with given dataset. Trains both SVM model
for anomoly detection, and DNN (Deep Neural Network) model for final classification
of the data. It then evaluates with common metrics - acurracy, precision, recall,
f1 score, and confusion matrix.

Contributor: Vanny Bundick
"""

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Input

"""
def train_svm: trains SVM model.
- SVM classifies traffic as normal or potential attack.
"""
def train_svm(X_train, y_train):
    # Support Vector Machine
    svm_model = LinearSVC(max_iter = 3000)              # Optimize so SVM runs faster
    svm_model.fit(X_train, y_train)

    return svm_model

"""
def train_dnn: trains DNN model.
- DNN is inspired by research paper architecture for final classification
  and is a fully connected feedforward neural network (FCFFN).
"""
def train_dnn(X_train, y_train):
    # Deep Neural Network
    dnn_model = Sequential()
    dnn_model.add(Input(shape = (X_train.shape[1],)))
    dnn_model.add(Dense(32, activation = 'relu'))     # Input layer, 1st hidden layer
    dnn_model.add(Dense(16, activation = 'relu'))                                        # Hidden Layer
    dnn_model.add(Dense(1, activation = 'sigmoid'))                                      # Output layer -> outputs probability

    # Compile model: binary classification -> binary crossentropy, optimizer -> adam (efficient gradient decent)
    dnn_model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    # Train DNN
    dnn_model.fit(X_train, y_train, epochs = 5, batch_size = 64, verbose = 0)   # Optimized for demo

    return dnn_model

"""
def evaluate_model: evaluated trained deep learning model with common metrics.
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
"""
def evaluate_model(name, y_true, y_pred):
    #Print evaluation metrics
    print(f"\n--- {name} Evaluation ---")
    print("Accuracy: ", accuracy_score(y_true, y_pred))

    print("\nClassification Report: ")
    print(classification_report(y_true, y_pred))

    print("Confusion Matrix: ")
    cm = confusion_matrix(y_true, y_pred)
    print_confusion_matrix(cm)

"""
def print_confusion_matrix: prints detailed version of confusion matrix.
"""
def print_confusion_matrix(cm):
    tn, fp, fn, tp = cm.ravel()

    print("\nConfusion Matrix")
    print("-" * 40)
    print(f"True Negatives (Correct Normal): {tn}")
    print(f"False Positives (False Alarm): {fp}")
    print(f"False Negatives (Missed Attack): {fn}")
    print(f"True Positives (Detected Attack): {tp}")
    print("-" * 40)

    total = tn + fp + fn + tp

    print(f"\nDetection Rate: {tp / (tp + fn):.2f}")
    print(f"False Alarm Rate: {fp / (fp + tn):.2f}")
