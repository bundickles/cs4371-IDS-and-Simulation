"""
File Name: model.py

Usage: Trains and evaluates model with given dataset. Trains both SVM model
for anomoly detection, and DNN (Deep Neural Network) model for final classification
of the data. It then evaluates with common metrics - acurracy, precision, recall,
f1 score, and confusion matrix.

Contributor: Vanny Bundick
"""

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

"""
def train_model: trains both SVM and DNN models.
- SVM classifies traffic as normal or potential attack.
- DNN is inspired by research paper architecture for final classification
  and is a fully connected feedforward neural network (FCFFN).
"""
def train_model(X_train, y_train):
    # Support Vector Machine
    svm_model = SVC(probability = True)
    svm_model.fit(X_train, y_train)

    # Deep Neural Network
    model = Sequential()
    model.add(Dense(16, input_dim = X_train.shape[1], activation = 'relu'))     # Input layer, 1st hidden layer
    model.add(Dense(12, activation = 'relu'))                                   # Hidden Layers
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(4, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))                                 # Output layer -> outputs probability

    # Compile model: binary classification -> binary crossentropy, optimizer -> adam (efficient gradient decent)
    model.compile(loss = 'binary_cossentropy', optimizer = 'adam', metrics = ['accuracy'])
    # Train DNN
    model.fit(X_train, y_train, epochs = 10, batch_size = 32, verbose = 0)

    return svm_model, model

"""
def evaluate_model: evaluated trained deep learning model with common metrics.
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
"""
def evaluate_model(model, X_test, y_test):
    # Predict probabilities, convert to binary
    y_pred = (model.predict(X_test) > 0.5).astype("int32")

    # Store evaluation results
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    return results