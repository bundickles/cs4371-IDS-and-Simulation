from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def train_model(X_train, y_train):
    svm_model = SVC(probability = True)
    svm_model.fit(X_train, y_train)

    model = Sequential()
    model.add(Dense(16, input_dim = X_train.shape[1], activation = 'relu'))
    model.add(Dense(12, activation = 'relu'))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(4, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(loss = 'binary_cossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.fit(X_train, y_train, epochs = 10, batch_size = 32, verbose = 0)

    return svm_model, model


def evaluate_model(model, X_test, y_test):
    y_pred = (model.predict(X_test) > 0.5).astype("int32")

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    return results