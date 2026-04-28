"""
File Name: preprocess_data.py

Usage: Loads the NSL_KDD dataset and converts its labels into
binary, 0 for normal packets, 1 for attack packets. 

Contributor: Vanny Bundick
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

"""
def load_and_preprocess: loads dataset and preprocesses it
- converts dataset labels to binary (0 = normal, 1 = attack)
- Loads separate training and testing datasets
"""
def load_and_preprocess(train_path, test_path):
    # Loads and preprocesses NSL_KDD dataset
    train_df = pd.read_csv(train_path, header = None)

    if test_path:
        test_df = pd.read_csv(test_path, header = None)
    else:
        test_df = None

    print("Original Training Set Size: ", len(train_df))
    print("Original Testing Set Size: ", len(test_df))


    train_df = train_df.sample(
        n=min(8000, len(train_df)),
        random_state=42
    )

    test_df = test_df.sample(
        n=min(2000, len(test_df)),
        random_state=42
    )

    print("Sampled Training Set Size: ", len(train_df))
    print("Sampled Testing Set Size: ", len(test_df))

    # Label is always last column of dataset
    label_col = train_df.columns[-1]

    # Converts labels to binary -> 0 = normal, 1 = attack
    def convert_label(x):
        return 0 if str(x).strip().lower() == "normal" else 1
    
    train_df[label_col] = train_df[label_col].apply(convert_label)

    if test_df is not None:
        test_df[label_col] = test_df[label_col].apply(convert_label)

    # Split features and labels
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]

    if test_df is not None:
        X_test = test_df.iloc[:, :-1]
        y_test = test_df.iloc[:, -1]
    else:
        X_test, y_test = None, None
    
    # Identify categorical columns such as TCP, HTTP, etc.
    categorical_col = X_train.select_dtypes(include = ['object']).columns

    # Encode categorical features numerically
    for col in categorical_col:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))

        # Handle unseen values in test set
        if X_test is not None:
            X_test[col] = X_test[col].astype(str).apply(
                lambda x: x if x in le.classes_ else "unknown"
            )
            le.classes_ = np.append(le.classes_, "unknown")
            X_test[col] = le.transform(X_test[col])

    # Convert to numeric
    X_train = X_train.apply(pd.to_numeric, errors = 'coerce').fillna(0)
    if X_test is not None:
        X_test = X_test.apply(pd.to_numeric, errors = 'coerce').fillna(0)

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler
