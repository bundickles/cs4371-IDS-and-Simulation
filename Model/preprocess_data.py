"""
File Name: preprocess_data.py

Usage: Loads the NSL_KDD dataset and converts its labels into
binary, 0 for normal packets, 1 for attack packets. 

Contributor: Vanny Bundick
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

"""
def load_and_preprocess: loads dataset and preprocesses it
- converts dataset labels to binary (0 = normal, 1 = attack)
- Loads separate training and testing datasets
"""
def load_and_preprocess(train_path, test_path):
    # Loads and preprocesses NSL_KDD dataset
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Converts labels to binary -> 0 = normal, 1 = attack
    train_df['label'] = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    test_df['label'] = test_df['label'].apply(lambda x: 0 if x == 'normal' else 1)

    # Encodes the categorical columns of data set
    categorical_col = train_df.select_dtypes(include = ['object']).columns
    encoders = {}

    for col in categorical_col:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
       
        encoders[col] = le

    # Split features and labels
    X_train = train_df.drop('label', axis = 1)
    y_train = train_df['label']

    X_test = test_df.drop('label', axis = 1)
    y_test = test_df['label']
     

    # Normalize train features so values are on similar scale -> for performance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train. y_test