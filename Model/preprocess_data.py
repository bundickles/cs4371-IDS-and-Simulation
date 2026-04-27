"""
File Name: preprocess_data.py

Usage: Loads the NSL_KDD dataset and converts its labels into
binary, 0 for normal packets, 1 for attack packets. 

Contributor: Vanny Bundick

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Loads and uses NSL_KDD dataset
def load_and_prepocess(path = "data/nsl_kdd_sample.csv"):
    df = pd.read_csv(path)

    # Converts labels to binary -> 0 = normal, 1 = attack
    df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

    # Encodes the categorical columns of data set
    categorical_col = df.select_dtypes(include = ['object']).columns
    for col in categorical_col:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

        X = df.drop('label', axis = 1)
        Y = df['label']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return train_test_split(X_scaled, Y, test_size=0.2, random_state=42)