import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    GlobalAveragePooling1D,
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_and_preprocess_data(file_list, sequence_length=500):
    data = []
    targets = []

    for file in file_list:
        df = pd.read_csv(file, usecols=[1, 2, 3, 4])
        scaler = MinMaxScaler()
        df[["note", "velocity", "time"]] = scaler.fit_transform(
            df[["note", "velocity", "time"]]
        )

        # Pad the input data if the number of notes is less than the sequence length
        if len(df) < sequence_length:
            padding = pd.DataFrame(
                np.zeros((sequence_length - len(df), 3)),
                columns=["note", "velocity", "time"],
            )
            df = pd.concat(
                [df[["note", "velocity", "time"]], padding], ignore_index=True
            )

        data.append(df.iloc[:sequence_length, :-1].values)
        targets.append(df["anomaly"].sum())

    return np.array(data), np.array(targets)


directory = "./anomalous"  # Replace with the path to your directory
anomalous_file_list = []  # Initialize an empty list

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory, filename)
        anomalous_file_list.append(file_path)

sequence_length = 500  # Updated sequence length
data, targets = load_and_preprocess_data(anomalous_file_list, sequence_length)

X_train, X_test, y_train, y_test = train_test_split(
    data, targets, test_size=0.2, random_state=42
)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

model = Sequential(
    [
        LSTM(
            64,
            activation="tanh",
            input_shape=(sequence_length, 3),
            return_sequences=True,
        ),  # Updated activation
        GlobalAveragePooling1D(),  # Updated layer
        Dense(1, activation="linear"),
    ]
)

model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))


def predict_anomalies(model, file, sequence_length=10):
    df = pd.read_csv(file)
    scaler = MinMaxScaler()
    df[["note", "velocity", "time"]] = scaler.fit_transform(
        df[["note", "velocity", "time"]]
    )

    input_data = []
    for i in range(len(df) - sequence_length):
        input_data.append(df.iloc[i : i + sequence_length, :-1].values)

    input_data = np.array(input_data)
    predictions = model.predict(input_data)
    predictions[predictions < 0] = 0
    return int(np.round(np.sum(predictions)))


new_file = "new_file.csv"
num_anomalies = predict_anomalies(model, new_file)
print(f"Number of anomalies: {num_anomalies}")
