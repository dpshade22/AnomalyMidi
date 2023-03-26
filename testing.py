import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras import backend as K

from src.syntheticDataGenerator import getMidiDF
from src.midiProcessing import csvToMidi

# Replace 'model_file.h5' with the name of your H5 file
model_file = "./col-model.h5"

# Load the model from the H5 file
model = load_model(model_file)


def predict_anomalies_col(model, file, sequence_length=500):
    df = pd.read_csv(file, usecols=[1, 2, 3], index_col=False)
    print(df.columns)
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
        df = pd.concat([df, padding], ignore_index=True)

    input_data = df.iloc[:sequence_length, :].values.reshape(1, sequence_length, -1)
    print(input_data.shape)
    predictions = model.predict(input_data)
    predictions[predictions < 0] = 0

    # Round predictions to the nearest integer
    rounded_predictions = np.round(predictions).flatten()

    # Calculate the number of anomalies by summing up the rounded predictions
    num_anomalies = int(np.sum(rounded_predictions))

    return num_anomalies, rounded_predictions


# getMidiDF("Alive again acc.mid")

new_file = "anomalous/waldstein_3_modified066.csv"
df = pd.read_csv(new_file, usecols=[1, 2, 3, 4], index_col=False)


sum_of_fourth_column = df.iloc[:, 3].sum()
print(df.head(50))

num_anomalies, column = predict_anomalies_col(model, new_file)

columnDF = pd.DataFrame(column, columns=["predicted_anomaly"])

print(f"Predicted number of anomalies: {num_anomalies}")

print(f"Real number of anomalies: {sum_of_fourth_column}")
df = pd.concat([df, columnDF], axis=1).head(100)

print(df[(df["anomaly"] == df["predicted_anomaly"]) & (df["anomaly"] != 0)])

# csvToMidi("anomalous/elise_modified01.csv", "elise_modified01.mid")
# csvToMidi("best_solution.csv", "best_solution.mid")
