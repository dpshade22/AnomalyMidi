import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras import backend as K


def precision(y_true, y_pred):
    """Precision metric."""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


# Register the custom metric function with Keras
tf.keras.metrics.Precision = precision
# Replace 'model_file.h5' with the name of your H5 file
model_file = "./model/model.h5"

# Load the model from the H5 file
model = load_model(model_file)


def predict_anomalies(model, file, sequence_length=500):
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
    return int(np.round(np.sum(predictions)))


new_file = "anomalous/waldstein_3_modified0.csv"
num_anomalies = predict_anomalies(model, new_file)
print(f"Number of anomalies: {num_anomalies}")
