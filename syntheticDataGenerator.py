import os
import pandas as pd
import numpy as np
import random
import mido
from pathlib import Path


import os
import mido
import pandas as pd

MAX_TIME_STEPS = 500


def getMidiDF(fileLoc: str):
    # Store the data from the MIDI file in a list of dictionaries
    mid = mido.MidiFile(fileLoc, clip=True)
    midi_data = []
    for i, track in enumerate(mid.tracks):
        addTime = 0
        for msg in track:
            if msg.type == "note_on":
                addTime += msg.time
                midi_data.append(
                    {
                        "track": i,
                        "note": msg.note,
                        "velocity": msg.velocity,
                        "time": msg.time,
                    }
                )

    # Convert the list of dictionaries to a Pandas DataFrame
    df = pd.DataFrame(midi_data)

    # Check if the DataFrame has fewer than MAX_TIME_STEP rows
    if df.shape[0] < MAX_TIME_STEPS:
        # Calculate the number of rows to pad with zeros
        num_rows_to_pad = MAX_TIME_STEPS - df.shape[0]

        # Create a new DataFrame with the necessary number of rows and columns
        padded_df = pd.DataFrame(
            np.zeros((num_rows_to_pad, df.shape[1])), columns=df.columns
        )

        # Concatenate the original DataFrame and the padded DataFrame
        df = pd.concat([df, padded_df])

    df = df[:MAX_TIME_STEPS]

    df.to_csv(
        os.path.join("csvMidiData", fileLoc.split("\\")[-1].split(".")[0] + ".csv"),
        index=False,
    )
    return df


def getAllMidiDFs(root_folder):
    # Create the output directory if it doesn't exist
    os.makedirs("midiData", exist_ok=True)

    for root, dirs, files in os.walk(root_folder):
        for filename in files:
            if filename.endswith(".mid"):
                fileLoc = os.path.join(root, filename)
                getMidiDF(fileLoc)


# getAllMidiDFs("./midiData")


def add_anomaly_column(csv_file_path, master_changes_file_path, percent_modified):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)
    df = df.head(500)
    # Add an "anomaly" column filled with zeros
    df["anomaly"] = 0

    # Create a new DataFrame to store the changes made
    changes_df = pd.DataFrame(columns=["filename", "row", "column", "amount"])

    # Iterate over each row in the DataFrame and modify a random column for a random subset of rows
    num_anomalies = int(len(df) * percent_modified)
    for i in random.sample(range(len(df)), num_anomalies):
        # Modify a random column
        cols = random.sample(["note", "velocity", "time"], k=random.randint(1, 3))
        for col in cols:
            options = df[col].unique()

            if col == "time":
                # If "time" column is selected, choose a unique time value from the DataFrame
                unique_times = df["time"].unique()
                unique_times = unique_times[unique_times != df.at[i, col]]
                new_time = random.choice(unique_times)
                amount = new_time - df.at[i, col]
                df.at[i, col] = new_time
            elif col == "velocity":
                # If "velocity" column is selected, choose a random value from a list and add it to the velocity value
                velocity = int(df.at[i, col])
                offset = (
                    random.randint(-80, -20)
                    if velocity >= 80
                    else random.randint(20, 80)
                )
                new_velocity = min(max(velocity + offset, 0), 127)
                amount = new_velocity - velocity
                df.at[i, col] = new_velocity
            elif col == "note":
                # If "note" column is selected, choose a random value from a list and add it to the note value
                note = int(df.at[i, col])
                offset = random.choice(
                    [-13, -12, -11, -6, -5, -3, -1, 1, 3, 5, 6, 11, 12, 13]
                )
                new_note = min(max(note + offset, 0), 127)
                amount = new_note - note
                df.at[i, col] = new_note
            else:
                # For other columns, choose a random value from the options and set the value to that
                new_value = random.choice(options)
                amount = new_value - df.at[i, col]
                df.at[i, col] = new_value

            # Log the changes made
            row = i + 2
            new_changes_df = pd.DataFrame(
                {
                    "filename": f"{csv_file_path.split('/')[-1]}{i}",
                    "row": row,
                    "column": col,
                    "amount": amount,
                },
                index=[i],
            )
            changes_df = pd.concat([changes_df, new_changes_df])

        # Set the "anomaly" flag for the modified row to 1
        df.at[i, "anomaly"] = len(cols)

    # Write the modified DataFrame to a new CSV file
    csv_filename = os.path.splitext(os.path.basename(csv_file_path))[0]
    modified_csv_file_path = os.path.join(
        "anomalous",
        f"{csv_filename}_modified{str(percent_modified)[0] + str(round(percent_modified, 2))[-2:]}.csv",
    )

    df.to_csv(modified_csv_file_path, index=False)

    # Append the changes made to the master changes CSV file
    changes_df.to_csv(master_changes_file_path, mode="a", header=False, index=False)
    print(
        os.path.join(
            "anomalous",
            f"{csv_filename}_modified{str(percent_modified)[0] + str(round(percent_modified, 2))[-2:]}.csv",
        )
    )


def generate_anomalous_data(csv_data_dir, master_changes_file):
    # Walk through every file in the csvMidiData directory
    for root, dirs, files in os.walk(csv_data_dir):
        for file in files:
            if file.endswith(".csv"):
                # Call the add_anomaly_column function six times, with percent_modified increasing by 0.1 each time
                percent_modified = 0.78
                for i in range(4):
                    add_anomaly_column(
                        os.path.join(root, file), master_changes_file, percent_modified
                    )
                    percent_modified += 0.06


def set_anomaly_column_none(csv_file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)
    df = df.head(500)

    # Add an "anomaly" column filled with zeros
    df["anomaly"] = 0

    # Define ranges of sensible note and velocity changes
    note_change_range = range(-12, 13)
    velocity_change_range = range(-30, 31)

    # Randomly select note and velocity changes
    note_change = np.random.choice(note_change_range)
    velocity_change = np.random.choice(velocity_change_range)

    # Clip note and velocity changes at MIDI bounds
    df["note"] = np.clip(df["note"] + note_change, 0, 127)
    df["velocity"] = np.clip(df["velocity"] + velocity_change, 0, 127)

    # Write the modified DataFrame to a new CSV file
    csv_filename = os.path.splitext(os.path.basename(csv_file_path))[0]
    modified_csv_file_path = os.path.join(
        "anomalous",
        f"{csv_filename}_modified-N{note_change}V{velocity_change}.csv",
    )

    df.to_csv(modified_csv_file_path, index=False)


def generate_nonanomalous_data(csv_data_dir):
    # Walk through every file in the csv_data_dir directory
    for root, dirs, files in os.walk(csv_data_dir):
        for file in files:
            if file.endswith(".csv"):
                for _ in range(6):
                    set_anomaly_column_none(os.path.join(root, file))


generate_anomalous_data("csvMidiData", "masterChanges.csv")


# generate_nonanomalous_data("csvMidiData")
