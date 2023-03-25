import pandas as pd
import mido


def csvToMidi(csv_filepath, midi_filepath):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_filepath)
    df["track"] = 1

    # Calculate the average for each column
    avg_note = df["note"].mean()
    avg_velocity = df["velocity"].mean()
    avg_time = df["time"].mean()

    # Custom function to replace values with average if they are outside the [0, 127] range
    def replace_with_avg(value, avg):
        if value < 0 or value > 127:
            return avg
        return value

    # Apply the custom function to the note, velocity, and time columns
    df["note"] = df["note"].apply(lambda x: replace_with_avg(x, avg_note))
    df["velocity"] = df["velocity"].apply(lambda x: replace_with_avg(x, avg_velocity))
    df["time"] = df["time"].apply(lambda x: replace_with_avg(x, avg_time))

    # Create a new MIDI file
    midi_file = mido.MidiFile()
    track = mido.MidiTrack()
    midi_file.tracks.append(track)

    # Iterate over the rows of the DataFrame and create MIDI messages
    for index, row in df.iterrows():
        msg = mido.Message(
            "note_on",
            note=int(row["note"]),
            velocity=int(row["velocity"]),
            time=int(row["time"]),
        )
        track.append(msg)

    # Write the MIDI file to disk
    midi_file.save(midi_filepath)
