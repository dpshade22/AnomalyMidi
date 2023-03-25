import mido
import pandas as pd


def df_csv_to_midi(csv_filepath, midi_filepath):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_filepath)

    # Create a new MIDI file
    midi_file = mido.MidiFile()
    track = mido.MidiTrack()
    midi_file.tracks.append(track)

    # Iterate over the rows of the DataFrame and create MIDI messages
    for index, row in df.iterrows():
        msg = None
        if row["type"] == "note_on":
            msg = mido.Message(
                "note_on", note=row["note"], velocity=row["velocity"], time=row["time"]
            )
        elif row["type"] == "note_off":
            msg = mido.Message(
                "note_off", note=row["note"], velocity=row["velocity"], time=row["time"]
            )
        if msg is not None:
            track.append(msg)

    # Write the MIDI file to disk
    midi_file.save(midi_filepath)
