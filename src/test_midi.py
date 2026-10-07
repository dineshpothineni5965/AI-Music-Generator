from music21 import converter

file_path = "dataset/midi/./maestro-v3.0.0/2011/MIDI-Unprocessed_19_R1_2011_MID--AUDIO_R1-D7_14_Track14_wav.midi"

score = converter.parse(file_path)

print("MIDI loaded successfully!")
print("Parts:", len(score.parts))
print("Duration:", score.duration.quarterLength)
print("Notes:", len(score.flatten().notes))
