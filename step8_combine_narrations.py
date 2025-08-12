import os
from pydub import AudioSegment

# Path to folder containing the narration files
folder_path = "books/sample/speech_v1/test_3"

# Create 3 seconds of silence
gap = AudioSegment.silent(duration=3000)

# Get narration files sorted by index
files = [f"narration_{i}.wav" for i in range(1, 35)]

# Start with the first file
combined = AudioSegment.from_wav(os.path.join(folder_path, files[0]))

# Append remaining files with gap
for fname in files[1:]:
    sound = AudioSegment.from_wav(os.path.join(folder_path, fname))
    combined += gap + sound

# Export combined file
output_path = os.path.join(folder_path, "combined_narration.wav")
combined.export(output_path, format="wav")

print(f"Combined narration saved to: {output_path}")
