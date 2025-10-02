import os
from pydub import AudioSegment

import os
print("Current working directory:", os.getcwd())

def convert_mp3_to_wav(input_dir: str, output_dir: str = None):
    """
    Convert all .mp3 files in a directory to .wav files.

    :param input_dir: Directory where mp3 files are stored
    :param output_dir: Directory where wav files will be saved (default: same as input)
    """
    if output_dir is None:
        output_dir = input_dir  # save in same directory

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith(".mp3"):
            mp3_path = os.path.join(input_dir, file_name).replace("\\", "/")
            wav_name = os.path.splitext(file_name)[0] + ".wav"
            wav_path = os.path.join(output_dir, wav_name).replace("\\", "/")

            # Load mp3 and export as wav
            try:
                audio = AudioSegment.from_mp3(mp3_path)
                audio.export(wav_path, format="wav")
                print(f"✅ Converted: {mp3_path} -> {wav_path}")
            except Exception as e:
                print(f"❌ Error converting {file_name}: {e}")

# Example usage:
convert_mp3_to_wav("reference_wav_audio_samples/hindi_female")
