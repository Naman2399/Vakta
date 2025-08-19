import os
from TTS.api import TTS

SOURCE_DIR = "books/sample/musical_prompt/test_10"
TARGET_WAV = "books/sample/HarishBhimaniVoiceSample.wav"
OUTPUT_DIR = "books/sample/musical_prompt/test_10_converted"

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load FreeVC model once
    vc = TTS("voice_conversion_models/multilingual/vctk/freevc24", gpu=True)

    # Iterate through all wav files in source directory
    for file_name in os.listdir(SOURCE_DIR):
        if file_name.endswith(".wav"):
            source_path = os.path.join(SOURCE_DIR, file_name)
            output_path = os.path.join(OUTPUT_DIR, file_name)  # same name in new folder

            print(f"Converting: {source_path} -> {output_path}")

            vc.voice_conversion_to_file(
                source_wav=source_path,
                target_wav=TARGET_WAV,
                file_path=output_path
            )

    print(f"\nAll files converted and saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
