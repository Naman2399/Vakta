import argparse
import os

import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoProcessor, MusicgenForConditionalGeneration

from step6_narration_speech import get_wav_duration


def get_arguments() :
    parser = argparse.ArgumentParser(description='Generate background music from text prompts.')
    parser.add_argument('--input', type=str, default='books/sample/speech_v1/idx_1_chapter_1_speech_v1_speech.csv', help='CSV file for background activities for music generation')
    parser.add_argument('--model_name', type=str, default='facebook/musicgen-small', help='Hugging Face MusicGen model name')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on (cpu or cuda)')

    return parser.parse_args()

def generate_music(prompt: str, duration_seconds: float = 10,
                   model_name: str = None,
                   output_path: str = None, device: str = 'cpu'):
    """
    Generates music from a text prompt for the given duration.
    """
    # Load model + processor
    device_map = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    model = MusicgenForConditionalGeneration.from_pretrained(model_name).to(device_map)
    processor = AutoProcessor.from_pretrained(model_name)

    # MusicGen uses tokens to set length
    tokens_per_5sec = 256
    max_new_tokens = int((duration_seconds / 5) * tokens_per_5sec)

    # Process input prompt
    inputs = processor(text=[prompt], return_tensors="pt").to(device_map)

    # Generate audio
    audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Convert to float32 NumPy
    audio_float32 = audio_values[0].cpu().numpy().astype("float32")

    # Ensure shape is (samples,) not multi-dimensional
    if audio_float32.ndim > 1:
        audio_float32 = audio_float32.squeeze()

    # Get sampling rate
    sampling_rate = model.config.audio_encoder.sampling_rate

    # Save as WAV (float32)
    sf.write(output_path, audio_float32, samplerate=sampling_rate, subtype='FLOAT')

    print(f"✅ Music saved to {output_path} ({len(audio_float32) / sampling_rate:.2f} sec)")

# Example usage
if __name__ == "__main__":

    # Parse command line arguments
    args = get_arguments()

    # Ensure the input path exists
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"The specified input file does not exist: {args.input_path}")

    # Ensure the model name is valid
    if not args.model_name:
        raise ValueError("Model name must be specified.")

    # Ensure device is valid
    if args.device not in ["cuda", "cpu"]:
        raise ValueError("Device must be either 'cuda' or 'cpu'.")

    model_name = args.model_name
    device =args.device

    # Read the dataframe
    df = pd.read_csv(args.input)

    # Now need to add one more column --> background_music_output_path
    df['background_music_output_path'] = df.apply(lambda row: f"background_music_{row.name + 1}.wav", axis=1)
    df['background_music_duration'] = None

    # Now we will iterate through the dataframe and generate background music for each row
    for index, row in tqdm(df.iterrows(), desc="Generating Background Music", total=len(df)):
        background_activity = row['Background Activities'].strip()
        speech_duration = row['speech_v1_duration']
        output_path = row['background_music_output_path'].strip()

        # Output path
        output_dir = os.path.dirname(args.input).replace('speech_v1', 'background_music_v1').replace('\\', '/')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, output_path).replace('\\', '/')
        row['background_music_output_path'] = output_path

        # Generate background music with specified activity
        generate_music(prompt= background_activity, model_name= model_name,  duration_seconds= float(speech_duration), output_path= output_path, device= device)

        # Add background music duration to the dataframe
        row['background_music_duration'] = f'{get_wav_duration(output_path):.2f}'

    # Save the updated dataframe with background music paths and durations
    output_csv_path = args.input.replace('speech_v1', 'background_music_v1').replace('.csv', '_background_music.csv')
    df.to_csv(output_csv_path, index=False)
    print(f"Background music generation complete. Output saved to {output_csv_path}")




