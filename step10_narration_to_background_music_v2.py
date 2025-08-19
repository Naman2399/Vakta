import argparse
import os

import audiocraft
import pandas as pd
import torchaudio
from audiocraft.models import MusicGen
from tqdm import tqdm

from step8_narration_speech import get_wav_duration


def get_arguments() :
    parser = argparse.ArgumentParser(description='Generate background music from text prompts.')
    parser.add_argument('--input', type=str, default='books/sample/musical_prompt/idx_1_chapter_1_speech_v1_enhanced_musical_prompt_speech.csv', help='CSV file for background activities for music generation')
    parser.add_argument('--model_name', type=str, default='facebook/musicgen-large', help='Hugging Face MusicGen model name')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on (cpu or cuda)')

    return parser.parse_args()

def generate_music(prompt: str, duration_seconds: float = 10,
                   model: audiocraft.models.MusicGen = None,
                   output_path: str = None, device: str = 'cpu'):
    """
    Generates music from a text prompt for the given duration.
    """

    model.set_generation_params(duration=duration_seconds)
    # Process input prompt
    wav = model.generate([prompt])
    # Save as WAV (float32)
    torchaudio.save(output_path, wav[0].cpu(), 24000)

    # Duration calculation
    sampling_rate = 24000  # MusicGen default
    num_samples = wav.shape[-1]
    duration_sec = num_samples / sampling_rate

    print(f"✅ Music saved to {output_path} ({duration_sec:.2f} sec)")

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

    # Load Model -
    model = MusicGen.get_pretrained('facebook/musicgen-large', device=device)  # use 'small' for even faster

    # Now we will iterate through the dataframe and generate background music for each row
    for index, row in tqdm(df.iterrows(), desc="Generating Background Music", total=len(df)):
        if index < 33 :
            continue # Skip first 8 rows as they are not relevant for background music generation
        background_activity = row['Musical Prompt'].strip()
        speech_duration = row['speech_v1_duration']
        output_path = row['background_music_output_path'].strip()

        # Output path
        output_dir = os.path.dirname(args.input).replace('speech_v1', 'background_music_v1').replace('\\', '/')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, output_path).replace('\\', '/')
        row['background_music_output_path'] = output_path

        # Generate background music with specified activity
        generate_music(prompt= background_activity, model= model,  duration_seconds= float(speech_duration), output_path= output_path, device= device)

        # Add background music duration to the dataframe
        row['background_music_duration'] = f'{get_wav_duration(output_path):.2f}'

    # Save the updated dataframe with background music paths and durations
    output_csv_path = args.input.replace('speech_v1', 'background_music_v1').replace('.csv', '_background_music.csv')
    df.to_csv(output_csv_path, index=False)
    print(f"Background music generation complete. Output saved to {output_csv_path}")




