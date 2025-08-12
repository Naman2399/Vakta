'''
Github page : https://github.com/huggingface/parler-tts
'''
import argparse
import os

import pandas as pd
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from tqdm import tqdm
from transformers import AutoTokenizer


# ##############################################################################

def get_argument():
    parser = argparse.ArgumentParser(description="Narration to Speech Synthesis")
    parser.add_argument("--input_path", type=str, default= "books/sample/narration/idx_1_chapter_1_narration.csv", help="Path to the PDF file")
    parser.add_argument("--model_name", type=str, default="parler-tts/parler-tts-mini-expresso", help="Name of the TTS model to use")
    parser.add_argument("--device", type=str, default="cuda", help = "Device to run the model on (cuda or cpu)")
    return parser.parse_args()

def synthesize_emotional_speech(text, emotion, speaker="deep female voice with rich and captivating tone", output_path="output.wav"):
    """

    Already tried -
    test_2 = soft female voice with calm tone
    test_3 = energetic male voice with engaging tone
    test_4 = "deep female voice with rich and captivating tone"
    test_5 = "rich female voice with confident and engaging tone"
    test_6 = "deep female voice with smooth and persuasive tone"
    test_7 = "low female voice with powerful and dynamic tone"
    test_8 = "deep female voice with warm and commanding tone"
    test_9 = "deep male voice with rich and captivating tone"
    test_10 = "rich male voice with confident and engaging tone"
    test_11 = "deep male voice with smooth and persuasive tone"
    test_12 = "low male voice with powerful and dynamic tone"
    test_13 = "deep male voice with warm and commanding tone"

    Generates speech from text with specified emotion and speaker style.
    """
    description = f"{speaker} speaking in a {emotion} tone with clarity"
    desc_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    text_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    gen = model.generate(input_ids=desc_ids, prompt_input_ids=text_ids)
    audio = gen.cpu().numpy().squeeze()
    sf.write(output_path, audio, model.config.sampling_rate)
    print(f"Generated {output_path}")

def get_wav_duration(wav_path):
    with sf.SoundFile(wav_path) as f:
        duration = len(f) / f.samplerate
    return duration

if __name__ == "__main__":

    # Parse command line arguments
    args = get_argument()

    # Ensure the input path exists
    if not os.path.isfile(args.input_path):
        raise FileNotFoundError(f"The specified input file does not exist: {args.input_path}")

    # Ensure the model name is valid
    if not args.model_name:
        raise ValueError("Model name must be specified.")

    # Ensure device is valid
    if args.device not in ["cuda", "cpu"]:
        raise ValueError("Device must be either 'cuda' or 'cpu'.")

    model_name = args.model_name
    device = args.device

    # Load the Parler-TTS model and tokenizer
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Read dataframe
    df = pd.read_csv(args.input_path)

    # Now need to add one more column --> speech_output_path_v1
    df['speech_output_path_v1'] = df.apply(lambda row: f"narration_{row.name + 1}.wav", axis=1)
    df['speech_v1_duration'] = None
    # Now we will iterate through the dataframe and generate speech for each row
    for index, row in tqdm(df.iterrows(), desc="Generating Speech", total=len(df)):
        speaker = row['Actor'].strip()
        text = row['Dialogue'].strip()
        emotion = row['Emotion'].strip()
        background_activities = row['Background Activities'].strip()
        output_path = row['speech_output_path_v1'].strip()

        output_dir = os.path.dirname(args.input_path).replace('narration', 'speech_v1').replace('\\', '/')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, output_path).replace('\\', '/')
        row['speech_output_path_v1'] = output_path

        # Generate speech with specified emotion and speaker
        synthesize_emotional_speech(text= text, emotion= emotion, output_path= output_path)

        # Add speech duration to the dataframe
        row['speech_v1_duration'] = f'{get_wav_duration(output_path):.2f}'

    # Save the updated DataFrame with speech output paths
    output_csv_path = args.input_path.replace('narration', 'speech_v1').replace('.csv', '_speech.csv')
    df.to_csv(output_csv_path, index=False)
    print(f"Speech synthesis complete. Output saved to {output_csv_path}")




