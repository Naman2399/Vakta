'''
Github page : https://github.com/huggingface/parler-tts
'''
import argparse
import os
import re

import numpy as np
import pandas as pd
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from tqdm import tqdm
from transformers import AutoTokenizer
from TTS.api import TTS


# ##############################################################################

def get_argument():
    parser = argparse.ArgumentParser(description="Narration to Speech Synthesis")
    parser.add_argument("--input_path", type=str, default= "books/sample/musical_prompt/idx_1_chapter_1_narration_enhanced_musical_prompt.csv", help="Path to the PDF file")
    parser.add_argument("--model_name", type=str, default="parler-tts/parler-tts-mini-expresso", help="Name of the TTS model to use")
    parser.add_argument("--device", type=str, default="cuda", help = "Device to run the model on (cuda or cpu)")
    return parser.parse_args()

# def split_into_sentences(text):
#     """Split text into sentences using regex (basic)."""
#     sentences = re.split(r'(?<=[.!?]) +', text.strip())
#     return [s for s in sentences if s]

def split_into_sentences(text, max_len=250):
    """
    Split text into sentences and ensure each chunk is <= max_len characters.
    """
    # First split by punctuation
    sentences = re.split(r'(?<=[।.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    # Remove leading/trailing punctuation marks
    # sentences = [re.sub(r'^[।.!?]+|[।.!?]+$', '', s).strip() for s in sentences]

    final_chunks = []
    for sent in sentences:
        if len(sent) <= max_len:
            final_chunks.append(sent)
        else:
            # Break long sentences into smaller parts
            words = sent.split()
            chunk = []
            length = 0
            for word in words:
                if length + len(word) + 1 > max_len:
                    final_chunks.append(" ".join(chunk))
                    chunk = [word]
                    length = len(word)
                else:
                    chunk.append(word)
                    length += len(word) + 1
            if chunk:
                final_chunks.append(" ".join(chunk))

    return final_chunks

# --------- Split Text into Paragraph-Sized Chunks ----------
def split_into_paragraphs(text, max_len=250):
    text = text.replace('"', '')  # remove double quotes
    text = text.strip()

    # split by sentence-ending punctuation
    sentences = re.split(r'(?<=[।.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) < max_len:
            current += " " + s
        else:
            if current.strip():
                chunks.append(current.strip())
            current = s

    if current.strip():
        chunks.append(current.strip())

    # filter out chunks with no alphabetic/word characters (only symbols/numbers/punctuations)
    cleaned_chunks = [
        c for c in chunks
        if re.search(r'[A-Za-z\u0900-\u097F]', c)  # keep if has English/Hindi letters
    ]

    return cleaned_chunks



# --------- Concatenate with Small Silence Padding ----------
def concat_with_silence(audios, sr, silence_sec=0.2):
    silence = np.zeros(int(silence_sec * sr), dtype=np.float32)
    final = []
    for idx, wav in enumerate(audios):
        final.append(wav)
        if idx < len(audios) - 1:
            final.append(silence)
    return np.concatenate(final)


# --------- Apply Fade-In/Fade-Out ----------
def apply_fades(wav, fade_len=200):
    fade_in = np.linspace(0, 1, fade_len)
    wav[:fade_len] *= fade_in
    fade_out = np.linspace(1, 0, fade_len)
    wav[-fade_len:] *= fade_out
    return wav


# --------- Denoise with Torchaudio ----------
def denoise_audio(wav: np.ndarray, sr: int) -> np.ndarray:
    """
    Denoise audio using spectral gating (noisereduce).
    - Estimates noise profile from the quietest 0.5 sec.
    - Works with any sample rate.
    """

    try:
        import noisereduce as nr

        # take first 0.5s (or less if audio shorter) as noise profile
        noise_len = min(len(wav), sr // 2)
        noise_clip = wav[:noise_len]

        reduced = nr.reduce_noise(
            y=wav,
            sr=sr,
            y_noise=noise_clip,
            stationary=False,  # adaptive noise profile
            prop_decrease=1.0  # aggressiveness, can tune (0.8–1.0)
        )

        return reduced.astype(np.float32)

    except ImportError:
        print("[WARN] noisereduce not installed, returning raw audio.")
        return wav.astype(np.float32)


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
    all_audio = []

    # Split into sentences
    sentences = split_into_paragraphs(text)
    print(sentences)
    print(f"Processing {len(sentences)} sentences...")

    for idx, sentence in enumerate(sentences, 1):
        # Build description + sentence prompt
        description = f"{speaker} speaking in a {emotion} tone with clarity: {sentence}"
        desc_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
        text_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        gen = model.generate(input_ids=desc_ids, prompt_input_ids=text_ids)
        audio = gen.cpu().numpy().squeeze()

        print(f"Generated sentence {idx}/{len(sentences)}")
        all_audio.append(audio)

    # Concatenate all chunks
    merged_audio = np.concatenate(all_audio)

    # Save final audio
    sf.write(output_path, merged_audio, model.config.sampling_rate)
    print(f"Final merged audio saved: {output_path}")


def synthesize_with_xtts(text, reference_wav="books/sample/ENG_UK_M_DavidS.wav", output_path= None, language="en"):
    """
    Generate speech using XTTS v2 by cloning reference voice.
    - text: the input text to speak
    - reference_wav: path to reference audio file (voice sample)
    - output_path: where to save generated audio
    - language: target language (e.g., 'en', 'hi')
    """
    # Generate speech
    all_audio = []

    sentences = split_into_paragraphs(text)
    print(sentences)
    print(f"Processing {len(sentences)} sentences...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    for idx, sentence in enumerate(sentences, 1):
        # Generate audio per sentence
        try :
            wav = tts.tts(
                text=sentence,
                speaker_wav=reference_wav,
                language=language
            )
        except Exception as e:
            print(f"Error generating audio for sentence {idx}: {e}")
            continue

        wav = np.array(wav, dtype=np.float32)

        # Normalize loudness
        wav = wav / (np.max(np.abs(wav)) + 1e-8)

        all_audio.append(wav)
        print(f"✅ Generated chunk {idx}/{len(sentences)}")

    # Concatenate with silence padding
    sr = 24000
    combined = concat_with_silence(all_audio, sr, silence_sec=0.2)

    # Apply fade smoothing
    combined = apply_fades(combined, fade_len=400)

    # Save intermediate raw file
    sf.write("raw_output.wav", combined, sr)
    print("💾 Raw file saved: raw_output.wav")

    # Denoise
    clean = denoise_audio(combined, sr)

    # Apply final fade-out
    clean = apply_fades(clean, fade_len=400)

    # Save final output
    sf.write(output_path, clean, sr)
    print(f"🎧 Final clean audio saved: {output_path}")

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
        output_dir = os.path.join(output_dir, 'test_10').replace('\\', '/')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, output_path).replace('\\', '/')
        row['speech_output_path_v1'] = output_path

        # Generate speech with specified emotion and speaker
        synthesize_with_xtts(text= text, output_path= output_path)

        # Add speech duration to the dataframe
        row['speech_v1_duration'] = f'{get_wav_duration(output_path):.2f}'

    # Save the updated DataFrame with speech output paths
    output_csv_path = args.input_path.replace('narration', 'speech_v1').replace('.csv', '_speech.csv')
    df.to_csv(output_csv_path, index=False)
    print(f"Speech synthesis complete. Output saved to {output_csv_path}")




