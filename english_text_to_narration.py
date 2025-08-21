'''
Code Description:
Step 1 - Extract text from a PDF using OCR and save it to a CSV file.
Step 2 - Clean the extracted text using OpenAI API to improve readability and remove errors.
Step 3 - Convert the cleaned text to narration dialogues.
Step 4 - Convert the narration dialogues to enhanced narration.
Step 5 - Convert background activities and dialogues to musical prompts.
Step 6 - Narration Check
Step 7 - Convert the narration dialogues to speech using XTTS v2.
Step 8 - Generate background music for each dialogue using MusicGen.
'''

import argparse
import csv
import os
import re

import audiocraft
import easyocr
import fitz
import numpy as np
import pandas as pd
import soundfile as sf
import torch.cuda
import torchaudio
from TTS.api import TTS
from audiocraft.models import MusicGen
from openai import OpenAI
from tqdm import tqdm

from config.open_ai_config import API_KEY, TEXT_MODEL

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"


def get_arguments():

    parser = argparse.ArgumentParser(description="Convert the English Story PDF to narration speech and background music.")
    # Step1 : Extract text from the PDF using OCR and save it to a CSV file.
    parser.add_argument('--pdf_path', type=str, default= "books/sample_v2/book.pdf",help='Path to the input PDF file.')
    parser.add_argument('---pdf_start_page', type=int, default=5, help='Start page number for text extraction.')
    parser.add_argument('--pdf_end_page', type=int, default=5, help='End page number for text extraction.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for OCR processing (e.g., "cpu" or "cuda").')
    # Output csv path ---> The output directory will be same as that of pdf_path, with the output file name specified.
    parser.add_argument('--output_csv_file_name', type=str, default='extracted_text.csv', help='Path to save the extracted text as CSV.')

    # Step 2 : Clean the extracted text and save it to a CSV file.
    parser.add_argument('--openai_api_key', type=str, default= API_KEY, help='OpenAI API key for text cleaning')
    parser.add_argument('--open_ai_text_model', type=str, default= TEXT_MODEL, help='OpenAI text model to use for cleaning')

    # Step 3 : Convert the cleaned text to narration dialogues.
    parser.add_argument('--output_csv_file_name_dialogue', type=str, default='narration_dialogues.csv', help='Path to save the narration dialogues as CSV.')

    # Step 7 : Convert the TTS
    parser.add_argument('--reference_wav', type=str, default='books/sample/ENG_UK_M_DavidS.wav', help='Path to the reference audio file for TTS synthesis.')
    parser.add_argument("--tts_model_name", type=str, default="parler-tts/parler-tts-mini-expresso", help="Name of the TTS model to use")

    return parser.parse_args()

class EnglishTextToNarration:

    def __init__(self, pdf_path, output_csv_file_name, start_page, end_page, open_ai_client, open_ai_model, device):
        self.pdf_path = pdf_path
        self.output_csv_file_name = output_csv_file_name
        self.start_page = start_page
        self.end_page = end_page
        self.open_ai_client = open_ai_client
        self.open_ai_model = open_ai_model
        self.device = device

    def extract_text_from_pdf(self):
        """
        Step 1:
        Extract text from the PDF file using OCR and save it to a CSV file.
        """

        reader = easyocr.Reader(['en'])
        doc = fitz.open(self.pdf_path)

        with open(self.output_csv_file_name, 'w', newline='', encoding='utf-8') as csvfile:
            # Create a CSV writer object
            writer = csv.writer(csvfile)
            # Write header to CSV
            writer.writerow(['page_number', 'content'])
            # Process each page in the specified range
            for page_num in tqdm(range(self.start_page - 1, self.end_page), desc="Processing pages",
                                 total=end_page - start_page + 1):
                pix = doc[page_num].get_pixmap(dpi=300)  # Higher DPI for better OCR
                img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                text = ' '.join(reader.readtext(img_np, detail=0))
                writer.writerow([page_num + 2 - start_page, text.strip()])

        print(f"Text extraction complete. Data saved to {self.output_csv_file_name}.")
        return

    def clean_ocr_text(self, text):
        """
        Step 2:
        Clean the OCR text using OpenAI API to improve readability and remove errors.
        :param text:
        :return: the cleaned text
        """
        system_prompt = """You are an assistant that cleans and refines OCR text while preserving its meaning. 
                    Your task:
                    1. Fix spacing issues and join broken words (e.g., "examp le" → "example").
                    2. Remove unnecessary line breaks and extra spaces.
                    3. Correct common OCR mistakes (e.g., '1' → 'l', '0' → 'o') where appropriate.
                    4. Ensure proper capitalization and punctuation.
                    5. Remove exact duplicate phrases or sentences if they appear in the text.
                    6. Eliminate filler or meaningless words that disrupt readability.
                    7. Smooth the flow so the text reads naturally, while keeping the original meaning intact.

                    Output only the cleaned and improved text without any commentary or explanation."""

        user_prompt = f"""Clean, correct, and improve the following OCR text:
        \"\"\"{text}\"\"\""""

        response = self.open_ai_client.responses.create(
            model=self.open_ai_model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.output_text.strip()

    def convert_page_to_script(self, prev_text, current_text):
        system_prompt = """
            You are a scriptwriter for an audiobook where the narrator tells the story as a continuous, immersive script. 
            Follow these principles:
            1. The 'Actor' will always be "Narrator".
            2. 'Dialogue' must be expressive, descriptive, and sound like a storyteller speaking to an audience. 
               - Merge multiple related ideas into a single, flowing narration. 
               - Each dialogue block should feel substantial (at least 4–6 sentences), painting vivid imagery and smoothly linking events.
            3. 'Emotion' should reflect the overall mood or tone of the narration (can have more than one word, e.g., "calm and hopeful").
            4. 'Background Activities' should describe subtle but fitting ambient sounds or music cues (e.g., "soft flute music", "birds chirping in distance").
            5. Avoid breaking the narration into too many short lines — prioritize fewer, longer segments that carry the story forward.

            Output format:
            - Output as below structure with separated by tag <break>.
            - Each line should follow the structure:
              Actor <break> Dialogue <break> Emotion <break> Background Activities
            - No extra commentary or explanations outside the CSV.
            - Do not put extra spaces after commas unless needed in the dialogue or descriptions.

            Example:
            Narrator <break> "Once upon a time, in a land far away, there lived a wise old king whose wisdom was sought by rulers from distant lands. His palace, with its golden domes and fragrant gardens, was a place of peace and reflection." <break> calm and nostalgic <break> "gentle harp music"
            Narrator <break> "One day, as the sun dipped low and painted the sky in shades of crimson, a weary messenger rode into the palace courtyard, carrying a letter sealed with urgency. The air grew heavy with anticipation as the king broke the seal." <break> tense and serious <break> "distant thunder"
            """

        user_prompt = f"""Convert the following text into a structured drama script as per the rules, ensuring fewer but longer narrations.

            Text for this page (with 50 words from the previous page for context):
            \"\"\"{prev_text} {current_text}\"\"\""""

        response = self.open_ai_client.responses.create(
            model= self.open_ai_model,  # Cheap and good quality
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.output_text.strip()

    def convert_narration_to_enhanced_narration(self, current_text, next_text):
        system_prompt = """
        You are a skilled audiobook script editor.
        You will be given two consecutive narration lines that may have incomplete sentences or awkward breaks.

        Your task:
        1. Merge them into a single, smooth-flowing narration without losing meaning.
        2. Ensure transitions are natural, no abrupt cuts.
        3. Actor is always "Narrator".
        4. Dialogue must be at least 4–6 sentences, descriptive, and immersive.
        5. Keep 'Emotion' as a concise mood label that fits the merged narration (multiple words allowed).
        6. Suggest 'Background Activities' as subtle ambient sound cues.

        Output format (each field separated by <break>):
        Actor <break> Dialogue <break> Emotion <break> Background Activities
        No extra explanations.
        Example:
        Narrator <break> "The sun dipped below the hills, painting the sky in gold and crimson, as villagers gathered around the fire to share stories. Children laughed, the old spoke in hushed tones, and a gentle wind carried the scent of pine." <break> warm and nostalgic <break> "soft crackling of fire"
        """

        user_prompt = f"""
        Merge these two narration segments into a single, smooth-flowing audiobook narration:
        Segment 1: "{current_text}"
        Segment 2: "{next_text}"
        """

        response = client.responses.create(
            model=text_model,  # Cheap and good quality
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.output_text.strip()

    def convert_background_activites_and_dialogues_to_musical_prompt(self, dialogue, background_activities):
        """
        Generate a melodic and audience-engaging background music description
        using dialogue + background activities.
        """

        # System prompt for the LLM with examples
        system_prompt = """You are a background music designer for an audio drama.
            Given a dialogue line and a short description of background activity, 
            you must generate a *concise* and *melodic* background music prompt 
            that can be used with facebook/musicgen-large.

            Rules:
            - Keep total description under 25 words.
            - Always suggest instruments and musical style.
            - Avoid noise-heavy sounds; make it pleasant and audience-engaging.
            - Music should blend with background activity but remain melodic.
            - Mention tempo or mood (e.g., gentle, uplifting, suspenseful).
            - Output must be a single line, no bullet points.

            Example outputs:
            1. "Gentle flute and soft tabla beats with warm strings, uplifting and calm."
            2. "Light acoustic guitar with soft chimes, peaceful and heartwarming."
            3. "Slow piano melody with soft rain sounds, reflective and soothing."
            4. "Bright marimba with gentle hand drums, cheerful and playful."
            5. "Warm cello and soft piano with light harp plucks, romantic and tender."
            """

        # Create the combined input
        user_prompt = f"""
            Dialogue: {dialogue}
            Background Activity: {background_activities}

            Now generate one line of background music description:
        """

        response = self.open_ai_client.responses.create(
            model= self.open_ai_model,  # Cheap and good quality
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.output_text.strip()

    def synthesize_with_xtts(self, text, reference_wav="books/sample/ENG_UK_M_DavidS.wav", output_path=None, language="en"):
        """
        Generate speech using XTTS v2 by cloning reference voice.
        - text: the input text to speak
        - reference_wav: path to reference audio file (voice sample)
        - output_path: where to save generated audio
        - language: target language (e.g., 'en', 'hi')
        """
        # Generate speech
        all_audio = []

        sentences = self.split_into_paragraphs(text)
        print(sentences)
        print(f"Processing {len(sentences)} sentences...")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)

        for idx, sentence in enumerate(sentences, 1):
            # Generate audio per sentence
            try:
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
        combined = self.concat_with_silence(all_audio, sr, silence_sec=0.2)

        # Apply fade smoothing
        combined = self.apply_fades(combined, fade_len=400)

        # Save intermediate raw file
        sf.write("raw_output.wav", combined, sr)
        print("💾 Raw file saved: raw_output.wav")

        # Denoise
        clean = self.denoise_audio(combined, sr)

        # Apply final fade-out
        clean = self.apply_fades(clean, fade_len=400)

        # Save final output
        sf.write(output_path, clean, sr)
        print(f"🎧 Final clean audio saved: {output_path}")

        return

    def get_wav_duration(self, wav_path):
        with sf.SoundFile(wav_path) as f:
            duration = len(f) / f.samplerate
        return duration

    # --------- Split Text into Paragraph-Sized Chunks ----------
    def split_into_paragraphs(self, text, max_len=250):
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
    def concat_with_silence(self, audios, sr, silence_sec=0.2):
        silence = np.zeros(int(silence_sec * sr), dtype=np.float32)
        final = []
        for idx, wav in enumerate(audios):
            final.append(wav)
            if idx < len(audios) - 1:
                final.append(silence)
        return np.concatenate(final)

    # --------- Apply Fade-In/Fade-Out ----------
    def apply_fades(self, wav, fade_len=200):
        fade_in = np.linspace(0, 1, fade_len)
        wav[:fade_len] *= fade_in
        fade_out = np.linspace(1, 0, fade_len)
        wav[-fade_len:] *= fade_out
        return wav

    # --------- Denoise with Torchaudio ----------
    def denoise_audio(self, wav: np.ndarray, sr: int) -> np.ndarray:
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

    def generate_music(self, prompt: str, duration_seconds: float = 10,
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
        return duration_sec


if __name__ == "__main__":

    # Parse command line arguments
    args = get_arguments()

    book_pdf_path = args.pdf_path
    start_page = args.pdf_start_page
    end_page = args.pdf_end_page
    client = OpenAI(api_key=args.openai_api_key)
    text_model = args.open_ai_text_model

    # Print the parsed arguments for debugging
    print(f"PDF Path: {book_pdf_path}")
    print(f"Start Page: {start_page}")
    print(f"End Page: {end_page}")
    print(f"Output CSV File Name: {args.output_csv_file_name}")
    print(f"OpenAI API Key: {args.openai_api_key}")
    print(f"OpenAI Text Model: {text_model}")

    # Ensure the PDF file exists
    if not os.path.isfile(book_pdf_path):
        raise FileNotFoundError(f"The specified PDF file does not exist: {book_pdf_path}")

    # Define the chapter mapping
    chapter_mapping = {  # key is the chapter name, value is a list of start and end pages
        'chapter_1': [start_page, end_page],
    }

    # Ensure start_page and end_page are valid
    for chapter_name, (start_page, end_page) in chapter_mapping.items():
        if not isinstance(start_page, int) or not isinstance(end_page, int):
            raise ValueError(f"Invalid page numbers for {chapter_name}: start_page and end_page must be integers.")
        # Ensure start_page and end_page are positive integers
        if start_page < 1 or end_page < 1 or end_page < start_page:
            raise ValueError(f"Invalid page numbers for {chapter_name}: start_page and end_page must be >= 1.")

    # Define the output CSV file path
    # Extract the directory from the book_path and create the output CSV file path
    book_dir = os.path.dirname(book_pdf_path)
    # Create the output CSV file path in the same directory as the PDF
    output_csv_path = os.path.join(book_dir, args.output_csv_file_name).replace('\\', '/')


    # Create an instance of the EnglishTextToNarration class
    narration_converter = EnglishTextToNarration(pdf_path= book_pdf_path,
                                                 output_csv_file_name= output_csv_path,
                                                 start_page=start_page,
                                                 end_page=end_page,
                                                 open_ai_client=client,
                                                 open_ai_model=text_model,
                                                 device= args.device)

    print("Step 1: Extracting text from the PDF...")
    # Extract text from the PDF and save to CSV
    narration_converter.extract_text_from_pdf()

    print(f"Step 2: Cleaning the extracted text saved in {output_csv_path}... ")
    df = pd.read_csv(str(output_csv_path), encoding='utf-8')
    df = df.sort_values(by='page_number')  # Sort all the content by page
    # Add one more column to the dataframe for cleaned text
    df['cleaned_content'] = None

    # Iterate through each row and clean the text
    for index, row in tqdm(df.iterrows(), desc="Cleaning text", total=len(df), unit="row"):
        page_number = row['page_number']
        ocr_text = row['content']

        # Clean the OCR text using OpenAI API
        cleaned_text = narration_converter.clean_ocr_text(text = ocr_text)
        # Update the cleaned content in the dataframe
        df.at[index, 'cleaned_content'] = cleaned_text

    # Saving the dataframe to the output CSV file
    df.to_csv(str(output_csv_path), index=False, encoding='utf-8')
    print(f"Text cleaning complete. Cleaned data saved to {output_csv_path}.")

    # Step 3: Convert the cleaned text to narration dialogues
    print("Step 3: Converting cleaned text to narration dialogues...")
    # Convert each page of text to a script format
    dialogues = []

    for i in tqdm(range(len(df)), desc="Converting text to dialogues", total=len(df)):
        # For each page, we need the previous page's context
        if i > 0:
            # Use the last 50 words of the previous text for context
            prev_text = ' '.join(df.iloc[i - 1]['cleaned_content'].split()[-50:])
        else:
            # If it's the first page, no previous context
            prev_text = ""

        current_text = df.iloc[i]['cleaned_content']
        script = narration_converter.convert_page_to_script(prev_text= prev_text, current_text= current_text)

        # Print content of script for debugging
        print(f"Script for page {i + 1}:\n{script}\n{'-' * 50}")

        # Convert the output text to a strucutred format -
        skipping_lines = 0
        for line in script.split("\n"):
            parts = line.strip().split("<break>")
            if len(parts) == 4:
                dialogues.append(parts)
            else:
                print(f"Skipping line due to unexpected format: {line}")
                skipping_lines += 1

        print(f"Skipped {skipping_lines} lines due to unexpected format.")

    # Create a DataFrame from the dialogues
    dialogues_df = pd.DataFrame(dialogues, columns=['Actor', 'Dialogue', 'Emotion', 'Background Activities'])

    # Read the dialogues DataFrame to a CSV file
    for dialogue in dialogues:
        actor, dialogue_text, emotion, background = dialogue[0], dialogue[1], dialogue[2], dialogue[3]
        # Add input of actor, dialogue_text, emotion, background to the dialgoues_df dataframe
        dialogues_df = dialogues_df.append({
            'Actor': actor,
            'Dialogue': dialogue_text,
            'Emotion': emotion,
            'Background Activities': background
        }, ignore_index=True)

    # Create the output CSV file path in the same directory as the PDF
    output_csv_path = os.path.join(book_dir, args.output_csv_file_name_dialogue).replace('\\', '/')
    # Saving the dialogues DataFrame to a CSV file
    print(f"Saving dialogues to {output_csv_path}")
    dialogues_df.to_csv(str(output_csv_path), index=False, encoding="utf-8")
    print(f"Dialogues saved to {output_csv_path}")

    # Step4: Convert the narration to enhanced narration
    print("Step 4: Converting narration to enhanced narration...")
    df = pd.read_csv(str(output_csv_path), encoding='utf-8')
    dialogues = []
    for i in tqdm(range(len(df) - 1), desc="Converting narration to enhanced narration", total=len(df) - 1):

        # For each page, we need the previous page's context
        current_text = df.iloc[i]["Dialogue"]
        next_text = df.iloc[i + 1]["Dialogue"]

        script = narration_converter.convert_narration_to_enhanced_narration(current_text=current_text, next_text=next_text)

        # Print content of script for debugging
        print(f"Script for page {i + 1}:\n{script}\n{'-' * 50}")

        # Convert the output text to a strucutred format -
        skipping_lines = 0
        for line in script.split("\n"):
            parts = line.strip().split("<break>")
            if len(parts) == 3:
                dialogues.append(parts)

                # For the first updat the dataframe
                if i <= len(df) - 2:
                    df.iloc[i] = parts[1]
                else:
                    df.iloc[i] = parts[1]
                    df.iloc[i + 1] = parts[2]

            else:
                print(f"Skipping line due to unexpected format: {line}")
                skipping_lines += 1

        print(f"Skipped {skipping_lines} lines due to unexpected format.")

    # Saving the enhanced dialogues DataFrame to a CSV file
    # Append _enhanced to the output CSV file name
    output_csv_path = os.path.join(book_dir, args.output_csv_file_name_dialogue.replace('.csv', '_enhanced.csv')).replace('\\', '/')
    print(f"Saving enhanced dialogues to {output_csv_path}")
    df.to_csv(str(output_csv_path), index=False, encoding="utf-8")
    print(f"Enhanced dialogues saved to {output_csv_path}")

    # Step 5: Background music to Musical Prompt
    print("Step 5: Converting background activities to musical prompt...")
    df = pd.read_csv(str(output_csv_path), encoding='utf-8')
    df["Musical Prompt"] = None  # Add new column for Musical Prompt each value to None

    # Convert each page of text to a script format
    dialogues = []
    for i in tqdm(range(len(df)), desc="Converting dialogue + background activites to musical prompt",
                  total=len(df)):
        # For each page, we need the previous page's context
        dialogue = df.iloc[i]["Dialogue"]
        background_activiy = df.iloc[i]["Background Activities"]

        musci_prompt = narration_converter.convert_background_activites_and_dialogues_to_musical_prompt(dialogue= dialogue,
                                                                                                        background_activities= background_activiy)
        # Print content of script for debugging
        print(f"Music prompt for dialogue {i + 1}:\n{musci_prompt}\n{'-' * 50}")
        # Add to the dataframe
        df["Musical Prompt"].iloc[i] = musci_prompt

    print(f"Saving musical prompt to {output_csv_path}")
    df.to_csv(str(output_csv_path), index=False, encoding="utf-8")
    print(f"Musical Prompts saved to {output_csv_path}")

    # Step 6: Narration Check
    print("Step 6: Narration Check")
    # Check for missing values in 'Dialogue' and 'Emotion' columns
    missing_dialogue = df['Dialogue'].isnull().sum()
    missing_emotion = df['Emotion'].isnull().sum()
    missing_background_activities = df['Background Activities'].isnull().sum()
    missing_musical_prompt = df['Musical Prompt'].isnull().sum()

    if missing_dialogue > 0:
        print(f"Warning: {missing_dialogue} rows have missing 'Dialogue'.")
    else:
        print("All 'Dialogue' entries are present.")

    if missing_emotion > 0:
        print(f"Warning: {missing_emotion} rows have missing 'Emotion'.")
    else:
        print("All 'Emotion' entries are present.")

    if missing_musical_prompt > 0:
        print(f"Warning: {missing_musical_prompt} rows have missing 'Musical Prompt'.")
    else:
        print("All 'Musical Prompt' entries are present.")

    if missing_background_activities > 0:
        print(f"Warning: {missing_background_activities} rows have missing 'Background Activities'.")
    else:
        print("All 'Background Activities' entries are present.")

    # Check for empty strings in 'Dialogue' and 'Emotion' columns
    empty_dialogue = (df['Dialogue'] == '').sum()
    empty_emotion = (df['Emotion'] == '').sum()
    empty_background_activities = (df['Background Activities'] == '').sum()
    empty_musical_prompt = (df['Musical Prompt'] == '').sum()

    if empty_dialogue > 0:
        print(f"Warning: {empty_dialogue} rows have empty 'Dialogue'.")
    else:
        print("No empty 'Dialogue' entries found.")

    if empty_emotion > 0:
        print(f"Warning: {empty_emotion} rows have empty 'Emotion'.")
    else:
        print("No empty 'Emotion' entries found.")

    if empty_background_activities > 0:
        print(f"Warning: {empty_background_activities} rows have empty 'Background Activities'.")
    else:
        print("No empty 'Background Activities' entries found.")

    if empty_musical_prompt > 0:
        print(f"Warning: {empty_musical_prompt} rows have empty 'Musical Prompt'.")
    else:
        print("No empty 'Musical Prompt' entries found.")

    # Print complete narration statistics
    total_rows = len(df)
    print(f"Total rows in narration: {total_rows}")
    print("-" * 50)
    print(f"Rows with missing 'Dialogue': {missing_dialogue}")
    print(f"Rows with missing 'Emotion': {missing_emotion}")
    print(f"Rows with missing 'Background Activities': {missing_background_activities}")
    print(f"Rows with missing 'Musical Prompt': {missing_musical_prompt}")
    print("-" * 50)
    print(f"Rows with empty 'Dialogue': {empty_dialogue}")
    print(f"Rows with empty 'Emotion': {empty_emotion}")
    print(f"Rows with empty 'Background Activities': {empty_background_activities}")
    print(f"Rows with empty 'Musical Prompt': {empty_musical_prompt}")
    print("-" * 50)

    # Print complete narration content
    print("\nComplete Narration Content:")
    for index, row in df.iterrows():
        print(f"Row {index + 1}: Actor: {row['Actor']}, Dialogue: {row['Dialogue']}, Emotion: {row['Emotion']}, Background Activities: {row['Background Activities']}, Musical Prompt: {row['Musical Prompt']}")


    # Step 7: Convert the narration dialogues to speech [Text to Speech]
    print("Step 7: Converting narration dialogues to speech using XTTS v2...")

    df['speech_output_path'] = df.apply(lambda row: f"narration_{row.name + 1}.wav", axis=1)
    df['speech_duration'] = None

    narration_output_dir = os.path.dirname(book_pdf_path)
    narration_output_dir = os.path.join(narration_output_dir, 'narration_speech').replace('\\', '/')
    if not os.path.exists(narration_output_dir):
        os.makedirs(narration_output_dir)

    # Now we will iterate through the dataframe and generate speech for each row
    for index, row in tqdm(df.iterrows(), desc="Generating Speech", total=len(df)):
        speaker = row['Actor'].strip()
        text = row['Dialogue'].strip()
        emotion = row['Emotion'].strip()
        background_activities = row['Background Activities'].strip()
        output_path = row['speech_output_path'].strip()

        output_path = os.path.join(str(narration_output_dir), output_path).replace('\\', '/')
        row['speech_output_path'] = output_path

        # Generate speech with specified emotion and speaker
        narration_converter.synthesize_with_xtts(text=text,
                                                 output_path=output_path,
                                                 reference_wav= args.reference_wav,
                                                 language='en')

        # Add speech duration to the dataframe
        row['speech_duration'] = f'{narration_converter.get_wav_duration(output_path):.2f}'

    # Save the updated dataframe with speech output paths and durations
    df.to_csv(str(output_csv_path), index=False, encoding="utf-8")
    print(f"Speech generation complete. Audio files saved in {narration_output_dir}.")
    print(f"Updated CSV with speech paths and durations saved to {output_csv_path}")

    # Step 8: Convert the musical prompt to background music
    print("Step 8: Generating background music for each narration dialogue...")
    torch.cuda.empty_cache()
    df['background_music_output_path'] = df.apply(lambda row: f"background_music_{row.name + 1}.wav", axis=1)
    df['background_music_duration'] = None

    background_music_output_path = os.path.dirname(book_pdf_path)
    background_music_output_path = os.path.join(background_music_output_path, 'background_music').replace('\\', '/')
    if not os.path.exists(background_music_output_path):
        os.makedirs(background_music_output_path)

    background_music_model = MusicGen.get_pretrained('facebook/musicgen-large', device= args.device)  # use 'small' for even faster

    # Now we will iterate through the dataframe and generate background music for each row
    for index, row in tqdm(df.iterrows(), desc="Generating Background Music", total=len(df)):

        background_activity = row['Musical Prompt'].strip()
        speech_duration = row['speech_duration']
        output_path = row['background_music_output_path'].strip()

        output_path = os.path.join(str(background_music_output_path), output_path).replace('\\', '/')
        row['background_music_output_path'] = output_path

        # Generate background music with specified activity
        duration_sec = narration_converter.generate_music(prompt=background_activity, model=background_music_model, duration_seconds=float(speech_duration),
                       output_path=output_path, device=args.device)

        # Add background music duration to the dataframe
        row['background_music_duration'] = f'{duration_sec:.2f}'

    df.to_csv(str(output_csv_path), index=False, encoding="utf-8")
    print(f"Background Music generation complete. Audio files saved in {background_music_output_path}.")
    print(f"Updated CSV with speech paths and durations saved to {output_csv_path}")



