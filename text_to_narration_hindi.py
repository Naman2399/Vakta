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
import gc
import os
import re

import audiocraft
import easyocr
import fitz
import librosa
import numpy as np
import pandas
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from TTS.api import TTS
from audiocraft.models import MusicGen
from openai import OpenAI
from pydub import AudioSegment
from tqdm import tqdm

from config.audio_reference_samples import ENG_UK_HUME_DIR, ENG_INDIAN_MALE_DIR, ENG_INDIAN_FEMALE_DIR, \
    HINDI_MALE_1_DIR, HINDI_FEMALE_1_DIR, HINDI_MALE_2_DIR, HINDI_FEMALE_2_DIR, ENG_INDIAN_MALE_MOHIT_DIR
from config.background_music_models import MUSIC_GEN_MELODY
from config.open_ai_config import API_KEY, TEXT_MODEL
from config.tts_model_config import XTTS_V2
from interfaces.narration import NarrationInterface
from interfaces.ocr import OCRInterface
from transformers import pipeline


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"

def get_arguments():

    parser = argparse.ArgumentParser(description="Convert the English Story PDF to narration speech and background music.")

    # Step1 : Extract text from the PDF using OCR and save it to a CSV file.
    parser.add_argument('--pdf_path', type=str, default= "books/sample_hindi_story/book.pdf",help='Path to the input PDF file.')
    parser.add_argument('---pdf_start_page', type=int, default=1, help='Start page number for text extraction.')
    parser.add_argument('--pdf_end_page', type=int, default=19, help='End page number for text extraction.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for OCR processing (e.g., "cpu" or "cuda").')
    # Output csv path ---> The output directory will be same as that of pdf_path, with the output file name specified.
    parser.add_argument('--output_csv_file_name_extracted_text', type=str, default='extracted_text_hindi.csv', help='Path to save the extracted text as CSV.')

    # Step 2 : Clean the extracted text and save it to a CSV file.
    parser.add_argument('--openai_api_key', type=str, default= API_KEY, help='OpenAI API key for text cleaning')
    parser.add_argument('--open_ai_text_model', type=str, default= TEXT_MODEL, help='OpenAI text model to use for cleaning')

    # Step 3 : Convert the cleaned text to narration dialogues.
    parser.add_argument('--output_csv_file_name_dialogue_english', type=str, default='narration_dialogues_english.csv',
                        help='Path to save the narration dialogues as CSV.')
    parser.add_argument('--output_csv_file_name_dialogue_hindi', type=str, default='narration_dialogues_hindi.csv',
                        help='Path to save the narration dialogues as CSV.')

    # Step 7 : Convert the TTS
    parser.add_argument('--reference_wav_dir_english', type=str, default=ENG_INDIAN_MALE_MOHIT_DIR,
                        help='Path to the reference audio file for TTS synthesis.')
    parser.add_argument('--reference_wav_dir_hindi', type=str, default=HINDI_MALE_2_DIR,
                        help='Path to the reference audio file for TTS synthesis.')

    parser.add_argument("--tts_model_name", type=str, default= XTTS_V2, help="Name of the TTS model to use")

    # Step 8 : Background Music Generation
    parser.add_argument('--background_music_model', type=str, default= MUSIC_GEN_MELODY, help= "Model for generating the Background Music using prompt")  # Adding small model for now [Original] ---> facebook/musicgen-large

    return parser.parse_args()

class StoryIntroGenerator:
    def __init__(self, openai_api_key: str, openai_text_model: str, df: pd.DataFrame):
        # Initialize OpenAI client
        self.open_ai_client = OpenAI(api_key=openai_api_key)
        self.open_ai_text_model = openai_text_model
        self.df = df

    def semantic_chunk_dialogues(self, df: pd.DataFrame, chunk_size=1000):
        """
        Splits dialogues into coherent chunks (scene-based or actor-based).
        chunk_size: approximate number of words per chunk
        """
        chunks = []
        current_chunk = []
        current_word_count = 0

        for dialogue in df["Dialogue Enhanced"]:
            word_count = len(str(dialogue).split())
            current_chunk.append(dialogue)
            current_word_count += word_count

            if current_word_count >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def generate_chunk_summary(self, chunk: str) -> str:
        """
        Generates a very brief, high-level summary for a given chunk of dialogues.
        Focuses only on what the story is generally about, not detailed retelling.
        """
        system_prompt = """आप एक संक्षिप्त कथावाचक हैं। 
            आपको संवादों का एक हिस्सा दिया जाएगा। 
            इन संवादों के आधार पर एक *बहुत ही छोटा, ऊपरी स्तर का सारांश* हिन्दी में लिखें।

            नियम:
            - 3 वाक्यों से कम रखें।
            - विस्तार से घटनाओं का वर्णन न करें।
            - केवल मुख्य विचार या विषय को पकड़ें।
            - आउटपुट केवल छोटा हिन्दी सारांश हो, और कुछ नहीं।"""

        user_prompt = f"संवाद अंश:\n{chunk}\n\nऊपरी स्तर का छोटा हिन्दी सारांश लिखें:"

        response = self.open_ai_client.responses.create(
            model=self.open_ai_text_model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.output_text.strip()

    def hierarchical_summarization_recursive(self, texts: list, chunk_limit=5, max_words=200) -> str:
        """
        Recursively summarizes texts until the final combined text is below max_words.
        """
        print(f"Processing {len(texts)} chunks...")

        # Step 1: Summarize each chunk
        mini_summaries = [self.generate_chunk_summary(text) for text in tqdm(texts)]

        # Step 2: If the combined summaries are still too large, recurse
        combined_text = " ".join(mini_summaries)
        total_words = len(combined_text.split())

        if total_words <= max_words:
            # Small enough, return as final summary
            return combined_text
        else:
            # Split mini-summaries into groups for next-level summarization
            next_level_groups = []
            for i in range(0, len(mini_summaries), chunk_limit):
                group = mini_summaries[i:i + chunk_limit]
                next_level_groups.append(" ".join(group))

            # Recursive call
            return self.hierarchical_summarization_recursive(next_level_groups, chunk_limit=chunk_limit,
                                                             max_words=max_words)

    def generate_story_intro_from_csv(self) -> str:
        """
        Main function to generate story introduction paragraph.
        """
        chunks = self.semantic_chunk_dialogues(self.df)
        final_intro = self.hierarchical_summarization_recursive(chunks)
        return final_intro

    def generate_chunk_takeaway(self, chunk: str) -> str:
        """
        Generates a very brief life lesson or moral takeaway in Hindi for a chunk of dialogues.
        """
        system_prompt = """आप एक बुद्धिमान कथावाचक हैं। 
        आपको संवादों का एक हिस्सा दिया जाएगा। 
        इन संवादों के आधार पर *मुख्य जीवन सीख, नैतिक संदेश, या सार* हिन्दी में लिखें।

        नियम:
        - 2 वाक्यों से अधिक न हो।
        - घटनाओं का विवरण न दें।
        - केवल जीवन का संदेश या सार्वभौमिक सीख प्रस्तुत करें।
        - आउटपुट केवल मुख्य संदेश हो, और कुछ नहीं।"""

        user_prompt = f"संवाद अंश:\n{chunk}\n\nमुख्य संदेश/सार लिखें:"

        response = self.open_ai_client.responses.create(
            model=self.open_ai_text_model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.output_text.strip()

    def hierarchical_takeaway_recursive(self, texts: list, chunk_limit=5, max_words=100) -> str:
        """
        Recursively extracts lessons until the final takeaway is concise.
        """
        print(f"Processing {len(texts)} chunks for takeaways...")

        # Step 1: Generate takeaway for each chunk
        mini_takeaways = [self.generate_chunk_takeaway(text) for text in tqdm(texts)]

        # Step 2: If combined is short enough, return
        combined_text = " ".join(mini_takeaways)
        total_words = len(combined_text.split())

        if total_words <= max_words:
            return combined_text
        else:
            # Group takeaways for recursive summarization
            next_level_groups = []
            for i in range(0, len(mini_takeaways), chunk_limit):
                group = mini_takeaways[i:i + chunk_limit]
                next_level_groups.append(" ".join(group))

            return self.hierarchical_takeaway_recursive(next_level_groups, chunk_limit=chunk_limit,
                                                        max_words=max_words)

    def generate_story_takeaway_from_csv(self) -> str:
        """
        Main function to generate final story takeaway/lesson.
        """
        chunks = self.semantic_chunk_dialogues(self.df)
        final_takeaway = self.hierarchical_takeaway_recursive(chunks)
        return final_takeaway


class HindiNarration(OCRInterface, NarrationInterface) :

    def __init__(self, pdf_path: str = None, pdf_start_page: int = None,
                 pdf_end_page: int = None, device: str = None,
                 extract_text_csv_file_path: str = None,
                 openai_api_key: str = None, open_ai_text_model: str = None,
                 output_csv_file_path_dialogue: str = None, reference_wav_dir: str = None,
                 tts_model_name: str = None, background_music_model: str = None)  :

        self.pdf_path = pdf_path
        self.pdf_start_page = pdf_start_page
        self.pdf_end_page = pdf_end_page
        self.device = device
        self.extract_text_csv_file_path = extract_text_csv_file_path
        self.openai_api_key = openai_api_key
        self.output_csv_file_path_dialogue = output_csv_file_path_dialogue
        self.reference_wav_dir = reference_wav_dir
        self.tts_model_name = tts_model_name
        self.background_music_model_name = background_music_model

        # Initialize OpenAI client
        if self.openai_api_key is not None :
            self.open_ai_client = OpenAI(api_key=self.openai_api_key)
            self.open_ai_text_model = open_ai_text_model
        else :
            self.open_ai_client = None
            self.open_ai_text_model = None

        # Initialize TTS model
        if self.tts_model_name is not None :
            self.tts = TTS(self.tts_model_name).to(self.device)
        else :
            self.tts = None

        # Initialize MusicGen model
        if self.background_music_model_name is not None:
            self.musicgen_model = MusicGen.get_pretrained(self.background_music_model_name, device=self.device)
        else:
            self.musicgen_model = None  # Temporarily disable MusicGen to avoid import issues during testing

        # Initialize DataFrame to hold extracted text
        self.df : pandas.DataFrame = None
        self.df_dialogues : pandas.DataFrame = None

        # List of voice emmotions available in audio reference samples
        self.voice_artist_emmotions = []
        for file_name in os.listdir(self.reference_wav_dir) :
            if file_name.lower().endswith(".wav") :  # check for .wav files
                name_without_ext = os.path.splitext(file_name)[0]  # remove extension
                self.voice_artist_emmotions.append(name_without_ext)

    def extract_text_from_pdf(self, pdf_path: str, start_page: int, end_page: int,
                              output_csv_file_name: str) -> pandas.DataFrame:
        """
            Extract text from the PDF file using OCR and save it to a CSV file.
            Output consists of the dataframe with columsn [Page Number, Content]
        """

        # Initialize OCR reader
        reader = easyocr.Reader(['hi', 'en'])  # Hindi + English fallback
        doc = fitz.open(self.pdf_path)

        with open(output_csv_file_name, 'w', newline='', encoding='utf-8') as csvfile:
            # Create a CSV writer object
            writer = csv.writer(csvfile)
            # Write header to CSV
            writer.writerow(['Page Number', 'Content'])
            # Process each page in the specified range
            for page_num in tqdm(range(self.pdf_start_page - 1, self.pdf_end_page), desc="Processing pages",
                                 total=self.pdf_end_page - self.pdf_start_page + 1):
                pix = doc[page_num].get_pixmap(dpi=300)  # Higher DPI for better OCR
                img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                text = ' '.join(reader.readtext(img_np, detail=0))
                writer.writerow([page_num + 2 - start_page, text.strip()])

        print(f"Text extraction complete. Data saved to {self.extract_text_csv_file_path}.")

        # Load the dataframe common to class and return the extracted text as a DataFrame
        self.df = pd.read_csv(self.extract_text_csv_file_path)
        return self.df

    def clean_ocr_text_iterrator(self, df: pandas.DataFrame) -> pandas.DataFrame :

        """
        :param df: Dataframe consits of [Page Number, Content]
        :return: Return DataFrame [Page Number, Content, Content Clean]
        """

        # Iterate through each row and clean the text
        self.df = df
        self.df= self.df.sort_values(by='Page Number')
        self.df['Content Clean'] = np.nan

        for index, row in tqdm(self.df.iterrows(), desc="Cleaning text", total=len(self.df), unit="row"):
            ocr_text = row['Content']
            # Clean the OCR text using OpenAI API
            cleaned_text = self.clean_ocr_text(text=ocr_text)
            # Update the cleaned content in the dataframe
            self.df.at[index, 'Content Clean'] = cleaned_text

        # Rewrite the file to dataframe
        self.df.to_csv(self.extract_text_csv_file_path)
        return self.df

    def clean_ocr_text(self, text: str) -> str:
        """
        Clean the Hindi OCR text using OpenAI API to improve readability and remove errors.
        :param text: Hindi OCR text input
        :return: cleaned Hindi text
        """

        system_prompt = """आप एक सहायक हैं जिसका कार्य OCR से निकले हुए हिंदी टेक्स्ट को
                           पढ़ने योग्य और स्वाभाविक बनाना है। 
                           आपका काम:
                           1. गलत स्पेसिंग और टूटी हुई मात्राएँ/शब्दों को सही करना 
                              (जैसे "ज िवन" → "जीवन")।
                           2. अनावश्यक लाइन ब्रेक और अतिरिक्त स्पेस हटाना।
                           3. आम OCR गलतियाँ सुधारना 
                              (जैसे "॥" को सही जगह लगाना, "।" का गलत प्रयोग ठीक करना)।
                           4. उचित विराम चिह्न (। , ? ! आदि) लगाना।
                           5. एक जैसे वाक्यांश या वाक्य दोहराए गए हों तो हटाना।
                           6. बेमतलब या अस्पष्ट अक्षर/शब्द हटाना।
                           7. वाक्य को स्वाभाविक और सरल हिंदी प्रवाह में सुधारना, 
                              लेकिन मूल अर्थ को बदले बिना।

                           आउटपुट केवल सुधारा हुआ टेक्स्ट हो, 
                           किसी प्रकार की टिप्पणी या व्याख्या न दें।"""

        user_prompt = f"""निम्न OCR से प्राप्त हिंदी टेक्स्ट को सुधारें और पठनीय बनाएं:
        \"\"\"{text}\"\"\""""

        response = self.open_ai_client.responses.create(
            model=self.open_ai_text_model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.output_text.strip()

    def generate_script_iterrator(self, df: pandas.DataFrame) -> pandas.DataFrame:

        """

        :param df: Input is the dataframe consists of columns [Page Number, Content, Content Clean]
        :return: df : Output consists of columns ['Actor', 'Dialogue', 'Emotion', 'Background Activity']
        """

        # Convert each page of text to a script format
        dialogues = []

        for i in tqdm(range(len(df)), desc="Converting text to dialogues", total=len(df)):
            # For each page, we need the previous page's context
            if i > 0:
                # Use the last 50 words of the previous text for context
                prev_text = ' '.join(df.iloc[i - 1]['Content Clean'].split()[-50:])
            else:
                # If it's the first page, no previous context
                prev_text = ""

            current_text = df.iloc[i]['Content Clean']
            script = self.generate_script(prev_text=prev_text, current_text=current_text)

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
        df_dialogues = pd.DataFrame(dialogues, columns=['Actor', 'Dialogue', 'Emotion', 'Background Activity'])

        # Create the output CSV file path in the same directory as the PDF
        print(f"Saving dialogues to {self.output_csv_file_path_dialogue}")
        self.df_dialogues = df_dialogues
        df_dialogues.to_csv(str(self.output_csv_file_path_dialogue), index=False, encoding="utf-8")
        print(f"Dialogues saved to {self.output_csv_file_path_dialogue}")

        return df_dialogues

    def generate_script(self, prev_text: str, current_text: str) -> str:

        """

        :param prev_text: Based on the continuity of the data from prev page content
        :param current_text: Current Page Content
        :return: ALl the dialgoues with appropirate pause in the format prescribed below with different parameters as mentioned in the iterrator function
        """

        system_prompt = """
        आप एक ऑडियोबुक स्क्रिप्ट लेखक हैं, जहाँ वाचक (Narrator) कहानी को एक निरंतर, रोचक और भावनात्मक ढंग से सुनाता है। 
        नियम इस प्रकार हैं:
        1. 'Actor' हमेशा "Narrator" होगा।
        2. 'Dialogue' अभिव्यक्तिपूर्ण, चित्रात्मक और श्रोताओं से संवाद जैसा होना चाहिए।
           - जुड़े हुए विचारों को एक ही प्रवाहपूर्ण वर्णन में मिलाएँ।
           - हर संवाद खंड कम से कम 4–6 वाक्यों का हो, जो दृश्य और भावनाएँ स्पष्ट रूप से चित्रित करे।
        3. 'Emotion' संवाद की मुख्य भावना या मूड को दर्शाए (एक से अधिक शब्द हो सकते हैं, जैसे "शांत और आशावान", "उदास और गंभीर")।
        4. 'Background Activities' में हल्की लेकिन उपयुक्त ध्वनियों या संगीत का वर्णन हो 
           (जैसे "मंद बांसुरी की धुन", "पक्षियों की चहचहाहट", "हल्की हवा की सरसराहट")।
        5. संवाद को बहुत छोटे-छोटे भागों में न बाँटें — कहानी को आगे बढ़ाने वाले लंबे खंड बनाएँ।

        आउटपुट प्रारूप:
        - हर पंक्ति <break> से अलग होगी।
        - संरचना इस प्रकार होगी:
          Actor <break> Dialogue <break> Emotion <break> Background Activities
        - आउटपुट में अतिरिक्त टिप्पणी या व्याख्या न दें।
        - कॉमा के बाद केवल वहीं स्पेस दें जहाँ वाक्य संरचना में ज़रूरी हो।

        उदाहरण:
        Narrator <break> "बहुत समय पहले, एक विशाल साम्राज्य में एक बुद्धिमान राजा रहता था। उसकी दयालुता और न्यायप्रियता दूर-दूर तक प्रसिद्ध थी। उसका महल सोने के गुम्बदों और सुगंधित उद्यानों से भरा हुआ था, जहाँ हर कोई शांति और सुकून महसूस करता था।" <break> शांत और स्मृतिपूर्ण <break> "मंद सितार की धुन"
        Narrator <break> "एक दिन, जब सूर्य अस्त हो रहा था और आकाश लालिमा से भर गया था, तभी एक थका हुआ संदेशवाहक महल के द्वार पर पहुँचा। उसके हाथों में एक मुहरबंद पत्र था, और वातावरण में उत्सुकता और गंभीरता की लहर दौड़ गई।" <break> गंभीर और तनावपूर्ण <break> "दूर से आती बादलों की गड़गड़ाहट"
        """

        user_prompt = f"""निम्नलिखित पाठ को दिए गए नियमों के अनुसार 
        एक संरचित नाटकीय स्क्रिप्ट में बदलें, यह सुनिश्चित करते हुए कि 
        कम लेकिन लंबे नैरेशन हों। 

        इस पृष्ठ का पाठ (पिछले पृष्ठ से 50 शब्द संदर्भ सहित): 
        \"\"\"{prev_text} {current_text}\"\"\""""

        response = self.open_ai_client.responses.create(
            model=self.open_ai_text_model,  # Cheap and good quality
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.output_text.strip()

    def convert_narration_to_enhanced_narration_iterrator(self, df: pandas.DataFrame) -> pandas.DataFrame:

        """

        :param df: Dataframe consists of ['Actor', 'Dialogue', 'Emotion', 'Background Activity']
        :return: To bring the continuity in the above method between each dailogue will return new column 'Dialogue Enhanced'
        """

        self.df = df.copy()
        self.df["Dialogue Enhanced"] = np.nan

        for i in tqdm(range(len(self.df)), desc="Converting narration to enhanced narration"):

            current_text = self.df.at[i, "Dialogue"]
            script = self.convert_narration_to_enhanced_narration(current_text= current_text)

            # Debug print
            print(f"Script for row {i + 1}:\n{script}\n{'-' * 50}")

            # Parse LLM response
            parts = [p.strip() for p in script.split("<break>")]
            if len(parts) == 2:  # Actor + merged narration
                self.df.at[i, "Dialogue Enhanced"] = parts[1]
            else:
                print(f"⚠️ Unexpected format at row {i + 1}: {script}")
                self.df.at[i, "Dialogue Enhanced"] = current_text  # fallback


        # Save results
        print(f"Saving enhanced dialogues to {self.output_csv_file_path_dialogue}")
        self.df.to_csv(str(self.output_csv_file_path_dialogue), index=False, encoding="utf-8")
        print(f"Enhanced dialogues saved to {self.output_csv_file_path_dialogue}")

        return self.df

    def convert_narration_to_enhanced_narration(self, current_text: str) -> str:
        """
        किसी एक डायलॉग को और भी सुगम व प्राकृतिक ऑडियोबुक नैरेशन में बदलना।
        यहाँ अगले डायलॉग की ज़रूरत नहीं है।
        """

        system_prompt = """
            आप एक कुशल ऑडियोबुक स्क्रिप्ट संपादक हैं।
            कार्य:
            1. दिए गए संवाद (डायलॉग) को लेकर उसे एक सहज, प्रवाहपूर्ण नैरेशन में बदलें,
               जो ऑडियोबुक में स्वाभाविक और निरंतर सुनाई दे।
            2. अगले संवाद की ओर कोई संकेत या टीज़र न दें।
            3. मूल अर्थ को बनाए रखें, केवल भाषा को और प्रवाहपूर्ण, श्रोताओं को जोड़ने वाला बनाएँ।
            4. 'Actor' हमेशा "Narrator" रहेगा।
            5. आउटपुट बिल्कुल इस प्रारूप में दें ( <break> से अलग करें ):
               Narrator <break> सुधरा हुआ नैरेशन
        """

        user_prompt = f"""
            मूल संवाद: "{current_text}"

            इसे एक और प्रवाहपूर्ण, प्राकृतिक ऑडियोबुक नैरेशन में बदलें,
            ताकि यह सहज और रोचक लगे।
        """

        response = self.open_ai_client.responses.create(
            model=self.open_ai_text_model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.output_text.strip()

    def convert_background_activites_and_dialogues_to_musical_prompt_iterrator(self, df: pandas.DataFrame) -> pandas.DataFrame:

        """

        :param df: Dataframe consists ['Actor', 'Dialogue', 'Emotion', 'Background Activity', 'Dialogue Enhanced']
        :return: Return a new column ['Musical Prompt'] based on the Dialogue Enhanced and Background Activity
        """

        self.df = df
        df['Musical Prompt'] = np.NAN
        for i in tqdm(range(len(df)), desc="Converting dialogue + background activites to musical prompt",
                      total=len(df)):
            # For each page, we need the previous page's context
            dialogue = df.iloc[i]["Dialogue Enhanced"]
            background_activiy = df.iloc[i]["Background Activity"]

            musci_prompt = self.convert_background_activites_and_dialogues_to_musical_prompt(
                dialogue=dialogue,
                background_activitiy=background_activiy)
            # Print content of script for debugging
            print(f"Music prompt for dialogue {i + 1}:\n{musci_prompt}\n{'-' * 50}")
            # Add to the dataframe
            df.at[i, "Musical Prompt"] = musci_prompt

        print(f"Saving musical prompt to {self.output_csv_file_path_dialogue}")
        df.to_csv(str(self.output_csv_file_path_dialogue), index=False, encoding="utf-8")
        print(f"Musical Prompts saved to {self.output_csv_file_path_dialogue}")

        return df

    def convert_background_activites_and_dialogues_to_musical_prompt(self, dialogue: str,
                                                                     background_activitiy: str) -> str:
        """
        डायलॉग और बैकग्राउंड गतिविधि के आधार पर,
        एक मधुर और श्रोताओं को जोड़ने वाला संगीत वर्णन तैयार करना।
        """

        # System prompt for the LLM with Hindi instructions + style
        system_prompt = """
            आप एक ऑडियो नाटक के बैकग्राउंड म्यूज़िक डिज़ाइनर हैं।
            दिए गए संवाद और बैकग्राउंड गतिविधि के आधार पर,
            आपको एक *संक्षिप्त* और *संगीतात्मक* पृष्ठभूमि संगीत विवरण तैयार करना है
            जिसे facebook/musicgen-large मॉडल के साथ प्रयोग किया जा सके।

            नियम:
            - पूरा विवरण 25 शब्दों से कम होना चाहिए।
            - हमेशा वाद्ययंत्रों और संगीत शैली का उल्लेख करें।
            - शोर-भरे या कठोर ध्वनियों से बचें; संगीत मधुर और श्रोताओं को आकर्षित करने वाला हो।
            - संगीत को बैकग्राउंड गतिविधि के साथ मिलाना है, परंतु वह प्रवाहपूर्ण और लयबद्ध रहे।
            - टेम्पो या मूड का ज़िक्र करें (जैसे: कोमल, प्रेरणादायक, रोमांचक)।
            - आउटपुट केवल एक पंक्ति में हो, बुलेट पॉइंट या अतिरिक्त व्याख्या न दें।

            उदाहरण आउटपुट:
            1. "मृदु बाँसुरी और हल्की तबला ताल के साथ गर्मजोशी भरे स्ट्रिंग्स, शांत और प्रेरणादायक।"
            2. "हल्का एकॉस्टिक गिटार और कोमल घंटियाँ, सुकूनदायक और दिल को छू लेने वाला।"
            3. "धीमी पियानो धुन और हल्की बारिश की ध्वनि, चिंतनशील और सुकूनदायक।"
            4. "चमकदार मरिम्बा और कोमल हेंड ड्रम्स, चंचल और हर्षित।"
            5. "गर्म सेलो और कोमल पियानो के साथ हल्के हार्प प्लक्स, रोमांटिक और कोमल।"
        """

        # Combined input
        user_prompt = f"""
            संवाद: {dialogue}
            बैकग्राउंड गतिविधि: {background_activitiy}

            अब एक पंक्ति में बैकग्राउंड संगीत का विवरण तैयार करें:
        """

        response = self.open_ai_client.responses.create(
            model=self.open_ai_text_model,  # Cheap and good quality
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.output_text.strip()

    def narration_check(self, df: pandas.DataFrame) -> None:

        # Columns you want to validate
        columns_to_check = ["Actor", "Dialogue", "Emotion", "Background Activity", "Dialogue Enhanced", "Musical Prompt"]

        # Total rows
        total_rows = len(df)
        print(f"Total rows in narration: {total_rows}")
        print("-" * 50)

        # Check for missing and empty values in a loop
        for col in columns_to_check:
            missing_count = df[col].isnull().sum()
            empty_count = (df[col] == '').sum()

            if missing_count > 0:
                print(f"⚠️ Warning: {missing_count} rows have missing '{col}'.")
            else:
                print(f"✅ All '{col}' entries are present.")

            if empty_count > 0:
                print(f"⚠️ Warning: {empty_count} rows have empty '{col}'.")
            else:
                print(f"✅ No empty '{col}' entries found.")

            print("-" * 50)

        # Print complete narration content
        print("\nComplete Narration Content:")
        for index, row in df.iterrows():
            row_data = ", ".join([f"{col}: {row[col]}" for col in columns_to_check])
            print(f"Row {index + 1}: Actor: {row['Actor']}, {row_data}")

        return

    def convert_text_to_speech_iterrator(self, df: pandas.DataFrame) -> pandas.DataFrame:

        """

        :param df: Input consists of ["Actor", "Dialogue", "Emotion", "Background Activity", "Dialogue Enhanced", "Musical Prompt"]
        :return: Add two more columns Speech Output Path, Speech Duration
        """

        df['Speech Output Path'] = df.apply(lambda row: f"narration_{row.name + 1}.wav", axis=1)
        df['Speech Duration'] = None

        narration_output_dir = os.path.dirname(self.output_csv_file_path_dialogue)
        narration_output_dir = os.path.join(narration_output_dir, 'narration_speech_hindi').replace('\\', '/')

        if not os.path.exists(narration_output_dir):
            os.makedirs(narration_output_dir, exist_ok= True)

        # Now we will iterate through the dataframe and generate speech for each row
        for index, row in tqdm(df.iterrows(), desc="Generating Speech", total=len(df)):
            text = row['Dialogue Enhanced'].strip()
            emotion = row['Emotion'].strip()
            output_path = row['Speech Output Path'].strip()

            output_path = os.path.join(str(narration_output_dir), output_path).replace('\\', '/')
            df.at[index, 'Speech Output Path'] = output_path

            # Generate speech with specified emotion and speaker
            self.convert_text_to_speech(text=text,
                                        output_path=output_path,
                                        reference_wav_dir=self.reference_wav_dir,
                                        language='hi', emotion= emotion)

            # Add speech duration to the dataframe
            df.at[index, 'Speech Duration'] = f'{self.get_wav_duration(output_path):.2f}'

        # Save the updated dataframe with speech output paths and durations
        df.to_csv(str(self.output_csv_file_path_dialogue), index=False, encoding="utf-8")
        print(f"Speech generation complete. Audio files saved in {narration_output_dir}.")
        print(f"Updated CSV with speech paths and durations saved to {self.output_csv_file_path_dialogue}")

        return df

    def convert_text_to_speech(self, text: str, reference_wav_dir: str, output_path: str, emotion: str, language: str = "en") -> None:

        """
        Generate speech using XTTS v2 by cloning reference voice.
        - text: the input text to speak
        - reference_wav: path to reference audio file (voice sample)
        - output_path: where to save generated audio
        - language: target language (e.g., 'en', 'hi')
        """

        voice_artist_emmotion_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        voice_artist_emmotion_classifier_result = voice_artist_emmotion_classifier(emotion, candidate_labels=self.voice_artist_emmotions)
        highest_index = voice_artist_emmotion_classifier_result['scores'].index(max(voice_artist_emmotion_classifier_result['scores']))
        file_name = voice_artist_emmotion_classifier_result['labels'][highest_index]
        reference_wav = os.path.join(reference_wav_dir, f"{file_name}.wav").replace('\\', '/')
        print(f"Generating audio for sentence with emotion '{file_name}' and reference '{reference_wav}'...")

        # Generate speech
        all_audio = []

        sentences = self.split_into_paragraphs(text)
        print(sentences)
        print(f"Processing {len(sentences)} sentences...")

        for idx, sentence in enumerate(sentences, 1):
            # Generate audio per sentence
            try:
                wav = self.tts.tts(
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
        # sf.write("raw_output.wav", combined, sr)
        # print("💾 Raw file saved: raw_output.wav")

        # Denoise
        clean = self.denoise_audio(combined, sr)

        # Apply final fade-out
        clean = self.apply_fades(clean, fade_len=400)

        # Save final output
        sf.write(output_path, clean, sr)
        print(f"🎧 Final clean audio saved: {output_path}")

        return

    def generate_background_music_iterator(self, df: pandas.DataFrame) -> pandas.DataFrame:
        """
            Iterator to generate background music from Musical Prompt
        :param df: Dataframe - ["Actor", "Dialogue", "Emotion", "Background Activity", "Dialogue Enhanced", "Musical Prompt", "Speech Output Path", "Speech Duration"]
        :return: Dataframe - ["Background Music Output Path", "Background Music Duration"]
        """

        df['Background Music Output Path'] = df.apply(lambda row: f"background_music_{row.name + 1}.wav", axis=1)
        df['Background Music Duration'] = None

        background_music_output_path = os.path.dirname(self.output_csv_file_path_dialogue)
        background_music_output_path = os.path.join(background_music_output_path, 'background_music').replace('\\', '/')
        if not os.path.exists(background_music_output_path):
            os.makedirs(background_music_output_path)

        # Now we will iterate through the dataframe and generate background music for each row
        for index, row in tqdm(df.iterrows(), desc="Generating Background Music", total=len(df)):
            background_activity = row['Musical Prompt'].strip()
            speech_duration = row['Speech Duration']
            output_path = row['Background Music Output Path'].strip()

            output_path = os.path.join(str(background_music_output_path), output_path).replace('\\', '/')
            df.at[index, 'Background Music Output Path'] = output_path

            # Generate background music with specified activity
            duration_sec = self.generate_background_music(prompt=background_activity, model=self.musicgen_model,
                                                              duration =float(speech_duration),
                                                              output_path=output_path, device=args.device)

            # Add background music duration to the dataframe
            df.at[index, 'Background Music Duration'] = f'{duration_sec:.2f}'

        df.to_csv(str(self.output_csv_file_path_dialogue), index=False, encoding="utf-8")
        print(f"Background Music generation complete. Audio files saved in {background_music_output_path}.")
        print(f"Updated CSV with speech paths and durations saved to {self.output_csv_file_path_dialogue}")

        return df


    def generate_background_music(self, prompt: str, duration: float, output_path: str,
                                  model: audiocraft.models.MusicGen, device: str) -> float:
        """Generate music based on a prompt and save it to a file.
        Input Arguments:
            prompt (str): The musical prompt for generation.
            duration (float): Duration of the generated music in seconds.
            output_path (str): Path to save the generated music audio file.
            model (audiocraft.models.MusicGen): The MusicGen model to use for generation.
            device (str): The device to run the model on (e.g., "cpu" or "cuda").
        Output:
            float: Duration of the generated music in seconds.
        """

        self.musicgen_model.set_generation_params(duration=duration)
        # Process input prompt
        try :
            wav = self.musicgen_model.generate([prompt])
        except :
            return float(-1)
        # Save as WAV (float32)
        torchaudio.save(output_path, wav[0].cpu(), 24000)

        # Duration calculation
        sampling_rate = 24000  # MusicGen default
        num_samples = wav.shape[-1]
        duration_sec = num_samples / sampling_rate

        print(f"✅ Music saved to {output_path} ({duration_sec:.2f} sec)")
        return duration_sec

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

    def get_wav_duration(self, wav_path: str) -> float:
        """Get the duration of a WAV audio file.
        Input Arguments:
            wav_path (str): Path to the WAV audio file.
        Output:
            float: Duration of the audio file in seconds.
        """
        with sf.SoundFile(wav_path) as f:
            duration = len(f) / f.samplerate
        return duration

    def merge_narration_background_music_iterrator(self, df: pandas.DataFrame ) -> pandas.DataFrame:

        """
            Iterator to generate background music from Musical Prompt
            :param df: Dataframe - ["Actor", "Dialogue", "Emotion", "Background Activity", "Dialogue Enhanced", "Musical Prompt", "Speech Output Path", "Speech Duration", "Background Music Output Path", "Background Music Duration"]
            :return: Dataframe - ["Narration with Background Music", "Narration with Background Music Duration"]
        """

        df['Narration with Background Music Output Path'] = df.apply(lambda row: f"narration_and_background_music_{row.name + 1}.wav", axis=1)
        df['Narration with Background Music Duration'] = None

        narration_and_background_music_output_path = os.path.dirname(self.output_csv_file_path_dialogue)
        narration_and_background_music_output_path = os.path.join(narration_and_background_music_output_path, 'narration_and_background_music_combined_hindi').replace('\\', '/')
        if not os.path.exists(narration_and_background_music_output_path):
            os.makedirs(narration_and_background_music_output_path)

        # Now we will iterate through the dataframe and generate background music for each row
        for index, row in tqdm(df.iterrows(), desc="Merging Narration + Background Music", total=len(df)):
            narration_path = row['Speech Output Path'].strip()
            background_music_path = row['Background Music Output Path'].strip()
            output_path = row['Narration with Background Music Output Path'].strip()

            output_path = os.path.join(str(narration_and_background_music_output_path), output_path).replace('\\', '/')
            df.at[index, 'Narration with Background Music Output Path'] = output_path

            # Generate background music with specified activity
            duration_sec = self.merge_narration_background_music(
                narration_path= narration_path,
                music_path= background_music_path,
                output_path= output_path,

                # --- Adjusted values ---
                music_base_db=-14.0,  # lower constant level (was -14.0)
                duck_extra_db=-10.0,  # stronger dip during speech (was -10.0)

                frame_ms=160,
                attack_ms=60,
                release_ms=220,
                final_fade_ms=300,
            )

            # Add background music duration to the dataframe
            df.at[index, 'Narration with Background Music Duration'] = f'{duration_sec:.2f}'

        df.to_csv(str(self.output_csv_file_path_dialogue), index=False, encoding="utf-8")
        print(f"Background Music generation complete. Audio files saved in {narration_and_background_music_output_path}.")
        print(f"Updated CSV with speech paths and durations saved to {self.output_csv_file_path_dialogue}")

        return df

    def ensure_output_path(self, output_path: str) -> str:
        if os.path.isdir(output_path) or os.path.splitext(output_path)[1].lower() != ".wav":
            os.makedirs(output_path if os.path.isdir(output_path) else os.path.dirname(output_path) or ".",
                        exist_ok=True)
            return os.path.join(output_path if os.path.isdir(output_path) else os.path.dirname(output_path),
                                "final_mix.wav")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        return output_path

    def db_to_amp(self, db: float) -> float:
        return 10.0 ** (db / 20.0)

    def fade_in_out(self, x: np.ndarray, sr: int, fade_ms: int = 300) -> np.ndarray:
        n = len(x)
        f = int(sr * fade_ms / 1000)
        if f > 0 and n > 2 * f:
            ramp_in = np.linspace(0.0, 1.0, f, endpoint=True, dtype=np.float32)
            ramp_out = np.linspace(1.0, 0.0, f, endpoint=True, dtype=np.float32)
            x[:f] *= ramp_in
            x[-f:] *= ramp_out
        return x

    def tile_or_trim(self, y: np.ndarray, target_len: int) -> np.ndarray:
        if len(y) >= target_len:
            return y[:target_len]
        reps = int(np.ceil(target_len / len(y)))
        return np.tile(y, reps)[:target_len]

    def activity_envelope(self, narr: np.ndarray, sr: int, frame_ms: int, attack_ms: int, release_ms: int) -> np.ndarray:
        # Short-time RMS → normalized → binary activity
        frame_len = max(256, int(sr * frame_ms / 1000))
        hop_len = max(128, frame_len // 2)
        rms = librosa.feature.rms(y=narr, frame_length=frame_len, hop_length=hop_len, center=True)[0]
        if rms.max() > 0:
            rms_norm = rms / rms.max()
        else:
            rms_norm = rms

        # Adaptive threshold: keep narration leading
        thr = max(0.02, float(np.quantile(rms_norm, 0.6)))
        act = (rms_norm > thr).astype(np.float32)

        # Upsample to per-sample
        act_samples = np.repeat(act, hop_len)
        if len(act_samples) < len(narr):
            act_samples = np.pad(act_samples, (0, len(narr) - len(act_samples)), mode="edge")
        else:
            act_samples = act_samples[:len(narr)]

        # Attack/Release smoothing (envelope follower)
        attack_a = np.exp(-1.0 / max(1, int(sr * attack_ms / 1000)))
        release_a = np.exp(-1.0 / max(1, int(sr * release_ms / 1000)))
        env = np.zeros_like(act_samples, dtype=np.float32)
        prev = 0.0
        for i, x in enumerate(act_samples):
            if x > prev:
                prev = attack_a * prev + (1 - attack_a) * x
            else:
                prev = release_a * prev + (1 - release_a) * x
            env[i] = prev
        return env

    def merge_narration_background_music(self,
            narration_path: str,
            music_path: str,
            output_path: str,
            # Make narration clearly dominant:
            music_base_db: float = -14.0,  # constant under-bed level (always applied)
            duck_extra_db: float = -10.0,  # additional reduction during speech
            frame_ms: int = 160,
            attack_ms: int = 60,
            release_ms: int = 220,
            final_fade_ms: int = 300,
    ):
        if not os.path.exists(narration_path):
            raise FileNotFoundError(f"Narration not found: {narration_path}")
        if not os.path.exists(music_path):
            raise FileNotFoundError(f"Music not found: {music_path}")

        # Load (mono); keep narration sample rate as master
        narr, sr = librosa.load(narration_path, sr=None, mono=True)
        music, _ = librosa.load(music_path, sr=sr, mono=True)

        # Length match
        music = self.tile_or_trim(music, len(narr))

        # Build smoothed activity envelope (0..1)
        env = self.activity_envelope(narr, sr, frame_ms=frame_ms, attack_ms=attack_ms, release_ms=release_ms)

        # Convert dB params to gains
        base_gain = self.db_to_amp(music_base_db)  # e.g., -14 dB → ~0.20
        duck_gain = self.db_to_amp(duck_extra_db)  # e.g., -10 dB → ~0.32

        # Music gain curve: base under-bed, and when env→1 apply extra duck
        # gain = base * [ 1 - env * (1 - duck_gain) ]
        gain_curve = base_gain * (1.0 - env * (1.0 - duck_gain))

        # Apply gain to music, add narration on top
        music_ducked = (music * gain_curve).astype(np.float32)
        mix = narr.astype(np.float32) + music_ducked

        # Soft-clip / normalize to avoid peaks
        peak = float(np.max(np.abs(mix)) + 1e-12)
        if peak > 1.0:
            mix = (mix / peak).astype(np.float32)

        # Gentle intro/outro fades
        mix = self.fade_in_out(mix, sr, fade_ms=final_fade_ms)

        # Save
        out_path = self.ensure_output_path(output_path)
        sf.write(out_path, mix, sr)
        print(f"✅ Final mix saved: {out_path}")

        # Return duration in seconds
        duration_sec = len(mix) / sr
        print(f"⏱️ Final audio duration: {duration_sec:.2f} seconds")

        return duration_sec

    def merge_wavs(self, df: pandas.DataFrame, crossfade_ms: int = 100, silence_ms: int = 500) -> None:
        """
        Merge multiple wav files listed in a DataFrame into one file with smooth transitions and natural pauses.

        Args:
            df (pd.DataFrame): DataFrame with column "Narration with Background Music Output Path".
            crossfade_ms (int): Crossfade duration in milliseconds between clips.
            silence_ms (int): Silence duration (pause) in milliseconds between clips.
        """

        final_audio = None

        for idx, row in df.iterrows():
            file_path = row["Narration with Background Music Output Path"]

            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                continue

            # Load wav file
            audio = AudioSegment.from_wav(file_path)
            audio = audio.fade_in(200)  # 200ms smooth fade-in

            if final_audio is None:
                final_audio = audio
            else:
                # Add a silence gap before appending next clip
                final_audio += AudioSegment.silent(duration=silence_ms)
                final_audio = final_audio.append(audio, crossfade=crossfade_ms)

            print(f"✅ Added: {file_path}")

        # Export final audio
        if final_audio:
            base_dir = os.path.dirname(self.output_csv_file_path_dialogue)
            final_audio_output_path = os.path.join(base_dir, "final_hindi.wav").replace('\\', '/')
            final_audio.export(final_audio_output_path, format="wav")
            print(f"🎵 Final merged audio saved at: {final_audio_output_path}")
        else:
            print("⚠️ No audio files merged!")

        return

    def change_language_iterrator(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert Hindi text in Emotion, Background Activity, and Musical Prompt
        columns into English for all rows in the DataFrame.
        """
        target_columns = ["Emotion", "Background Activity", "Musical Prompt"]

        for col in target_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: self.change_language(x))

        print(f"Saving language converted dialogues to {self.output_csv_file_path_dialogue}")
        df.to_csv(str(self.output_csv_file_path_dialogue), index=False, encoding="utf-8")
        return df

    def change_language(self, text: str) -> str:
        """
        Convert Hindi text into English while keeping tone and meaning intact.
        """

        system_prompt = """You are a professional translator.
        Your task:
        - Translate the given Hindi text into fluent, natural English.
        - Preserve tone, meaning, and cultural nuance.
        - Keep the translation concise and audience-friendly.
        - Do not add explanations, output only the translated text.
        """

        user_prompt = f"""Translate the following Hindi text into English:
        \"\"\"{text}\"\"\""""

        response = self.open_ai_client.responses.create(
            model=self.open_ai_text_model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.output_text.strip()

    def convert_hindi_dialogues_to_english_iterator(self, df: pandas.DataFrame, english_csv_path) -> pandas.DataFrame:
        """
        :param df: Dataframe consists of ['Actor', 'Dialogue', 'Emotion', 'Background Activity', 'Dialogue Enhanced']
        :return: Return a new column ['Dialogue Hindi'] translated from 'Dialogue Enhanced'
        """

        # Create a copy of the dataframe to avoid modifying the original
        new_df = df.copy()

        for i in tqdm(range(len(new_df)), desc="Converting Hindi dialogue to English", total=len(new_df)):
            dialogue = new_df.iloc[i]["Dialogue Enhanced"]
            english_dialogue = self.convert_hindi_dialogues_to_english(dialogue=dialogue)

            # Debug print
            print(f"Dialogue {i + 1} (HI): {dialogue}")
            print(f"Dialogue {i + 1} (EN): {english_dialogue}\n{'-' * 50}")

            # Update the Dialogue Enhanced column in the new dataframe
            new_df.at[i, "Dialogue Enhanced"] = english_dialogue

        # Construct a new file name for Hindi dialogues
        print(f"Saving English dialogues to {english_csv_path}")
        new_df.to_csv(english_csv_path, index=False, encoding="utf-8")
        print(f"English dialogues saved to {english_csv_path}")

        return new_df

    def convert_hindi_dialogues_to_english(self, dialogue: str) -> str:
        """
        Translate Hindi dialogue into smooth, natural English suitable for narration/drama.
        """

        # System prompt with strict rule
        system_prompt = """You are a skilled translator for audio drama dialogues.
        Given a dialogue in Hindi, translate it into natural, expressive English.
    
        Rules:
        - Output ONLY the English dialogue.
        - Do not give explanations, notes, or extra text.
        - Preserve the emotional tone of the dialogue.
        - Keep it concise and natural.
    
        Example translations:
        Hindi: "हमें साहस के साथ लड़ना होगा, क्योंकि सत्य हमारे पक्ष में है।"
        English: "We must fight with courage, for the truth is on our side."
    
        Hindi: "डरो मत मेरे मित्र, क्योंकि अंधकार के बाद हमेशा प्रकाश आता है।"
        English: "Do not fear, my friend, for light always follows darkness."
        """

        # User input
        user_prompt = f"Hindi: {dialogue}\nEnglish:"

        response = self.open_ai_client.responses.create(
            model=self.open_ai_text_model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.output_text.strip()

if __name__ == '__main__' :

    # Parse the arguments
    args = get_arguments()

    # Loading the details from the arguments
    pdf_path = args.pdf_path
    pdf_start_page = args.pdf_start_page
    pdf_end_page = args.pdf_end_page
    device = args.device

    output_csv_file_name_extracted_text = args.output_csv_file_name_extracted_text

    openai_api_key = args.openai_api_key
    open_ai_text_model = args.open_ai_text_model

    output_csv_file_name_dialogue_english = args.output_csv_file_name_dialogue_english
    output_csv_file_name_dialogue_hindi = args.output_csv_file_name_dialogue_hindi

    reference_wav_dir_english = args.reference_wav_dir_english
    reference_wav_dir_hindi = args.reference_wav_dir_hindi

    tts_model_name = args.tts_model_name
    background_music_model = args.background_music_model

    # Print the arguments
    print(f"PDF Path: {pdf_path}")
    print(f"PDF Start Page: {pdf_start_page}")
    print(f"PDF End Page: {pdf_end_page}")
    print(f"Device: {device}")

    print(f"Output CSV File Name: {output_csv_file_name_extracted_text}")

    print(f"OpenAI API Key: {openai_api_key}")
    print(f"OpenAI Text Model: {open_ai_text_model}")

    print(f"Output CSV File Name Dialogue English: {output_csv_file_name_dialogue_english}")
    print(f"Output CSV File Name Dialogue Hindi: {output_csv_file_name_dialogue_hindi}")
    print(f"Reference WAV English: {reference_wav_dir_english}")
    print(f"Reference WAV Hindi: {reference_wav_dir_hindi}")

    print(f"TTS Model Name: {tts_model_name}")
    print(f"Background Music Model : {background_music_model}")

    # Sanity check for the arguments
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Define the chapter mapping
    chapter_mapping = {  # key is the chapter name, value is a list of start and end pages
        'chapter_1': [pdf_start_page, pdf_end_page],
    }

    # Ensure start_page and end_page are valid
    for chapter_name, (start_page, end_page) in chapter_mapping.items():
        if not isinstance(start_page, int) or not isinstance(end_page, int):
            raise ValueError(f"Invalid page numbers for {chapter_name}: start_page and end_page must be integers.")
        # Ensure start_page and end_page are positive integers
        if start_page < 1 or end_page < 1 or end_page < start_page:
            raise ValueError(f"Invalid page numbers for {chapter_name}: start_page and end_page must be >= 1.")

    for chapter_name, (start_page, end_page) in chapter_mapping.items():

        print(f"Processing {chapter_name} from page {start_page} to {end_page}")

        print(f"Starting with the Hindi Narration, i.e. base language of book")

        # Creating the directory to save the CSV file path
        base_dir = os.path.dirname(pdf_path)
        base_dir = os.path.join(base_dir, f"{chapter_name}").replace('\\', '/')
        os.makedirs(base_dir, exist_ok= True)

        # Creating the output CSV file path in base directory
        output_csv_file_path_extracted_text = os.path.join(base_dir, f"{output_csv_file_name_extracted_text}").replace('\\', '/')
        output_csv_file_path_dialogue_english = os.path.join(base_dir, f"{output_csv_file_name_dialogue_english}").replace('\\','/')
        output_csv_file_path_dialogue_hindi = os.path.join(base_dir, f"{output_csv_file_name_dialogue_hindi}").replace('\\', '/')

        # Print Arguments
        print(f"Extracted text csv path : {output_csv_file_path_extracted_text}")
        print(f"Dialogue English csv path : {output_csv_file_path_dialogue_english}")
        print(f"Dialogue Hindi csv path : {output_csv_file_name_dialogue_hindi}")

        # Create an instance of the EnglishNarration class
        narration_hindi = HindiNarration(
            pdf_path=pdf_path,
            pdf_start_page=start_page,
            pdf_end_page=end_page,
            device=device,
            extract_text_csv_file_path= output_csv_file_path_extracted_text,
            openai_api_key=openai_api_key,
            open_ai_text_model=open_ai_text_model,
            output_csv_file_path_dialogue=output_csv_file_path_dialogue_hindi,
            reference_wav_dir =reference_wav_dir_hindi,
            tts_model_name=tts_model_name,
            background_music_model= background_music_model
        )

        print("Step 1: Extracting text from PDF")
        df = narration_hindi.extract_text_from_pdf()

        print(f"Step 2 : Cleaning the extractd text saved in {narration_hindi.extract_text_csv_file_path}")
        df = narration_hindi.clean_ocr_text_iterrator(df)

        print(f"Step 3: Converting cleaned text to narration dialogues ...")
        df = narration_hindi.generate_script_iterrator(df)

        print("Step 4: Converting narration to enhanced narration...")
        df = narration_hindi.convert_narration_to_enhanced_narration_iterrator(df)

        print("Step 5: Converting background activities to musical prompt...")
        df = narration_hindi.convert_background_activites_and_dialogues_to_musical_prompt_iterrator(df)

        print("Step 6: Narration Validation")
        narration_hindi.narration_check(df= df)

        print("Step 6.1: Changing language from Hindi to English for Emotion, Background Activity and Musical Prompt...")
        df = narration_hindi.change_language_iterrator(df=df)

        print("Step 7 : Generate a introduction for the story to begin with ...")
        generator = StoryIntroGenerator(openai_api_key=openai_api_key,
                                        openai_text_model=open_ai_text_model,
                                        df=pd.read_csv(narration_hindi.output_csv_file_path_dialogue))
        story_intro_paragraph = generator.generate_story_intro_from_csv()
        story_intro_paragraph += " अब, आइए कहानी में प्रवेश करते हैं।"
        print("Story Intro:\n", story_intro_paragraph)

        print("Step 8 : Generate a takeaway less for the story to end with ...")
        story_takeaway_paragraph = generator.generate_story_takeaway_from_csv()
        story_takeaway_paragraph = "इस कहानी से मुख्य सीख यह है: " + story_takeaway_paragraph
        print("Story Takeaway:\n", story_takeaway_paragraph)

        # Load existing narration file
        df = pd.read_csv(narration_hindi.output_csv_file_path_dialogue)

        # Intro row
        intro_row = pd.DataFrame({
            "Dialogue Enhanced": [story_intro_paragraph],
            "Background Activity": ["A gentle dawn, with soft light spreading across the horizon."],
            "Emotion": ["Anticipation, warmth, and curiosity."],
            "Musical Prompt": ["Calm sitar with soft flute and light strings, uplifting and inviting."]
        })

        # Takeaway row
        takeaway_row = pd.DataFrame({
            "Dialogue Enhanced": [story_takeaway_paragraph],
            "Background Activity": ["The sun setting slowly, with calm winds in the background."],
            "Emotion": ["Reflection, peace, and inner clarity."],
            "Musical Prompt": ["Gentle piano with warm cello and fading harp, thoughtful and soothing."]
        })

        # Combine intro, main dialogues, and takeaway
        df = pd.concat([intro_row, df, takeaway_row], ignore_index=True)
        # Save updated CSV
        df.to_csv(narration_hindi.output_csv_file_path_dialogue, index=False, encoding="utf-8")
        print(f"Updated CSV with intro and takeaway saved to {narration_hindi.output_csv_file_path_dialogue}")

        print("Step 9: Converting narration dialogues to speech using XTTS v2...")
        torch.cuda.empty_cache()
        df = narration_hindi.convert_text_to_speech_iterrator(df)

        print("Step 10: Generating background music for each narration dialogue...")
        torch.cuda.empty_cache()
        df = narration_hindi.generate_background_music_iterator(df)

        print("Step 11: Merging the narration and background music ...")
        df = narration_hindi.merge_narration_background_music_iterrator(df)

        print("Step 12 : Need to merge all different chunks into a single file ...")
        narration_hindi.merge_wavs(df)

        # Clearning the cache memory and object which are initialized before
        narration_hindi = None
        torch.cuda.empty_cache()
        gc.collect()

        print(f"Completed Narration for  {chapter_name} from page {start_page} to {end_page}")

