import argparse
import csv
import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from config.open_ai_config import API_KEY, TEXT_MODEL


def get_arguments() :

    parser = argparse.ArgumentParser(description='Convert the text from the PDF to a clean text csv file.')
    parser.add_argument('--input_dir', type=str, default='books/sample/chapters', help='Path to the input CSV file with OCR text')
    parser.add_argument('--openai_api_key', type=str, default= API_KEY,  help='OpenAI API key for text cleaning')
    parser.add_argument('--open_ai_text_model', type=str, default=TEXT_MODEL, help='OpenAI text model to use for cleaning')

    return parser.parse_args()

def clean_ocr_text(client, model, ocr_text):
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
    \"\"\"{ocr_text}\"\"\""""

    response = client.responses.create(
        model= model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.output_text.strip()

if __name__ == "__main__":

    # Parse command line arguments
    args = get_arguments()

    input_dir = args.input_dir
    client = OpenAI(api_key=args.openai_api_key)
    text_model = args.open_ai_text_model

    # Ensure the input directory exists
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"The specified input directory does not exist: {input_dir}")

    # Create the output directory if it doesn't exist
    # Say if we have input_dir = 'books/sample/chapters', then output_dir will be 'books/sample/cleaned_text'
    output_dir = os.path.join(os.path.dirname(input_dir), 'cleaned_text').replace('\\', '/')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Input directory: {input_dir}")
    print(f"Output directory created at: {output_dir}")

    # Process each CSV file in the input directory
    for filename in tqdm(os.listdir(input_dir), desc="Processing files", unit="file", total=len(os.listdir(input_dir))):
        if filename.endswith('.csv'):
            input_file_path = os.path.join(input_dir, filename).replace('\\', '/')
            output_file_path = os.path.join(output_dir, filename).replace('\\', '/')

            print(f"Processing file: {input_file_path}")
            print(f"Output will be saved to: {output_file_path}")

            # Input file contains page_number and content columns
            df = pd.read_csv(input_file_path, encoding='utf-8')
            df = df.sort_values(by='page_number')             # Sort all the content by page_number

            with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['page_number', 'content'])

                for index, row in tqdm(df.iterrows(), desc=f"Cleaning {filename}", total=len(df), unit="row"):
                    page_number = row['page_number']
                    ocr_text = row['content']

                    # Clean the OCR text using OpenAI API
                    cleaned_text = clean_ocr_text(client, text_model, ocr_text)

                    # Write the cleaned text to the output CSV file
                    writer.writerow([page_number, cleaned_text])
