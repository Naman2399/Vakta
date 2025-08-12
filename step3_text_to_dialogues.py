import argparse
import os

from openai import OpenAI
import pandas as pd
from tqdm import tqdm

from config.open_ai_config import API_KEY, TEXT_MODEL


def get_arguments():
    parser = argparse.ArgumentParser(description="Convert cleaned text to dialogues using OpenAI API.")
    parser.add_argument('--input_dir', type=str, default = 'books/sample/cleaned_text/idx_1_chapter_1.csv', help='Directory containing cleaned text CSV files.')
    parser.add_argument('--openai_api_key', type=str, default=API_KEY, help='OpenAI API key for text cleaning')
    parser.add_argument('--open_ai_text_model', type=str, default=TEXT_MODEL, help='OpenAI text model to use for cleaning')
    return parser.parse_args()

def convert_page_to_script(client, text_model, prev_text, current_text):
    system_prompt = """You are an assistant that converts plain text into a structured dramatic script.

Strict Rules:
1. Process the given text and turn it into a scene.
2. Identify suitable characters (e.g., Narrator, main characters from the text).
3. Break the scene into dialogues.
4. For each dialogue, produce exactly 4 fields in this order:
   - Actor: Narrator or Character name
   - Dialogue: Spoken lines with no commas, no special characters, and no quotation marks
   - Emotion: A single emotion word (e.g., Calm, Angry, Joyful, Thoughtful) without commas or symbols
   - Background Activities: Short description of what is happening in the scene, no commas or symbols
5. Maintain the meaning of the original text but present it in a natural dramatic flow.
6. Output must be plain text, one dialogue per line, in the exact format:
   Actor,Dialogue,Emotion,Background Activities
7. Do not include any extra commentary, explanations, headers, numbering, or formatting.
8. Do not insert commas inside any of the four fields.
9. If the content naturally contains commas, replace them with semicolons or rephrase to avoid commas."""

    user_prompt = f"""Convert the following text into a structured drama script as per the rules.

Text for this page (with 50 words from the previous page for context):
\"\"\"{prev_text} {current_text}\"\"\""""

    response = client.responses.create(
        model=text_model,  # Cheap and good quality
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
    if not os.path.isfile(input_dir):
        raise FileNotFoundError(f"The specified input file does not exist: {input_dir}")

    # Read the cleaned text CSV file
    df = pd.read_csv(input_dir)

    # Convert each page of text to a script format
    dialogues = []
    for i in tqdm(range(len(df) - 1), desc="Converting text to dialogues", total=len(df) - 1):
        # For each page, we need the previous page's context
        if i > 0 :
            # Use the last 50 words of the previous text for context
            prev_text = ' '.join(df.iloc[i - 1]['content'].split()[-50:])
        else:
            # If it's the first page, no previous context
            prev_text = ""

        current_text = df.iloc[i + 1]['content']
        script = convert_page_to_script(client, text_model, prev_text, current_text)

        # Print content of script for debugging
        print(f"Script for page {i + 1}:\n{script}\n{'-'*50}")

        # Convert the output text to a strucutred format -
        for line in script.split("\n"):
            if line.strip():
                parts = [p.strip() for p in line.split(",", 3)]
                if len(parts) == 4:
                    dialogues.append(parts)

    # Create a DataFrame from the dialogues
    dialogues_df = pd.DataFrame(dialogues, columns=['Actor', 'Dialogue', 'Emotion', 'Background Activities'])
    # Ensure the output directory exists
    output_dir = os.path.dirname(input_dir).replace('cleaned_text', 'dialogues')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Save the dialogues DataFrame to a CSV file
    output_file = os.path.join(output_dir, os.path.basename(input_dir).replace('.csv', '_dialogues.csv'))

    # Read the dialogues DataFrame to a CSV file
    for dialogue in dialogues :
        actor, dialogue_text, emotion, background = dialogue[0], dialogue[1], dialogue[2], dialogue[3]
        # Add input of actor, dialogue_text, emotion, background to the dialgoues_df dataframe
        dialogues_df = dialogues_df.append({
            'Actor': actor,
            'Dialogue': dialogue_text,
            'Emotion': emotion,
            'Background Activities': background
        }, ignore_index=True)

    # Save the dialogues DataFrame to a CSV file
    print(f"Saving dialogues to {output_file}")
    dialogues_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Dialogues saved to {output_file}")