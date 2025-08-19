import argparse
import os

from openai import OpenAI
import pandas as pd
from tqdm import tqdm

from config.open_ai_config import API_KEY, TEXT_MODEL


def get_arguments():
    parser = argparse.ArgumentParser(description="Convert cleaned text to dialogues using OpenAI API.")
    parser.add_argument('--input_dir', type=str, default = 'books/sample/narration/idx_1_chapter_1_narration.csv', help='Directory containing cleaned text CSV files.')
    parser.add_argument('--openai_api_key', type=str, default=API_KEY, help='OpenAI API key for text cleaning')
    parser.add_argument('--open_ai_text_model', type=str, default=TEXT_MODEL, help='OpenAI text model to use for cleaning')
    return parser.parse_args()

def convert_page_to_script(client, text_model, current_text, next_text):
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
    for i in tqdm(range(len(df) - 1), desc="Converting narration to enhanced narration", total=len(df) - 1):

        # For each page, we need the previous page's context
        current_text = df.iloc[i]["Dialogue"]
        next_text = df.iloc[i + 1]["Dialogue"]

        script = convert_page_to_script(client, text_model, current_text, next_text)

        # Print content of script for debugging
        print(f"Script for page {i + 1}:\n{script}\n{'-'*50}")

        # Convert the output text to a strucutred format -
        skipping_lines = 0
        for line in script.split("\n"):
            parts = line.strip().split("<break>")
            if len(parts) == 3:
                dialogues.append(parts)

                # For the first updat the dataframe
                if i <= len(df) - 2 :
                    df.iloc[i] = parts[1]
                else :
                    df.iloc[i] = parts[1]
                    df.iloc[i + 1] = parts[2]

            else :
                print(f"Skipping line due to unexpected format: {line}")
                skipping_lines += 1

        print(f"Skipped {skipping_lines} lines due to unexpected format.")

    # Ensure the output directory exists
    output_dir = os.path.dirname(input_dir).replace('narration', 'narration_enhanced')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Save the dialogues DataFrame to a CSV file
    output_file = os.path.join(output_dir, os.path.basename(input_dir).replace('.csv', '_enhanced.csv'))

    # Save the dialogues DataFrame to a CSV file
    print(f"Saving dialogues to {output_file}")
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Dialogues saved to {output_file}")