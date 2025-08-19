import argparse
import os

from openai import OpenAI
import pandas as pd
from tqdm import tqdm

from config.open_ai_config import API_KEY, TEXT_MODEL


def get_arguments():
    parser = argparse.ArgumentParser(description="Convert cleaned text to dialogues using OpenAI API.")
    parser.add_argument('--input_dir', type=str, default = 'books/sample/narration_enhanced/idx_1_chapter_1_narration_enhanced.csv', help='Directory containing cleaned text CSV files.')
    parser.add_argument('--openai_api_key', type=str, default=API_KEY, help='OpenAI API key for text cleaning')
    parser.add_argument('--open_ai_text_model', type=str, default=TEXT_MODEL, help='OpenAI text model to use for cleaning')
    return parser.parse_args()

def convert_page_to_script(client, text_model, dialogue, background_activities):
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

    # Add new column for Musical Prompt each value to None
    df["Musical Prompt"] = None

    # Convert each page of text to a script format
    dialogues = []
    for i in tqdm(range(len(df) - 1), desc="Converting dialogue + background activites to musical prompt", total=len(df) - 1):

        # For each page, we need the previous page's context
        dialogue = df.iloc[i]["Dialogue"]
        background_activiy = df.iloc[i]["Background Activities"]

        musci_prompt = convert_page_to_script(client, text_model, dialogue, background_activiy)

        # Print content of script for debugging
        print(f"Music prompt for dialogue {i + 1}:\n{musci_prompt}\n{'-'*50}")
        # Add to the dataframe
        df["Musical Prompt"].iloc[i] = musci_prompt

    # Ensure the output directory exists
    output_dir = os.path.dirname(input_dir).replace('narration_enhanced', 'musical_prompt')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Save the dialogues DataFrame to a CSV file
    output_file = os.path.join(output_dir, os.path.basename(input_dir).replace('.csv', '_musical_prompt.csv'))

    # Save the dialogues DataFrame to a CSV file
    print(f"Saving dialogues to {output_file}")
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Dialogues saved to {output_file}")