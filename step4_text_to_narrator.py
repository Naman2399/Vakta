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
        skipping_lines = 0
        for line in script.split("\n"):
            parts = line.strip().split("<break>")
            if len(parts) == 4:
                dialogues.append(parts)
            else :
                print(f"Skipping line due to unexpected format: {line}")
                skipping_lines += 1

        print(f"Skipped {skipping_lines} lines due to unexpected format.")

    # Create a DataFrame from the dialogues
    dialogues_df = pd.DataFrame(dialogues, columns=['Actor', 'Dialogue', 'Emotion', 'Background Activities'])
    # Ensure the output directory exists
    output_dir = os.path.dirname(input_dir).replace('cleaned_text', 'narration')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Save the dialogues DataFrame to a CSV file
    output_file = os.path.join(output_dir, os.path.basename(input_dir).replace('.csv', '_narration.csv'))

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