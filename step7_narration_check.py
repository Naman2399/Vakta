import pandas as pd

import argparse

def get_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Check narration text for consistency.')
    parser.add_argument('--input_dir', type=str, default='books/sample/musical_prompt/idx_1_chapter_1_narration_enhanced_musical_prompt.csv', help='Path to the input CSV file')
    return parser.parse_args()

if __name__ == "__main__":

    # Parse command line arguments
    args = get_arguments()

    input_dir = args.input_dir

    # Read the narration CSV file
    df = pd.read_csv(input_dir)

    # Check for missing values in 'Dialogue' and 'Emotion' columns
    missing_dialogue = df['Dialogue'].isnull().sum()
    missing_emotion = df['Emotion'].isnull().sum()

    if missing_dialogue > 0:
        print(f"Warning: {missing_dialogue} rows have missing 'Dialogue'.")
    else:
        print("All 'Dialogue' entries are present.")

    if missing_emotion > 0:
        print(f"Warning: {missing_emotion} rows have missing 'Emotion'.")
    else:
        print("All 'Emotion' entries are present.")

    # Check for empty strings in 'Dialogue' and 'Emotion' columns
    empty_dialogue = (df['Dialogue'] == '').sum()
    empty_emotion = (df['Emotion'] == '').sum()

    if empty_dialogue > 0:
        print(f"Warning: {empty_dialogue} rows have empty 'Dialogue'.")
    else:
        print("No empty 'Dialogue' entries found.")

    if empty_emotion > 0:
        print(f"Warning: {empty_emotion} rows have empty 'Emotion'.")
    else:
        print("No empty 'Emotion' entries found.")

    # Print complete narration statistics
    total_rows = len(df)
    print(f"Total rows in narration: {total_rows}")
    print(f"Rows with missing 'Dialogue': {missing_dialogue}")
    print(f"Rows with missing 'Emotion': {missing_emotion}")
    print(f"Rows with empty 'Dialogue': {empty_dialogue}")
    print(f"Rows with empty 'Emotion': {empty_emotion}")

    # Print complete narration content
    print("\nComplete Narration Content:")
    for index, row in df.iterrows():
        # print(f"Row {index + 1}: Actor: {row['Actor']}, Dialogue: {row['Dialogue']}, Emotion: {row['Emotion']}, Background Activities: {row['Background Activities']}")
        print(f"Row {index + 1} Dialogue: {row['Dialogue']}")
