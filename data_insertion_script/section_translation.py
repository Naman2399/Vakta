import pandas as pd
import uuid
import mysql.connector

# Database Configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Naman@23",
    "database": "vakta",
    "charset" : "utf8mb4"
}

# Connect to MySQL
conn = mysql.connector.connect(**DB_CONFIG)
cursor = conn.cursor()

# Load CSV Files
bhagwad_gita_df = pd.read_csv("../data/1/raw/Bhagwad_Gita.csv")  # Contains section_number, chapter_number, Verse, sanskrit, sanskrit_english_words, hindi, english
sections_df = pd.read_csv("../data/1/updated_csv/sections.csv")  # Contains section_number, chapter_number, section_title, chapter_id, section_id

# Create a mapping of section_number -> section_id
section_mapping = dict(zip(sections_df["section_number"], sections_df["section_id"]))

# Add section_id to Bhagwad_Gita DataFrame
bhagwad_gita_df["section_id"] = bhagwad_gita_df["section_number"].map(section_mapping)

# Drop rows where section_id is missing
bhagwad_gita_df.dropna(subset=["section_id"], inplace=True)

# Create separate DataFrames for each language
languages = {
    "sn": "sanskrit",
    "hn": "hindi",
    "en": "english"
}

# Function to check if a translation_id already exists
def translation_exists(translation_id):
    query = "SELECT COUNT(*) FROM section_translations WHERE translation_id = %s"
    cursor.execute(query, (translation_id,))
    return cursor.fetchone()[0] > 0

# Insert Data into MySQL and save updated CSV files
for lang_code, column in languages.items():
    lang_df = bhagwad_gita_df[["section_id", column]].copy()
    lang_df["language_code"] = lang_code
    lang_df.rename(columns={column: "content"}, inplace=True)

    # Generate unique translation_id
    lang_df["translation_id"] = [str(uuid.uuid4()) for _ in range(len(lang_df))]

    # Ensure translation_id is unique in database
    for index, row in lang_df.iterrows():
        while translation_exists(row["translation_id"]):  # Ensure uniqueness
            row["translation_id"] = str(uuid.uuid4())

        # Insert into MySQL
        insert_query = """
        INSERT INTO section_translations (translation_id, section_id, language_code, content)
        VALUES (%s, %s, %s, %s)
        """
        cursor.execute(insert_query, (row["translation_id"], row["section_id"], row["language_code"], row["content"]))

    # Save the updated CSV with translation_id
    lang_df.to_csv(f"../data/1/bhagwad_gita_{lang_code}.csv", index=False)

# Commit and close connection
conn.commit()
cursor.close()
conn.close()

print("Data inserted into section_translations and CSV files updated successfully!")
