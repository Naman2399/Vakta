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
chapters_df = pd.read_csv("../data/1/updated_csv/chapters.csv")  # Contains chapter_number, chapter_title, book_id, chapter_id
sections_df = pd.read_csv("../data/1/raw_csv/sections.csv")  # Contains section_number, chapter_number, section_title

# Create a mapping of chapter_number -> chapter_id
chapter_mapping = dict(zip(chapters_df["chapter_number"], chapters_df["chapter_id"]))

# Add chapter_id to sections_df based on chapter_number
sections_df["chapter_id"] = sections_df["chapter_number"].map(chapter_mapping)

# Function to generate a unique UUID and ensure it does not exist in the database
def generate_unique_section_id():
    while True:
        section_id = str(uuid.uuid4())
        cursor.execute("SELECT COUNT(*) FROM sections WHERE section_id = %s", (section_id,))
        if cursor.fetchone()[0] == 0:
            return section_id

# Generate unique section_id for each row
sections_df["section_id"] = [generate_unique_section_id() for _ in range(len(sections_df))]

# Insert sections into the database
for index, row in sections_df.iterrows():
    insert_query = """
        INSERT INTO sections (section_id, chapter_id, section_number, title)
        VALUES (%s, %s, %s, %s)
    """
    cursor.execute(insert_query, (row["section_id"], row["chapter_id"], row["section_number"], row["section_title"]))

# Commit and close connection
conn.commit()
cursor.close()
conn.close()

# Save the updated sections.csv
sections_df.to_csv("../data/1/updated_csv/sections.csv", index=False)

print("Sections inserted successfully and CSV updated!")
