import pandas as pd
import uuid
import mysql.connector

# Database Configuration (Update as needed)
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

# Load Books CSV
books_df = pd.read_csv("../data/1/updated_csv/books.csv")

# Load Chapters CSV
chapters_df = pd.read_csv("../data/1/raw_csv/chapters.csv")

# Fetch book_id (Assuming only one book in books.csv)
book_id = books_df["book_id"].iloc[0]

# Function to check if chapter_id exists
def chapter_id_exists(chapter_id):
    query = "SELECT 1 FROM chapters WHERE chapter_id = %s"
    cursor.execute(query, (chapter_id,))
    return cursor.fetchone() is not None  # Returns True if exists, False otherwise

# Generate unique chapter_id
def get_unique_chapter_id():
    while True:
        new_uuid = str(uuid.uuid4())
        if not chapter_id_exists(new_uuid):
            return new_uuid

# Add book_id and generate unique chapter_id
chapters_df["book_id"] = book_id
chapters_df["chapter_id"] = [get_unique_chapter_id() for _ in range(len(chapters_df))]

# Insert chapters into the database
for index, row in chapters_df.iterrows():
    insert_query = "INSERT INTO chapters (chapter_id, book_id, chapter_number, title) VALUES (%s, %s, %s, %s)"
    cursor.execute(insert_query, (row["chapter_id"], row["book_id"], row["chapter_number"], row["chapter_title"]))

# Commit and close connection
conn.commit()
cursor.close()
conn.close()

# Save updated CSV for future use
chapters_df.to_csv("../data/1/chapters.csv", index=False)

print("Chapters inserted successfully!")
