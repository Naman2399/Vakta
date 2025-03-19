import pandas as pd
import uuid
import mysql.connector

# Database Configuration (Update with your details)
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

# Read CSV file
csv_file = "../data/1/updated_csv/books.csv"  # Update with the actual file path
df = pd.read_csv(csv_file)

# Function to check if a book exists
def book_exists(isbn, title):
    query = "SELECT book_id FROM books WHERE isbn = %s OR title = %s"
    cursor.execute(query, (isbn, title))
    return cursor.fetchone()  # Returns None if not found

# Insert books into the database
for index, row in df.iterrows():
    existing_book = book_exists(row["isbn"], row["title"])

    # Generate new UUID if book already exists
    book_id = str(uuid.uuid4()) if existing_book is None else existing_book[0]

    # Insert only if book is new
    if existing_book is None:
        insert_query = "INSERT INTO books (book_id, title, isbn, published_date, page_count) VALUES (%s, %s, %s, %s, %s)"
        cursor.execute(insert_query, (book_id, row["title"], row["isbn"], row["published_date"], row["page_count"]))
        conn.commit()

    # Add book_id to DataFrame
    df.at[index, "book_id"] = book_id

# Save updated DataFrame to a new CSV file (optional)
df.to_csv("../data/1/books.csv", index=False)

# Close database connection
cursor.close()
conn.close()

print("Data inserted successfully!")
