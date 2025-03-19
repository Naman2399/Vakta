import uuid
import mysql.connector

# Database Configuration (Update with your credentials)
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Naman@23",  # Update with your MySQL password
    "database": "vakta",
    "charset" : "utf8mb4"
}

# Connect to MySQL
conn = mysql.connector.connect(**DB_CONFIG)
cursor = conn.cursor()

# Author Details
author_name = "Ved Vyasa (Commentary by Jayadayal Goyandka & Hanuman Prasad Poddar)"
author_id = str(uuid.uuid4())  # Generate UUID

# Check if the author already exists
check_query = "SELECT author_id FROM authors WHERE name = %s"
cursor.execute(check_query, (author_name,))
existing_author = cursor.fetchone()

if not existing_author:
    # Insert Author if not exists
    insert_query = "INSERT INTO authors (author_id, name) VALUES (%s, %s)"
    cursor.execute(insert_query, (author_id, author_name))
    conn.commit()
    print(f"Inserted Author: {author_name} with ID: {author_id}")
else:
    print(f"Author already exists in the database with ID: {existing_author[0]}")

# Close connection
cursor.close()
conn.close()
