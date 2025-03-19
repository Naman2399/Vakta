# import uuid
# import mysql.connector
#
# # Database connection
# conn = mysql.connector.connect(
#     host="localhost",
#     user="root",       # Replace with your MySQL username
#     password="Naman@23",  # Replace with your MySQL password
#     database="test"    # Replace with your MySQL database name
# )
#
# cursor = conn.cursor()
#
# # Create table with UUID as primary key
# cursor.execute("""
#     CREATE TABLE IF NOT EXISTS users (
#         id CHAR(36) PRIMARY KEY,
#         name VARCHAR(100) NOT NULL
#     )
# """)
#
# # Insert data with UUID as the primary key
# user_id = str(uuid.uuid4())  # Generate UUID
# name = "John Doe"
#
# cursor.execute("INSERT INTO users (id, name) VALUES (%s, %s)", (user_id, name))
# conn.commit()
#
# # Fetch and print records
# cursor.execute("SELECT * FROM users")
# for row in cursor.fetchall():
#     print(row)
#
# # Close connection
# cursor.close()
# conn.close()


# import mysql.connector
#
# # Database connection
# conn = mysql.connector.connect(
#     host="localhost",
#     user="root",       # Replace with your MySQL username
#     password="Naman@23",  # Replace with your MySQL password
#     database="vakta"    # Replace with your MySQL database name
# )
# cursor = conn.cursor()
#
# # Step 1: Find all foreign key constraints on user_id
# cursor.execute("""
#     SELECT TABLE_NAME, CONSTRAINT_NAME
#     FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
#     WHERE COLUMN_NAME = 'user_id'
#     AND REFERENCED_TABLE_NAME IS NOT NULL
#     AND TABLE_SCHEMA = DATABASE();
# """)
# fk_constraints = cursor.fetchall()
#
# # Step 2: Drop foreign key constraints
# for table_name, constraint_name in fk_constraints:
#     drop_fk_query = f"ALTER TABLE {table_name} DROP FOREIGN KEY {constraint_name};"
#     print(f"Dropping foreign key: {drop_fk_query}")
#     cursor.execute(drop_fk_query)
#
# # Step 3: Find all tables containing user_id column
# cursor.execute("""
#     SELECT TABLE_NAME
#     FROM INFORMATION_SCHEMA.COLUMNS
#     WHERE COLUMN_NAME = 'user_id' AND TABLE_SCHEMA = DATABASE();
# """)
# tables = cursor.fetchall()
#
# # Step 4: Modify user_id column in all tables
# for (table_name,) in tables:
#     alter_query = f"ALTER TABLE {table_name} MODIFY user_id CHAR(36);"
#     print(f"Modifying column: {alter_query}")
#     cursor.execute(alter_query)
#
# # Step 5: Recreate foreign keys
# for table_name, constraint_name in fk_constraints:
#     add_fk_query = f"""
#         ALTER TABLE {table_name}
#         ADD CONSTRAINT {constraint_name}
#         FOREIGN KEY (user_id) REFERENCES books(user_id);
#     """
#     print(f"Recreating foreign key: {add_fk_query}")
#     cursor.execute(add_fk_query)
#
# # Commit changes
# conn.commit()
#
# # Close connection
# cursor.close()
# conn.close()
#
# print("Successfully modified 'user_id' and restored foreign keys.")

import mysql.connector

# Database Configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Naman@23",
    "database": "vakta"
}

# Connect to MySQL
conn = mysql.connector.connect(**DB_CONFIG)
cursor = conn.cursor()

# Step 1: Get all Foreign Keys in the Database
cursor.execute("""
    SELECT TABLE_NAME, CONSTRAINT_NAME
    FROM information_schema.TABLE_CONSTRAINTS
    WHERE CONSTRAINT_TYPE = 'FOREIGN KEY' AND TABLE_SCHEMA = %s
""", (DB_CONFIG["database"],))

foreign_keys = cursor.fetchall()

# Step 2: Drop Foreign Keys
drop_fk_queries = []
for table, fk_name in foreign_keys:
    drop_fk_queries.append(f"ALTER TABLE {table} DROP FOREIGN KEY {fk_name};")

# Execute drop FK queries
for query in drop_fk_queries:
    try:
        cursor.execute(query)
    except mysql.connector.Error as err:
        print(f"Error dropping FK {query}: {err}")

# Step 3: Convert Entire Database to UTF8MB4
cursor.execute(f"ALTER DATABASE {DB_CONFIG['database']} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")

# Step 4: Modify Character Set for All Tables
cursor.execute("""
    SELECT TABLE_NAME
    FROM information_schema.TABLES
    WHERE TABLE_SCHEMA = %s
""", (DB_CONFIG["database"],))

tables = cursor.fetchall()

for (table,) in tables:
    alter_table_query = f"ALTER TABLE {table} CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
    try:
        cursor.execute(alter_table_query)
    except mysql.connector.Error as err:
        print(f"Error modifying table {table}: {err}")

# Step 5: Modify All VARCHAR & TEXT Columns to UTF8MB4
cursor.execute("""
    SELECT TABLE_NAME, COLUMN_NAME, COLUMN_TYPE
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = %s AND DATA_TYPE IN ('varchar', 'text')
""", (DB_CONFIG["database"],))

columns = cursor.fetchall()

for table, column, col_type in columns:
    modify_column_query = f"ALTER TABLE {table} MODIFY {column} {col_type} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
    try:
        cursor.execute(modify_column_query)
    except mysql.connector.Error as err:
        print(f"Error modifying column {column} in table {table}: {err}")

# Step 6: Restore Foreign Keys
cursor.execute("""
    SELECT TABLE_NAME, COLUMN_NAME, CONSTRAINT_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
    FROM information_schema.KEY_COLUMN_USAGE
    WHERE TABLE_SCHEMA = %s AND CONSTRAINT_NAME != 'PRIMARY'
""", (DB_CONFIG["database"],))

restore_fk_queries = []
for table, column, fk_name, ref_table, ref_column in cursor.fetchall():
    restore_fk_queries.append(
        f"ALTER TABLE {table} ADD CONSTRAINT {fk_name} FOREIGN KEY ({column}) REFERENCES {ref_table}({ref_column}) ON DELETE CASCADE ON UPDATE CASCADE;"
    )

# Execute restore FK queries
for query in restore_fk_queries:
    try:
        cursor.execute(query)
    except mysql.connector.Error as err:
        print(f"Error restoring FK {query}: {err}")

# Commit changes and close connection
conn.commit()
cursor.close()
conn.close()

print("Database character set updated successfully with foreign key constraints preserved!")

