import sqlite3

conn = sqlite3.connect("checkpoints.db")

cursor = conn.cursor()

cursor.execute(
    "SELECT * FROM checkpoints"
)

rows = cursor.fetchall()

for row in rows:
    print(row)