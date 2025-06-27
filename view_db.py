import sqlite3

def show_users():
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row  # Optional: access by column name
    cur = conn.cursor()
    cur.execute("SELECT * FROM users")
    rows = cur.fetchall()
    for row in rows:
     print(dict(row))
    conn.close()

show_users()

