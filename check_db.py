import sqlite3
conn = sqlite3.connect("data/chat_logs/sessions.db")
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("Tables:", tables)
for (t,) in tables:
    count = conn.execute("SELECT COUNT(*) FROM " + t).fetchone()[0]
    print("  " + t + ": " + str(count) + " rows")

print("\nLast 5 sessions:")
rows = conn.execute("SELECT session_id, user_id, started_at FROM sessions ORDER BY rowid DESC LIMIT 5").fetchall()
for r in rows:
    print(" ", r)

if conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0] > 0:
    print("\nSample turns:")
    rows = conn.execute("SELECT id, session_id, user_id, turn_number, user_input FROM turns LIMIT 5").fetchall()
    for r in rows:
        print(" ", r)

conn.close()
