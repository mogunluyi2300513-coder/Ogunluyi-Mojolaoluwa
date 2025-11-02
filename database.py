import sqlite3

def init_db():
    conn = sqlite3.connect('emotions.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT,
            emotion TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_prediction(image_name, emotion):
    conn = sqlite3.connect('emotions.db')
    c = conn.cursor()
    c.execute('INSERT INTO predictions (image_name, emotion) VALUES (?, ?)', (image_name, emotion))
    conn.commit()
    conn.close()
