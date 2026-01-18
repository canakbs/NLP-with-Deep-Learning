import sqlite3
import os

DB_PATH = os.path.join("data", "assistant.db")

def initialize_db():
    """Veritabanını ve gerekli tabloları oluşturur"""
    if not os.path.exists("data"):
        os.makedirs("data")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # interactions tablosu
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # notes tablosu
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # calendar tablosu
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS calendar (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event TEXT NOT NULL,
            event_date DATE NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()


def add_note(content):
    """Not ekler"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO notes (content) VALUES (?)", (content,))
    conn.commit()
    conn.close()


def add_event(event, event_date):
    """Takvime etkinlik ekler"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO calendar (event, event_date) VALUES (?, ?)", (event, event_date))
    conn.commit()
    conn.close()


def get_notes():
    """Tüm notları getirir"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT content, timestamp FROM notes ORDER BY timestamp DESC")
    notes = cursor.fetchall()
    conn.close()
    return notes


def get_events():
    """Tüm etkinlikleri getirir"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT event, event_date FROM calendar ORDER BY event_date ASC")
    events = cursor.fetchall()
    conn.close()
    return events


def delete_note(note_index):
    """Not siler - index kullanarak"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Tüm notları al
    cursor.execute("SELECT id FROM notes ORDER BY timestamp DESC")
    notes = cursor.fetchall()
    
    if 1 <= note_index <= len(notes):
        note_id = notes[note_index - 1][0]
        cursor.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        conn.commit()
    
    conn.close()


def delete_event(event_index):
    """Etkinlik siler - index kullanarak"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Tüm etkinlikleri al
    cursor.execute("SELECT id FROM calendar ORDER BY event_date ASC")
    events = cursor.fetchall()
    
    if 1 <= event_index <= len(events):
        event_id = events[event_index - 1][0]
        cursor.execute("DELETE FROM calendar WHERE id = ?", (event_id,))
        conn.commit()
    
    conn.close()


if __name__ == "__main__":
    initialize_db()
    print("✅ Veritabanı başarıyla oluşturuldu!")