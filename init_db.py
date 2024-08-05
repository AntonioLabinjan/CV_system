import sqlite3

def create_tables():
    conn = sqlite3.connect('employee_attendance.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS employees (
        employee_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        picture_folder TEXT NOT NULL,
        hourly_rate REAL NOT NULL
    );
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        record_id INTEGER PRIMARY KEY,
        employee_id INTEGER,
        entry_time TIMESTAMP,
        exit_time TIMESTAMP,
        FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
    );
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS payments (
        payment_id INTEGER PRIMARY KEY,
        employee_id INTEGER,
        date DATE,
        hours_worked REAL,
        overtime_hours REAL,
        vacation_days INTEGER,
        sick_days INTEGER,
        total_payment REAL,
        FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
    );
    ''')
    conn.commit()
    conn.close()

if __name__ == '__main__':
    create_tables()
