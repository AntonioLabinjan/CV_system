import os
import cv2
import numpy as np
from flask import Flask, Response, render_template, request, redirect, url_for, jsonify
from transformers import CLIPProcessor, CLIPModel
import torch
import sqlite3
from datetime import datetime

from plyer import notification

def send_notification(title, message):
    notification.notify(
        title=title,
        message=message,
        app_name='Employee Attendance System',
        timeout=10  # Notification will disappear after 10 seconds
    )


# Define paths
DATASET_PATH = "C:/Users/Korisnik/Desktop/eaps/dataset"
DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'employee_attendance.db')

# Initialize CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# In-memory storage for known face embeddings and their labels
known_face_encodings = []
known_face_names = []

# Track attendance for the current session
logged_names = set()
message_printed = False

# Function to add a known face
def add_known_face(image_path, name):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} not found.")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = processor(images=image_rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    embedding = outputs.cpu().numpy().flatten()
    known_face_encodings.append(embedding / np.linalg.norm(embedding))  # Normalize the embedding
    known_face_names.append(name)
    print(f"Added face for {name} from {image_path}")

# Load all known faces from the 'dataset' directory
def load_known_faces():
    for employee_name in os.listdir(DATASET_PATH):
        employee_dir = os.path.join(DATASET_PATH, employee_name)
        if os.path.isdir(employee_dir):
            for filename in os.listdir(employee_dir):
                image_path = os.path.join(employee_dir, filename)
                try:
                    add_known_face(image_path, employee_name)
                except ValueError as e:
                    print(e)

    # Debugging output
    print(f"Loaded {len(known_face_encodings)} known face encodings")
    print(f"Known face names: {known_face_names}")

# Load known faces at startup
load_known_faces()

def create_tables():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS employees (
        employee_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        picture_folder TEXT NOT NULL,
        hourly_rate REAL NOT NULL DEFAULT 0.0
    );
    ''')
    
    print("Employee table - check")
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        record_id INTEGER PRIMARY KEY,
        name TEXT,
        entry_time TIMESTAMP,
        exit_time TIMESTAMP,
        work_hours REAL,
        overtime_hours REAL DEFAULT 0.0,
        date DATE,
        late_entrance_penalty INTEGER DEFAULT 0,
        FOREIGN KEY (name) REFERENCES employees (name)
    );
    ''')

    print("Attendance table - check")
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS payments (
        payment_id INTEGER PRIMARY KEY,
        name TEXT,
        hours_worked REAL,
        overtime_hours REAL,
        vacation_days INTEGER,
        sick_days INTEGER,
        late_entrance_penalty INTEGER DEFAULT 0,
        total_payment REAL,
        FOREIGN KEY (name) REFERENCES employees (name)
    );
    ''')

    # New table for sick days and vacation days
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS absences (
        absence_id INTEGER PRIMARY KEY,
        name TEXT,
        date DATE,
        type TEXT,
        FOREIGN KEY (name) REFERENCES employees (name)
    );
    ''')

    print("Absences table - check")

    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS chat (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, message TEXT, time TEXT)''')
    
    print("Chat table - check")

    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS worktime_start (id INTEGER PRIMARY KEY AUTOINCREMENT, start_time TIMESTAMP)''')
    print("Worktime start table - check")

    conn.commit()
    conn.close()



# Initialize Flask app
app = Flask(__name__)

@app.route('/set_starttime', methods=['GET', 'POST'])
def set_starttime():
    if request.method == 'POST':
        time = request.form['start_time']
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO worktime_start (start_time) VALUES (?)', (time,))
        conn.commit()
        conn.close()
        return redirect(url_for('index'))
    return render_template('start_time_set.html')

from flask import flash, redirect, url_for

@app.route('/check_starttime', methods=['GET', 'POST'])
def check_starttime():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Fetch the current start time from the database
    cursor.execute('SELECT start_time FROM worktime_start ORDER BY id DESC LIMIT 1')
    result = cursor.fetchone()
    
    conn.close()
    
    # If a start time exists, print it in the terminal
    if result:
        start_time = result[0]
        print(f'Current start time is: {start_time}')
    else:
        print('No start time has been set.')
    
    return redirect(url_for('index'))





# Function to recognize face in the frame
def recognize_face(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=image_rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    embedding = outputs.cpu().numpy().flatten()
    embedding = embedding / np.linalg.norm(embedding)  # Normalize the embedding

    # Compare with known faces
    distances = np.linalg.norm(known_face_encodings - embedding, axis=1)
    min_distance = np.min(distances)
    if min_distance < 0.6:  # Threshold for recognizing a face
        index = np.argmin(distances)
        name = known_face_names[index]
        return name
    return None


# Route to stream video
@app.route('/video_feed', methods=['GET', 'POST'])
def video_feed():
    global button_clicked
    if request.method == 'POST':
        action = request.args.get('action')
        button_clicked = action
        return '', 204
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    global button_clicked, message_printed
    button_clicked = None
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert the frame to RGB (OpenCV uses BGR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Load a pre-trained face detector from OpenCV
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the face region
            face_region = frame[y:y+h, x:x+w]
            
            # Recognize the face
            name = recognize_face(face_region)
            
            # Draw the bounding box
            color = (0, 255, 0) if name else (0, 0, 255)  # Green for known, red for unknown
            label = name if name else "unknown"
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Log attendance if recognized and button clicked
            if name and button_clicked:
                if button_clicked == 'entry' and name not in logged_names:
                    logged_names.add(name)
                    log_attendance(name, 'entry')
                elif button_clicked == 'exit' and name in logged_names:
                    log_attendance(name, 'exit')
                    logged_names.remove(name)

                button_clicked = None  # Reset button click state

            if not name and not message_printed:
                print("Person unknown, door remains locked")
                message_printed = True

        # Convert the frame back to BGR (for display with OpenCV)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Function to log attendance
import sqlite3
import time
from datetime import datetime


def log_attendance(name, action):
    print(f"Attempting to log {action} for {name}")

    conn = None
    cursor = None

    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        today = datetime.now().date()
        today_str = today.strftime('%Y-%m-%d')

        now = datetime.now()
        now_str = now.strftime('%Y-%m-%d %H:%M:%S')

        print(f"Date: {today_str}, Time: {now_str}")

        late_penalty = 0  # Initialize late penalty

        if action == 'entry':
            cursor.execute('SELECT start_time FROM worktime_start ORDER BY id DESC LIMIT 1')
            start_time_record = cursor.fetchone()
            
            if start_time_record:
                start_time_str = start_time_record[0]
                try:
                    start_time = datetime.strptime(f'{today_str} {start_time_str}', '%Y-%m-%d %H:%M')
                except ValueError:
                    start_time = datetime.strptime(f'{today_str} {start_time_str}', '%Y-%m-%d %H:%M:%S')

                entry_time = now
                
                # Check if the entry time is later than the start time
                if entry_time > start_time:
                    print("LATE ENTRANCE")
                    late_penalty = 10  # Apply late entrance penalty

            cursor.execute('SELECT * FROM attendance WHERE name = ? AND date = ?', (name, today_str))
            entry = cursor.fetchone()
            print(f"Existing entry: {entry}")

            if entry is None:
                retries = 3
                while retries > 0:
                    try:
                        cursor.execute('INSERT INTO attendance (name, entry_time, date, late_entrance_penalty) VALUES (?, ?, ?, ?)', 
                                       (name, now_str, today_str, late_penalty))
                        conn.commit()
                        print(f"Logged entry for {name} at {now_str} with late penalty {late_penalty}")
                        send_notification(f"Entry Logged", f"{name} entered at {now_str}")
                        speak_message(f"Attendance for {name} successfully logged at {now_str} on {today_str}")
                        break
                    except sqlite3.OperationalError as e:
                        if "locked" in str(e):
                            print("Database is locked, retrying...")
                            time.sleep(1)
                            retries -= 1
                        else:
                            raise e
                else:
                    print(f"Failed to log entry for {name} due to database lock.")
            else:
                print(f"Entry for {name} already exists for today.")
        
        elif action == 'exit':
            cursor.execute('SELECT * FROM attendance WHERE name = ? AND date = ?', (name, today_str))
            entry = cursor.fetchone()
            print(f"Existing entry: {entry}")

            if entry and entry[3] is None:
                entry_time = datetime.strptime(entry[2], '%Y-%m-%d %H:%M:%S')
                exit_time = datetime.strptime(now_str, '%Y-%m-%d %H:%M:%S')
                total_work_hours = (exit_time - entry_time).total_seconds() / 3600

                regular_hours = min(total_work_hours, 8)
                overtime_hours = max(total_work_hours - 8, 0)

                cursor.execute('UPDATE attendance SET exit_time = ?, work_hours = ?, overtime_hours = ? WHERE record_id = ?', 
                               (now_str, regular_hours, overtime_hours, entry[0]))
                conn.commit()
                print(f"Logged exit for {name} at {now_str}, work hours: {total_work_hours:.2f}, overtime: {overtime_hours:.2f}")
                send_notification(f"Exit Logged", f"{name} exited at {now_str}")
                speak_message(f"Attendance for {name} successfully logged at {now_str} on {today_str}")
            else:
                print(f"No entry found for {name} to log exit.")

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()








@app.route('/')
def index():
    api_key = "fe2e5f9339b2434db60124446241408"
    location = "London" # Ili nešto drugo
    weather_condition = get_weather_forecast(api_key, location)
    
    if predict_absence_due_to_weather(weather_condition):
        message = "Bad weather predicted, late entries due to traffic problems are possible."
    else:
        message = "No significant weather issues expected. Employees should come on time"
    
    return render_template('index.html', weather_condition=weather_condition, message=message)

@app.route('/video')
def video():
    return render_template('video_feed.html')

@app.route('/add_employee', methods=['GET', 'POST'])
def add_employee():
    if request.method == 'POST':
        name = request.form['name']
        hourly_rate = request.form['hourly_rate']
        images = request.files.getlist('images')

        # Create a new subfolder in the dataset directory
        employee_dir = os.path.join(DATASET_PATH, name)
        os.makedirs(employee_dir, exist_ok=True)

        for image in images:
            # Save each uploaded image in the new employee's subfolder
            image_path = os.path.join(employee_dir, image.filename)
            image.save(image_path)
            add_known_face(image_path, name)
        
        # Add employee details to the database
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO employees (name, picture_folder, hourly_rate) VALUES (?, ?, ?)', (name, employee_dir, hourly_rate))
        conn.commit()
        conn.close()
        
        return render_template('add_employee_success.html', name=name)

    return render_template('add_employee.html')

def check_employee(name):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT name FROM employees WHERE name = ?', (name,))
    employee = cursor.fetchone()
    conn.close()
    return employee

@app.route('/check_employee/<name>')
def check_employee_route(name):
    employee = check_employee(name)
    if employee:
        return f"Employee found: {employee[0]}"
    else:
        return "Employee not found"

@app.route('/employees', methods=['GET'])
def list_employees():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT name, hourly_rate, employee_id FROM employees')
    employees = cursor.fetchall()
    conn.close()
    return render_template('list_employees.html', employees=employees)


@app.route('/attendance_report', methods=['GET'])
def attendance_report():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    today = datetime.now().date()
    cursor.execute('''
    SELECT name, entry_time, exit_time, work_hours
    FROM attendance
    WHERE date = ?
    ''', (today,))
    records = cursor.fetchall()
    conn.close()
    return render_template('attendance_report.html', records=records, today=today)

@app.route('/payment_report', methods=['GET'])
def payment_report():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    today = datetime.now().date()
    first_day_of_month = today.replace(day=1)

    # Calculate total hours worked, overtime, and late penalties
    cursor.execute('''
    SELECT e.name, e.hourly_rate, SUM(a.work_hours), SUM(a.overtime_hours), SUM(a.late_entrance_penalty)
    FROM employees e
    LEFT JOIN attendance a ON e.name = a.name
    WHERE a.date BETWEEN ? AND ?
    GROUP BY e.name
    ''', (first_day_of_month, today))
    
    work_records = cursor.fetchall()

    # Calculate sick days and vacation days
    cursor.execute('''
    SELECT e.name, e.hourly_rate, 
           COALESCE(SUM(CASE WHEN type = 'sick' THEN 1 ELSE 0 END), 0) as sick_days, 
           COALESCE(SUM(CASE WHEN type = 'vacation' THEN 1 ELSE 0 END), 0) as vacation_days
    FROM employees e
    LEFT JOIN absences a ON e.name = a.name
    WHERE a.date BETWEEN ? AND ?
    GROUP BY e.name, e.hourly_rate
    ''', (first_day_of_month, today))
    
    absence_records = cursor.fetchall()

    # Merge work and absence records
    employee_data = {record[0]: {
        "name": record[0],
        "hourly_rate": record[1],
        "total_hours": 0,
        "overtime_hours": 0,
        "late_entrance_penalty": 0,
        "sick_days": record[2],
        "vacation_days": record[3],
        "total_payment": 0
    } for record in absence_records}

    for work_record in work_records:
        name = work_record[0]
        hourly_rate = work_record[1]
        total_hours = work_record[2] if work_record[2] else 0
        overtime_hours = work_record[3] if work_record[3] else 0
        late_entrance_penalty = work_record[4] if work_record[4] else 0

        if name not in employee_data:
            employee_data[name] = {
                "name": name,
                "hourly_rate": hourly_rate,
                "total_hours": total_hours,
                "overtime_hours": overtime_hours,
                "late_entrance_penalty": late_entrance_penalty,
                "sick_days": 0,
                "vacation_days": 0,
                "total_payment": 0
            }
        else:
            employee_data[name]["total_hours"] = total_hours
            employee_data[name]["overtime_hours"] = overtime_hours
            employee_data[name]["late_entrance_penalty"] = late_entrance_penalty

    payments = []
    for data in employee_data.values():
        name = data["name"]
        hourly_rate = data["hourly_rate"]
        total_hours = data["total_hours"]
        overtime_hours = data["overtime_hours"]
        sick_days = data["sick_days"]
        vacation_days = data["vacation_days"]
        late_penalty = data["late_entrance_penalty"]
        
        # Payment for work hours (excluding overtime)
        work_payment = hourly_rate * total_hours

        # Payment for overtime hours (120% of hourly rate)
        overtime_payment = overtime_hours * hourly_rate * 1.2

        # Payment for vacation days
        vacation_payment = 8 * hourly_rate * vacation_days

        # Payment for sick days (70% of 8-hour workday)
        sick_payment = 0.7 * 8 * hourly_rate * sick_days

        # Subtract late penalties
        penalty_deduction = late_penalty

        # Calculate the total payment
        total_payment = work_payment + overtime_payment + vacation_payment + sick_payment - penalty_deduction
        data["total_payment"] = total_payment

        payments.append(data)
    
    conn.close()
    return render_template('payment_report.html', payments=payments, today=today)







import csv
from flask import send_file, Response
from io import StringIO
import sqlite3
from datetime import datetime

@app.route('/export_payment_report', methods=['GET'])
def export_payment_report():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    today = datetime.now().date()
    first_day_of_month = today.replace(day=1)

    cursor.execute('''
    SELECT e.name, e.hourly_rate, SUM(a.work_hours)
    FROM employees e
    LEFT JOIN attendance a ON e.name = a.name
    WHERE a.date BETWEEN ? AND ?
    GROUP BY e.name
    ''', (first_day_of_month, today))
    
    records = cursor.fetchall()
    
    payments = [
        {
            "name": record[0],
            "hourly_rate": record[1],
            "total_hours": record[2] if record[2] else 0,
            "total_payment": record[1] * record[2] if record[2] else 0
        }
        for record in records
    ]
    
    conn.close()

    # Create a string buffer to write CSV data
    output = StringIO()
    writer = csv.writer(output)

    # Write CSV headers
    writer.writerow(['Employee Name', 'Hourly Rate', 'Total Hours Worked', 'Total Payment'])
    
    # Write CSV data
    for payment in payments:
        writer.writerow([payment["name"], payment["hourly_rate"], payment["total_hours"], payment["total_payment"]])
    
    # Seek to the beginning of the stream
    output.seek(0)

    return Response(output, mimetype="text/csv", headers={"Content-Disposition": "attachment;filename=payment_report.csv"})


@app.route('/edit_employee/<int:employee_id>', methods=['GET', 'POST'])
def edit_employee(employee_id):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    if request.method == 'POST':
        name = request.form['name']
        hourly_rate = request.form['hourly_rate']
        
        # Update employee details in the database
        cursor.execute('UPDATE employees SET name = ?, hourly_rate = ? WHERE employee_id = ?', (name, hourly_rate, employee_id))
        conn.commit()
        conn.close()
        
        # Reload known faces to reflect name changes
        load_known_faces()
        
        return redirect(url_for('list_employees'))
    
    # Fetch the current employee details
    cursor.execute('SELECT name, hourly_rate FROM employees WHERE employee_id = ?', (employee_id,))
    employee = cursor.fetchone()
    conn.close()
    
    return render_template('edit_employee.html', employee=employee, employee_id=employee_id)


@app.route('/delete_employee/<int:employee_id>')
def delete_employee(employee_id):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Delete the employee record from the database
    cursor.execute('DELETE FROM employees WHERE employee_id = ?', (employee_id,))
    conn.commit()
    conn.close()
    
    # Reload known faces to remove deleted employees
    load_known_faces()
    
    return redirect(url_for('list_employees'))


@app.route('/log_absence', methods=['GET', 'POST'])
def log_absence():
    if request.method == 'POST':
        name = request.form['name']
        date = request.form['date']
        absence_type = request.form['absence_type']
        
        # Insert the absence into the database
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO absences (name, date, type) VALUES (?, ?, ?)', (name, date, absence_type))
        conn.commit()
        conn.close()

        # Send notification for the logged absence
        send_notification(f"Absence Logged", f"{name} logged a {absence_type} on {date}")
        
        return render_template('log_absence_success.html', name=name, date=date, absence_type=absence_type)
    
    return render_template('log_absence.html')


import matplotlib.pyplot as plt
from io import BytesIO
from flask import send_file

# Visualization route for attendance overview
@app.route('/attendance_overview')
def attendance_overview():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT name, COUNT(*) as days_present
    FROM attendance
    WHERE date >= date('now', 'start of month')
    GROUP BY name
    ''')
    
    records = cursor.fetchall()
    conn.close()
    
    names = [record[0] for record in records]
    days_present = [record[1] for record in records]

    fig, ax = plt.subplots()
    ax.bar(names, days_present, color='blue')
    ax.set_xlabel('Employee')
    ax.set_ylabel('Days Present')
    ax.set_title('Attendance Overview - Current Month')
    plt.xticks(rotation=45, ha='right')
    
    # Save the plot to a BytesIO object and send it as a response
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')

# Visualization route for work hours distribution
@app.route('/work_hours_distribution')
def work_hours_distribution():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT e.name, SUM(a.work_hours) as total_hours, SUM(a.overtime_hours) as overtime_hours
    FROM employees e
    LEFT JOIN attendance a ON e.name = a.name
    WHERE a.date >= date('now', 'start of month')
    GROUP BY e.name
    ''')
    
    records = cursor.fetchall()
    conn.close()
    
    names = [record[0] for record in records]
    total_hours = [record[1] if record[1] else 0 for record in records]
    overtime_hours = [record[2] if record[2] else 0 for record in records]
    
    fig, ax = plt.subplots()
    width = 0.35
    x = np.arange(len(names))
    ax.bar(x - width/2, total_hours, width, label='Total Hours')
    ax.bar(x + width/2, overtime_hours, width, label='Overtime Hours')
    
    ax.set_xlabel('Employee')
    ax.set_ylabel('Hours Worked')
    ax.set_title('Work Hours Distribution - Current Month')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()

    # Save the plot to a BytesIO object and send it as a response
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')

# Visualization route for payment breakdown
@app.route('/payment_breakdown')
def payment_breakdown():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    today = datetime.now().date()
    first_day_of_month = today.replace(day=1)

    cursor.execute('''
    SELECT e.name, e.hourly_rate, SUM(a.work_hours), SUM(a.overtime_hours),
           COALESCE(SUM(CASE WHEN abs.type = 'sick' THEN 1 ELSE 0 END), 0) as sick_days,
           COALESCE(SUM(CASE WHEN abs.type = 'vacation' THEN 1 ELSE 0 END), 0) as vacation_days
    FROM employees e
    LEFT JOIN attendance a ON e.name = a.name
    LEFT JOIN absences abs ON e.name = abs.name
    WHERE a.date BETWEEN ? AND ?
    GROUP BY e.name
    ''', (first_day_of_month, today))

    records = cursor.fetchall()
    conn.close()
    
    names = [record[0] for record in records]
    total_payments = []
    
    for record in records:
        hourly_rate = record[1]
        total_hours = record[2] if record[2] else 0
        overtime_hours = record[3] if record[3] else 0
        sick_days = record[4] if record[4] else 0
        vacation_days = record[5] if record[5] else 0
        
        # Payment calculations
        work_payment = hourly_rate * total_hours
        overtime_payment = overtime_hours * hourly_rate * 1.2
        vacation_payment = 8 * hourly_rate * vacation_days
        sick_payment = 0.7 * 8 * hourly_rate * sick_days
        
        total_payment = work_payment + overtime_payment + vacation_payment + sick_payment
        total_payments.append(total_payment)
    
    fig, ax = plt.subplots()
    ax.bar(names, total_payments, color='green')
    ax.set_xlabel('Employee')
    ax.set_ylabel('Total Payment')
    ax.set_title('Payment Breakdown - Current Month')
    plt.xticks(rotation=45, ha='right')
    
    # Save the plot to a BytesIO object and send it as a response
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')

import pyttsx3

def speak_message(message):
    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()

chat_history = []

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        message = request.form.get('message')

        # Here you should recognize the person using a current frame
        cap = cv2.VideoCapture(0)
        success, frame = cap.read()
        if success:
            name = recognize_face(frame)
            person_name = name if name else "Unknown"
        else:
            person_name = "Unknown"

        cap.release()

        # Format the chat message
        chat_message = f"{person_name}: {message}"

        # Append the message to the chat history
        chat_history.append(chat_message)

        return jsonify({"status": "success", "chat_message": chat_message})

    return render_template('chat.html', chat_history=chat_history)


import requests


def get_weather_forecast(api_key, location="your_city"):
    url = f"http://api.weatherapi.com/v1/forecast.json?key={api_key}&q={location}&days=1"
    response = requests.get(url)
    data = response.json()
    return data["forecast"]["forecastday"][0]["day"]["condition"]["text"]

def predict_absence_due_to_weather(weather_condition):
    bad_weather_keywords = ["rain", "storm", "snow", "fog", "hurricane"]
    for keyword in bad_weather_keywords:
        if keyword in weather_condition.lower():
            return True
    return False

@app.route('/predict_absence', methods=['GET'])
def predict_absence():
    api_key = "fe2e5f9339b2434db60124446241408"
    location = "London"
    weather_condition = get_weather_forecast(api_key, location)
    
    if predict_absence_due_to_weather(weather_condition):
        message = "Bad weather predicted, late entries due to traffic problems are possible."
    else:
        message = "No significant weather issues expected."
    
    # Return both the weather condition and the prediction message
    return jsonify({
        "weather_condition": weather_condition,
        "message": message
    })



if __name__ == '__main__':
    create_tables()  # Initialize the database tables
    app.run(debug=True)
