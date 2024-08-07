import os
import cv2
import numpy as np
from flask import Flask, Response, render_template, request, redirect, url_for
from transformers import CLIPProcessor, CLIPModel
import torch
import sqlite3
from datetime import datetime

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
        date DATE,
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
        total_payment REAL,
        FOREIGN KEY (name) REFERENCES employees (name)
    );
    ''')
    print("Payments table - check")
    conn.commit()
    conn.close()

# Initialize Flask app
app = Flask(__name__)

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
def log_attendance(name, action):
    print(f"Attempting to log {action} for {name}")

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    today = datetime.now().date()
    today_str = today.strftime('%Y-%m-%d')

    now = datetime.now()
    now_str = now.strftime('%Y-%m-%d %H:%M:%S')

    print(f"Date: {today_str}, Time: {now_str}")

    if action == 'entry':
        cursor.execute('SELECT * FROM attendance WHERE name = ? AND date = ?', (name, today_str))
        entry = cursor.fetchone()
        print(f"Existing entry: {entry}")

        if entry is None:
            cursor.execute('INSERT INTO attendance (name, entry_time, date) VALUES (?, ?, ?)', (name, now_str, today_str))
            conn.commit()
            print(f"Logged entry for {name} at {now_str}")
        else:
            print(f"Entry for {name} already exists for today.")
    elif action == 'exit':
        cursor.execute('SELECT * FROM attendance WHERE name = ? AND date = ?', (name, today_str))
        entry = cursor.fetchone()
        print(f"Existing entry: {entry}")

        if entry and entry[3] is None:
            entry_time = datetime.strptime(entry[2], '%Y-%m-%d %H:%M:%S')
            exit_time = datetime.strptime(now_str, '%Y-%m-%d %H:%M:%S')
            work_hours = (exit_time - entry_time).total_seconds() / 3600
            cursor.execute('UPDATE attendance SET exit_time = ?, work_hours = ? WHERE record_id = ?', (now_str, work_hours, entry[0]))
            conn.commit()
            print(f"Logged exit for {name} at {now_str}, work hours: {work_hours:.2f}")
        else:
            print(f"No entry found for {name} to log exit.")

    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

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
    cursor.execute('SELECT name, hourly_rate FROM employees')
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




if __name__ == '__main__':
    create_tables()  # Initialize the database tables
    app.run(debug=True)
