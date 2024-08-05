import os
import cv2
import numpy as np
from flask import Flask, Response, render_template, request, redirect, url_for
from transformers import CLIPProcessor, CLIPModel
import torch
import sqlite3
from datetime import datetime

# Initialize CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# In-memory storage for known face embeddings and their labels
known_face_encodings = []
known_face_names = []

# Track attendance for the current session
logged_names = set()

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
    for employee_name in os.listdir('dataset'):
        employee_dir = os.path.join('dataset', employee_name)
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
    conn = sqlite3.connect('employee_attendance.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS employees (
        employee_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        picture_folder TEXT NOT NULL,
        hourly_rate REAL NOT NULL DEFAULT 0.0
    );
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        record_id INTEGER PRIMARY KEY,
        employee_id INTEGER,
        entry_time TIMESTAMP,
        exit_time TIMESTAMP,
        date DATE,
        FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
    );
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS payments (
        payment_id INTEGER PRIMARY KEY,
        employee_id INTEGER,
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
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
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

            # Log attendance if recognized and not already logged
            if name and name not in logged_names:
                logged_names.add(name)
                log_attendance(name)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield the frame to the client
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


# Function to log attendance to the database
def log_attendance(name):
    conn = sqlite3.connect('employee_attendance.db')
    cursor = conn.cursor()

    # Check if entry already exists for today
    today = datetime.now().date()
    cursor.execute('SELECT * FROM attendance WHERE employee_id = (SELECT employee_id FROM employees WHERE name = ?) AND date = ?', (name, today))
    entry = cursor.fetchone()

    if entry is None:
        cursor.execute('INSERT INTO attendance (employee_id, entry_time, date) VALUES ((SELECT employee_id FROM employees WHERE name = ?), ?, ?)',
                       (name, datetime.now(), today))
        conn.commit()
        print(f"Logged entry for {name} at {datetime.now()}")

    conn.close()

# Function to initialize the database


# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return render_template('video_feed.html')

if __name__ == '__main__':
    create_tables()  # Initialize the database tables
    app.run(debug=True)
