import os
import cv2
import numpy as np
from flask import Flask, Response, render_template, request, redirect, url_for
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
            send_notification(f"Entry Logged", f"{name} entered at {now_str}")
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

    # Calculate total hours worked and overtime
    cursor.execute('''
    SELECT e.name, e.hourly_rate, SUM(a.work_hours), SUM(a.overtime_hours)
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
        "sick_days": record[2],
        "vacation_days": record[3],
        "total_payment": 0
    } for record in absence_records}

    for work_record in work_records:
        name = work_record[0]
        hourly_rate = work_record[1]
        total_hours = work_record[2] if work_record[2] else 0
        overtime_hours = work_record[3] if work_record[3] else 0

        if name not in employee_data:
            employee_data[name] = {
                "name": name,
                "hourly_rate": hourly_rate,
                "total_hours": total_hours,
                "overtime_hours": overtime_hours,
                "sick_days": 0,
                "vacation_days": 0,
                "total_payment": 0
            }
        else:
            employee_data[name]["total_hours"] = total_hours
            employee_data[name]["overtime_hours"] = overtime_hours

    payments = []
    for data in employee_data.values():
        name = data["name"]
        hourly_rate = data["hourly_rate"]
        total_hours = data["total_hours"]
        overtime_hours = data["overtime_hours"]
        sick_days = data["sick_days"]
        vacation_days = data["vacation_days"]
        
        # Payment for work hours (excluding overtime)
        work_payment = hourly_rate * total_hours

        # Payment for overtime hours (120% of hourly rate)
        overtime_payment = overtime_hours * hourly_rate * 1.2

        # Payment for vacation days
        vacation_payment = 8 * hourly_rate * vacation_days

        # Payment for sick days (70% of 8-hour workday)
        sick_payment = 0.7 * 8 * hourly_rate * sick_days

        # Calculate the total payment
        total_payment = work_payment + overtime_payment + vacation_payment + sick_payment
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


if __name__ == '__main__':
    create_tables()  # Initialize the database tables
    app.run(debug=True)
