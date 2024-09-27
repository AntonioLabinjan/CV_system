Dodat live feed koji prepoznaje zaposlenike i/ili unauthorized osobe

Alert za unauthorized osobe i za mjere zaštite (npr, za pasat kroz neka vrata moramo imat šljem, oćale i zaštitne postole)

Svaki prepoznati zaposlenik će poli sebe dobit basic info

Dodat email api

Dodat deepface koji prati emocije zaposlenika, pa vidi dali dolaze/odlaze sretni/žalosni ili kakovi već i onda to analizirat

Dodat bolju navigaciju (dropdown)

pregled profila za svakega zaposlenika 


### CV_system
### Dokumentacija + smartlock (forši)


### EAPS FOLDER JE NAJBITNIJI

Project Proposal: Smart Employee Attendance and Payment System
Project Overview
The proposed project aims to develop a smart employee attendance and payment system using facial recognition technology. The system will streamline the process of tracking employee attendance, calculating work hours, and computing payments, including adjustments for overtime, vacations, and sick days. This system will enhance security, accuracy, and efficiency in managing employee attendance and payroll.

Key Components
Facial Recognition System:

Hardware: Face scanners at the entrance and exit of the building.
Software: Facial recognition model (using OpenAI's CLIP model or similar) to identify employees.
Smart Lock Integration: Interface with a smart lock to control door access based on facial recognition results.
Employee Database:

Database: SQLite database to store employee information, attendance records, and payment details.
Data Structure:
Employee ID
Name
Picture (folder with subfolders for each employee)
Hourly Rate
Attendance Records (entry and exit times)
Overtime, Vacation, and Sick Day Records
Payment Calculation:

Work Hours Calculation: Subtract exit time from entry time to calculate daily work hours.
Overtime Calculation: Apply a 15% pay increase for overtime hours.
Vacation and Sick Day Calculation: Pay vacation days at the regular rate and sick days at 70% of the regular rate.
Total Payment Calculation: Multiply the number of work hours by the hourly rate, including adjustments for overtime, vacations, and sick days.
User Interface:

Employee Management: Interface for inputting and updating employee information.
Attendance and Payroll Reports: Interface to view and export attendance and payroll records.
Detailed Implementation Plan
Facial Recognition System:

Dataset Preparation:
Create a main folder for employees.
Subfolders for each employee containing their pictures.
Facial Recognition Model:
Implement the CLIP model or similar for face recognition.
Train the model using the prepared dataset.
Smart Lock Integration:
Interface with Arduino or similar hardware.
Implement logic to open/close the door based on recognition results.
Print statements for simulation (e.g., "Person recognized, door opened" / "Person unknown, door closed").

Database Design:
CREATE TABLE employees (
    employee_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    hourly_rate REAL NOT NULL
);

CREATE TABLE attendance (
    record_id INTEGER PRIMARY KEY,
    employee_id INTEGER,
    entry_time TIMESTAMP,
    exit_time TIMESTAMP,
    FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
);

CREATE TABLE payments (
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

Work hours & payment calculation:

def calculate_work_hours(entry_time, exit_time):
    return (exit_time - entry_time).total_seconds() / 3600  # convert seconds to hours

def calculate_payment(hours_worked, hourly_rate, overtime_hours=0, vacation_days=0, sick_days=0):
    base_payment = hours_worked * hourly_rate
    overtime_payment = overtime_hours * hourly_rate * 1.15
    vacation_payment = vacation_days * 8 * hourly_rate  # assuming 8 hours per vacation day
    sick_payment = sick_days * 8 * hourly_rate * 0.70  # assuming 8 hours per sick day
    return base_payment + overtime_payment + vacation_payment + sick_payment

User Interface Development:

Employee Management Interface:
Input form for adding/updating employee information.
List view of employees with options to edit or delete records.
Attendance and Payroll Reports:
Display daily attendance records.
Generate and export payroll reports based on selected date ranges.


Development Timeline
Phase 1: Initial Setup and Database Design (Weeks 1-2)

Set up project environment.
Design and implement the SQLite database schema.

Phase 2: Facial Recognition System (Weeks 3-5)

Prepare the dataset and train the facial recognition model.
Develop the face recognition and smart lock integration logic.

Phase 3: Payment Calculation and Business Logic (Weeks 6-7)

Implement work hours and payment calculation functions.
Integrate these functions with the database.

Phase 4: User Interface Development (Weeks 8-10)

Develop the employee management interface.
Create attendance and payroll report interfaces.

Phase 5: Testing and Deployment (Weeks 11-12)

Conduct thorough testing of the system.
Deploy the system and provide documentation.

Technologies and Tools
Programming Languages: Python (for facial recognition and backend logic)
Database: SQLite
Hardware: Face scanners, Arduino (or similar) for smart lock integration
Frameworks/Libraries: OpenCV (for facial recognition), Flask/Django (for web interface), SQLAlchemy (for database interaction)
Development Environment: Git for version control

