import streamlit as st
import pandas as pd
import cv2
import dlib
import numpy as np
import pygame
import sqlite3
import hashlib
import imutils
import time
import face_recognition
import pickle
from PIL import Image
import base64
from streamlit_option_menu import option_menu

# Hide Streamlit's default menu and footer
hide_st_style = """
               <style>
               #mainMenu {visibility:hidden;}
               footer {visibility:hidden;}
               </style>
               """
st.markdown(hide_st_style, unsafe_allow_html=True)

def add_sidebar_image(image_url):
    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] {{
            background: url({image_url});
            background-size: cover; 
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to encode an image file as a base64 string
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Set background image function
def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    [data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpeg;base64,{bin_str}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set the background image globally
set_background('C:/Users/nsgma/OneDrive/Desktop/D3/car.jpg')

# Initialize Pygame mixer for sound
pygame.mixer.init()

def create_db():
    try:
        conn = sqlite3.connect('users.db')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password TEXT,
                name TEXT,
                age INTEGER,
                email TEXT,
                mobile TEXT,
                alt_mobile TEXT,
                vehicle_type TEXT,
                face_encoding BLOB
            )
        ''')
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        st.error(f"❌ Database error: {e}")

# Call the function to create the database and table
create_db()
# Add this code to manually insert test data into the ride_history table
def add_test_ride_history():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO ride_history (username, start_time, end_time, date, duration)
        VALUES (?, ?, ?, ?, ?)
    ''', ("testuser", "2023-10-01 10:00:00", "2023-10-01 11:00:00", "2023-10-01", 60))
    conn.commit()
    conn.close()
    st.success("Test ride data added successfully!")

# Call the function to add test data
# Hash Passwords for Security
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Capture and Encode Face
def capture_face_encoding():
    video_capture = cv2.VideoCapture(0)
    st.info("📸 Capturing face... Please look at the camera.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            st.error("❌ Camera access failed. Please check your webcam.")
            video_capture.release()
            return None
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if face_encodings:
            st.success("✅ Face captured successfully!")
            video_capture.release()
            return pickle.dumps(face_encodings[0])  # Convert to binary before storing

    video_capture.release()
    return None
def background(png_file):
            bin_str = get_base64_of_bin_file(png_file)
            page_bg_img = f'''
            <style>
            [data-testid="stAppViewContainer"] {{
                background-image: url("data:image/jpeg;base64,{bin_str}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            '''
            st.markdown(page_bg_img, unsafe_allow_html=True)
# Create New User with Face Encoding
def create_user(username, password, name, age, email, mobile, alt_mobile, vehicle_type):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    # Capture Face Encoding
    face_encoding = capture_face_encoding()
    
    if face_encoding is None:
        st.error("❌ Face not detected. Please try again.")
        return  # Stop if face is not detected

    try:
        cursor.execute('''
            INSERT INTO users (username, password, name, age, email, mobile, alt_mobile, vehicle_type, face_encoding) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (username, hash_password(password), name, age, email, mobile, alt_mobile, vehicle_type, face_encoding))
        conn.commit()
        st.success("✅ Account created successfully! You can now log in with your face or password.")
    except sqlite3.IntegrityError:
        st.error("❌ Username already exists. Please choose another.")
    except sqlite3.Error as e:
        st.error(f"❌ Database error: {e}")
    finally:
        conn.close()

# Authenticate User via Password
def authenticate(username, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hash_password(password)))
    user = cursor.fetchone()
    conn.close()
    return user is not None

# Authenticate User via Face Recognition
def authenticate_face():
    st.warning("📸 Please look at the camera to log in...")
    cap = cv2.VideoCapture(0)
    
    # Load user data first to avoid repeated DB calls
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT username, face_encoding FROM users")
        users = cursor.fetchall()
    except sqlite3.OperationalError:
        st.error("❌ Database error. Please re-register.")
        return None
    finally:
        conn.close()

    # Preprocess known encodings
    known_encodings = []
    known_usernames = []
    for user in users:
        if user[1]:
            known_usernames.append(user[0])
            known_encodings.append(pickle.loads(user[1]))
    
    # Convert to numpy array for faster calculations
    known_encodings = np.array(known_encodings)
    
    # Performance parameters
    TOLERANCE = 0.5
    FRAME_SKIP = 5  # Process every 5th frame
    TIMEOUT = 30    # Max 30 seconds
    frame_count = 0
    start_time = time.time()
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        while (time.time() - start_time) < TIMEOUT:
            ret, frame = cap.read()
            if not ret:
                status_text.error("❌ Camera error")
                break

            # Skip frames for faster processing
            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                continue

            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Find faces with faster HOG model
            face_locations = face_recognition.face_locations(
                rgb_small_frame, 
                model="hog"  # Faster than CNN
            )

            if face_locations:
                # Get encodings with reduced accuracy for speed
                face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, 
                    face_locations,
                    num_jitters=0  # Reduced from default 1
                )

                for encoding in face_encodings:
                    if known_encodings.size == 0:
                        continue

                    # Vectorized comparison for speed
                    distances = face_recognition.face_distance(known_encodings, encoding)
                    best_match_index = np.argmin(distances)
                    
                    if distances[best_match_index] < TOLERANCE:
                        username = known_usernames[best_match_index]
                        progress_bar.progress(100)
                        status_text.success(f"✅ Welcome, {username}!")
                        return username

            # Update progress
            elapsed = time.time() - start_time
            progress = min(int((elapsed/TIMEOUT)*100), 100)
            progress_bar.progress(progress)
            status_text.info(f"🔍 Scanning... ({progress}%)")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if (time.time() - start_time) >= TIMEOUT:
            status_text.error("⏰ Authentication timed out")
        progress_bar.empty()

    return None

# Initialize Session
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Display Logo
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_base64 = get_base64_image("logo-removebg-preview.png")

st.markdown(
    f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <img src="data:image/png;base64,{img_base64}" style="max-width: 500px; width: 100%; height: auto;">
    </div>
    """,
    unsafe_allow_html=True
)

# Show Login/Signup/Face Login options only if the user is not logged in
if not st.session_state.get('logged_in', False):
    menu = ["Login", "Signup", "Face Login"]  # Add "Face Login" to the dropdown
    choice = st.selectbox("Select the option", menu)

    if choice == "Signup":
        st.subheader("Create a New Account")
        username = st.text_input("Username")
        name = st.text_input("Full Name")
        age = st.number_input("Age", min_value=1, max_value=100, step=1)
        email = st.text_input("Email")
        mobile = st.text_input("Mobile Number")
        alt_mobile = st.text_input("Alternative Mobile Number")
        vehicle_type = st.text_input("Type of Vehicle")
        password = st.text_input("Password", type='password')
        confirm_password = st.text_input("Confirm Password", type='password')

        if st.button("Signup"):
            if not username or not name or age <= 0 or not email or not mobile or not alt_mobile or not password or not confirm_password or not vehicle_type:
                st.error("❌ Please fill in all fields.")
            elif password != confirm_password:
                st.error("❌ Passwords do not match. Please try again.")
            else:
                create_user(username, password, name, age, email, mobile, alt_mobile, vehicle_type)

    elif choice == "Login":
        st.subheader("🔐 Login to Your Account")
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')

        if st.button("Login"):
            if authenticate(username, password):
                st.success("✅ Login successful!")
                st.session_state.logged_in = True
                st.session_state.username = username  # Ensure this line is present
                st.experimental_rerun()
            else:
                st.error("❌ Invalid credentials. Please try again.")

    elif choice == "Face Login":  # New option for Face Login
        st.subheader("🔍 Face Recognition Login")
        if st.button("Login with Face"):
            username = authenticate_face()
            if username:
                st.success(f"✅ Welcome, {username}! Login successful.")
                st.session_state.logged_in = True
                st.session_state.username = username
                st.experimental_rerun()
            else:
                st.error("❌ Face not recognized. Try again.")
# Show Sidebar options only if the user is logged in
if st.session_state.get('logged_in', False):
    add_sidebar_image("https://wallpaperaccess.com/full/2454628.png")
    with st.sidebar:
        st.image("logo-removebg-preview.png", width=200)
        selected = option_menu(
            menu_title=None,
            options=["Home", "Profile", "About", "Contact us"],
            icons=["house-door-fill", "person-circle", "info-circle", "telephone-fill"],
        )
        st.write("---")
        st.write("Guide Name: Mr. Abdul Aziz Md")

    # Handle sidebar selection
    if selected == "Home":
        background(r'C:\Users\nsgma\OneDrive\Desktop\D3\dark.jpg')
    # Add the custom heading for "Driver Drowsiness Detection"
        st.markdown("""
        <h1 style='text-align: center; color: #00698f; font-size: 50px; font-family: "Arial", sans-serif; text-shadow: 2px 2px 5px rgba(0,0,0,0.5);'>
            Driver Drowsiness Detection
        </h1>
        """, unsafe_allow_html=True)
        description = """
            <p style='font-size: 15px; color: white; text-align: justify; font-family: Arial, sans-serif;'>
            The Driver Drowsiness Detection System is an innovative safety technology designed to prevent accidents caused by driver fatigue, a major cause of road accidents globally. 
            This system continuously monitors the driver's face, particularly focusing on the eyes, and detects early signs of drowsiness. 
            When the system detects symptoms of fatigue, such as slow blinking or eye closure, it triggers an alert to wake the driver, potentially preventing dangerous situations.
            </p>
            """

        # Display the description
        st.markdown(description, unsafe_allow_html=True)
        st.markdown("""
            <style>
            div.stButton > button:first-child {
                background-color: blue;
                color: white;
                width: 50%;
                margin: 0 auto;
                display: block;
            }
            </style>
            """, unsafe_allow_html=True)

        # Start ride detection button
        if st.button("Start Ride"):
            ride_start_time = time.strftime("%Y-%m-%d %H:%M:%S")  # Record start time
            ride_date = time.strftime("%Y-%m-%d")  
            # Load the alarm sound
            alarm_sound = pygame.mixer.Sound('alarm-car-or-home-62554.mp3')

            # Function to calculate the Eye Aspect Ratio (EAR)
            def eye_aspect_ratio(eye):
                A = np.linalg.norm(eye[1] - eye[5])
                B = np.linalg.norm(eye[2] - eye[4])
                C = np.linalg.norm(eye[0] - eye[3])
                ear = (A + B) / (2.0 * C)
                return ear

            # Load face and eye detection models
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            detector = dlib.get_frontal_face_detector()

            # Update the path to the shape predictor model
            predictor_path = r'C:\Users\nsgma\OneDrive\Desktop\D3\shape_predictor_68_face_landmarks.dat'  # Update this path
            try:
                predictor = dlib.shape_predictor(predictor_path)
            except Exception as e:
                st.error(f"Error loading shape predictor: {e}. Please ensure the file exists at the specified path.")
                st.stop()  # Stop execution if the predictor cannot be loaded

            # EAR threshold for detecting closed eyes
            EYE_AR_THRESH = 0.2

            # Start video capture
            cap = cv2.VideoCapture(0)

            alarm_played = False
            alarm_start_time = 0

            video_frame = st.empty()  # Streamlit component to display video

            # Stop detection button
            stop_detection = st.button("Stop Detection")

            while True:
                ret, frame = cap.read()
                frame = imutils.resize(frame, width=640)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                if len(faces) == 0:
                    # No face detected
                    cv2.putText(frame, "No Driver Detected", (frame.shape[1] // 2 - 100, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                else:
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        dlib_faces = detector(gray)

                        for face in dlib_faces:
                            landmarks = predictor(gray, face)
                            landmarks_array = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])

                            left_eye = landmarks_array[36:42]
                            right_eye = landmarks_array[42:48]
                            left_ear = eye_aspect_ratio(left_eye)
                            right_ear = eye_aspect_ratio(right_eye)
                            ear = (left_ear + right_ear) / 2.0

                            # Detect drowsiness based on EAR
                            if ear < EYE_AR_THRESH:
                                cv2.putText(frame, "DROWSY", (frame.shape[1] // 2 - 50, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                                if not alarm_played:
                                    alarm_sound.play()
                                    alarm_played = True
                                    alarm_start_time = time.time()
                            else:
                                cv2.putText(frame, "AWAKE", (frame.shape[1] // 2 - 50, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                                if alarm_played:
                                    alarm_sound.stop()
                                    alarm_played = False

                            if alarm_played and time.time() - alarm_start_time > 10:
                                alarm_sound.stop()
                                alarm_played = False

                            # Draw facial landmarks
                            for n in range(0, 68):
                                cv2.circle(frame, (landmarks_array[n, 0], landmarks_array[n, 1]), 1, (255, 0, 0), -1)

                # Display video in Streamlit
                video_frame.image(frame, channels="BGR")

                if stop_detection:
                    ride_end_time = time.strftime("%Y-%m-%d %H:%M:%S")  # Record end time
                    ride_duration = int((time.time() - ride_start_time) / 60)  # Calculate duration
                    st.success("Ride details saved successfully!")
                    break

            # Release camera and destroy windows
            cap.release()
            cv2.destroyAllWindows()
    elif selected == "About":
        background(r'C:\Users\nsgma\OneDrive\Desktop\D3\2454628.webp')
        # About the System Section
        st.markdown("""
        <h1 style='text-align: center; color: #00698f; font-size: 50px; font-family: "Arial", sans-serif; text-shadow: 2px 2px 5px rgba(0,0,0,0.5);'>
            About
        </h1>
        """, unsafe_allow_html=True)
        st.write("---")
        
        # Display video with custom styling
        video_file = r"C:\Users\nsgma\OneDrive\Desktop\D3\AV .mp4"  # Replace with your video file path
        st.video(video_file)

        # Introduction Section
        st.markdown("<h2 style='color: #2E8B57; font-family: Arial, sans-serif; text-shadow: 1px 1px 2px #000;'>🌟 Introduction</h2>", unsafe_allow_html=True)
        st.write("<p>The <b>Driver Drowsiness Detection System</b> is a cutting-edge solution aimed at addressing one of the most critical issues on the road: <b>driver fatigue</b>.</p>", unsafe_allow_html=True)
        st.write("<p>Statistics reveal that drowsy driving is a significant contributor to road accidents, leading to countless injuries and fatalities each year.</p>", unsafe_allow_html=True)
        st.write("<p>Fatigued drivers often exhibit slower reaction times and impaired judgment, increasing the risk of catastrophic accidents.</p>", unsafe_allow_html=True)

        # How it Works Section
        st.markdown("<h2 style='color: #2E8B57; font-family: Arial, sans-serif; text-shadow: 1px 1px 2px #000;'>🔍 How It Works</h2>", unsafe_allow_html=True)
        st.write("<p>Our system utilizes <b>advanced computer vision</b> and <b>machine learning</b> to monitor the driver's facial expressions, particularly focusing on eye movements and blinking patterns.</p>", unsafe_allow_html=True)
        st.write("<p>By analyzing the <b>Eye Aspect Ratio (EAR)</b> through real-time video feeds, the system detects signs of drowsiness, such as prolonged eye closure or frequent blinking, and promptly alerts the driver with an <b>auditory alarm</b>.</p>", unsafe_allow_html=True)

        # Why is this Important Section
        st.markdown("<h2 style='color: #2E8B57; font-family: Arial, sans-serif; text-shadow: 1px 1px 2px #000;'>🚦 Why Is This Important?</h2>", unsafe_allow_html=True)

        st.markdown("<h3 style='color: #FF6347; font-family: Arial, sans-serif; text-shadow: 1px 1px 2px #000;'>🛡️ Enhancing Road Safety</h3>", unsafe_allow_html=True)
        st.write("<p>Fatigue-related accidents can occur suddenly, especially during long or monotonous drives. Early detection provides crucial time for drivers to take necessary actions, such as pulling over or resting.</p>", unsafe_allow_html=True)

        st.markdown("<h3 style='color: #FF6347; font-family: Arial, sans-serif; text-shadow: 1px 1px 2px #000;'>⚠️ Preventing Fatalities</h3>", unsafe_allow_html=True)
        st.write("<p>Drowsy driving poses risks comparable to drunk driving. Research indicates that being awake for 18+ hours equates to a <b>blood alcohol content (BAC) of 0.05%</b>. Our system is designed to mitigate this risk and save lives.</p>", unsafe_allow_html=True)

        st.markdown("<h3 style='color: #FF6347; font-family: Arial, sans-serif; text-shadow: 1px 1px 2px #000;'>💰 Reducing Costs</h3>", unsafe_allow_html=True)
        st.write("<p>Fatigue-related accidents can incur substantial financial losses due to vehicle damage, legal fees, and medical expenses. Our system aims to alleviate these burdens.</p>", unsafe_allow_html=True)
    elif selected == "Profile":
        # Function to set a custom background with a gradient or attractive color
        background(r'C:\Users\nsgma\OneDrive\Desktop\D3\profile_bg.jpg')
        def profile_background():
            page_bg_style = '''
            <style>
            [data-testid="stAppViewContainer"] {
                background: linear-gradient(135deg, #74ebd5, #ACB6E5) !important; /* Soft blue gradient */
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }
            [data-testid="stHeader"], [data-testid="stToolbar"] {
                background: rgba(0,0,0,0); /* Hide header */
            }
            h1 {
                color: #ffffff; /* White for the profile heading */
                font-size: 40px;
                font-family: 'Segoe UI', sans-serif;
                text-align: center;
                margin-bottom: 30px;
                text-shadow: 2px 2px 5px rgba(0,0,0,0.5); /* Slight shadow for the heading */
                display: flex;
                justify-content: center;
                align-items: center;
            }
            h1 i {
                margin-right: 10px; /* Space between icon and text */
                font-size: 40px; /* Icon size */
            }
            h2 {
                color: #00698f; /* Blue for profile labels */
                font-size: 30px;
                font-family: 'Arial', sans-serif;
                margin-bottom: 5px;
                font-weight: bold;
                text-shadow: 1px 1px 3px rgba(0,0,0,0.3); /* Shadow for headings */
            }
            p {
                color: #ffffff; /* White for the profile details */
                font-size: 45px;
                font-family: 'Arial', sans-serif;
                background-color: rgba(0, 0, 0, 0.3); /* Slightly transparent black background */
                padding: 10px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2); /* Add box shadow for text sections */
            }
            .value {
                color: #FFD700; /* Gold color for values (e.g., "Mani Sunkari") */
                font-size: 40px; /* Slightly larger font size for values */
                font-weight: normal; /* Regular weight for values */
            }
            </style>
            '''
            st.markdown(page_bg_style, unsafe_allow_html=True)
        # Include Font Awesome
        st.markdown("""
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
        """, unsafe_allow_html=True)

        # Profile Heading with Font Awesome icon
        st.markdown("""
        <h1><i class="fas fa-user-circle"></i>Your Profile</h1>
        """, unsafe_allow_html=True)

        # Connect to database and fetch user data
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (st.session_state.username,))
        user_data = cursor.fetchone()
        conn.close()

        # Function to format text for profile fields
        def formatted_text(label, value):
            return f"<h2 style='color: #00698f;'>{label}:</h2><p style='font-size: 30px; font-weight: bold;'>{value}</p>"

        st.markdown("<style>.value {font-size: 50px; font-weight: bold; color: #333;}</style>", unsafe_allow_html=True)

        if user_data:
            st.markdown(formatted_text("Name", user_data[3]), unsafe_allow_html=True)
            st.markdown(formatted_text("Age", user_data[4]), unsafe_allow_html=True)
            st.markdown(formatted_text("Email", user_data[5]), unsafe_allow_html=True)
            st.markdown(formatted_text("Mobile", user_data[6]), unsafe_allow_html=True)
            st.markdown(formatted_text("Alt_moblie", user_data[7]), unsafe_allow_html=True)
            st.markdown(formatted_text("Vehicle Type", user_data[8]), unsafe_allow_html=True)  # Assuming vehicle_type is at index 8
        else:
            st.error("Unable to fetch user data.")

        # Logout button
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.experimental_rerun()
    elif selected == "Contact us":
        background(r'C:\Users\nsgma\OneDrive\Desktop\D3\dark2.jpg')
        
        # Contact Us Section with white heading color
        st.title("Contact Us")
        
        st.write("If you have any questions or need support, please feel free to reach out:")

        # Highlighted email
        st.markdown('<p style="font-size: 20px;font-weight: bold;color: #FF5733; /* Bright orange for the email */background-color: #FFF3CD; /* Light yellow background */padding: 5px 10px;border-radius: 5px;display: inline-block;">bhargavsunkari13@gmail.com</p>', unsafe_allow_html=True)
        
        st.write("---")

        # Contact details in a table
        data = [
            ["Leader", "BHARGAV SUNKARI ", "8106012509","bhargavsunkari13@gmail.com"],
            ["Team #1", "SANTHOSH GEMBALI", "7569926814","gembalisantoshkumar14@gmail.com"],
            ["Team #2", "PAVAN KUMAR SINGISETTI","9398207933","pavansingisetti@gmail.com"],
            ["Team #3", "ARAVIND KUMAR RAVIPUDI", "9100543554","aravindravipudi@gmail.com"]
        ]
        
        df = pd.DataFrame(data, columns=["Position","Name", "Phone Number","E-Mail"])
        st.table(df)

        # Chatbot Section
        st.write("---")
        st.subheader("Chat with Us")

        # Simple chatbot interface
        user_input = st.text_input("Type your message here...")

        if st.button("Send"):
            # Predefined responses
            responses = {
                "hi": "Hello! How can I assist you today?",
                "help": "Sure! What do you need help with?",
                "support": "You can reach our support team at bhargavsunkari13@gmail.com.",
                "thanks": "You're welcome! If you have any more questions, feel free to ask.",
                "bye": "Goodbye! Have a great day!"
            }
            # Get the response based on user input
            response = responses.get(user_input.lower(), "I'm sorry, I didn't understand that. Can you please rephrase?")
            
            # Display the chatbot response
            st.markdown(f"<p style='font-size: 18px; color: #007BFF; font-weight: bold;'>{response}</p>", unsafe_allow_html=True)