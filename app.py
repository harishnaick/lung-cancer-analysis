import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for,session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from PIL import Image
import sqlite3
import os
from flask_migrate import Migrate
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
# Load the trained models
model1 = load_model("models/lung_cancer_detection_model_cnn.h5")  # Model 1 (Histopathological)
model2 = load_model("models/lung_cancer_detection_2DCNN_15000.h5")  # Model 2 (CT Scan)

# Class labels for both models
class_labels_model1 = ['Benign', 'Malignant', 'Normal']
class_labels_model2 = ['Adenocarcinoma', 'Benign', 'Squamous Cell Carcinoma']

# Define the upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
from flask_sqlalchemy import SQLAlchemy

# Function to initialize database
def init_db():
     with sqlite3.connect('users.db') as conn:
            print("Opened database successfully")
            conn.execute('''
                CREATE TABLE IF NOT EXISTS customers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    password TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL
                );
            ''')
            conn.commit()  # Ensure changes are saved
            print("Table created successfully")
def init_contact_messages_db():
    with sqlite3.connect('users.db') as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS contact_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                message TEXT NOT NULL
            );
        ''')
        conn.commit()
init_db()
# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
@app.route('/home')
def home():
    return render_template('home.html')
# Route for prediction page
@app.route('/prediction')
def prediction_page():
    return render_template('index.html')
@app.route('/doctors')
def doctors():
    return render_template('doctors.html')
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # Get data from the form
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        # Insert the data into the contact_messages table
        with sqlite3.connect('users.db') as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO contact_messages (name, email, message) VALUES (?, ?, ?)", (name, email, message))
            conn.commit()
        print("We will reach you soon")
    return render_template('contact.html')

# Route for signup
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['username']
        password = request.form['password']
        email = request.form['email']

        with sqlite3.connect('users.db') as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM customers WHERE email=?", (email,))
            existing_user = cur.fetchone()

            if existing_user:
                return "User with this email already exists. <a href='/signup'>Go back to Signup</a>"
            else:
                cur.execute("INSERT INTO customers (name, password, email) VALUES (?, ?, ?)", (name, password, email))
                conn.commit()
                return redirect(url_for('login'))

    return render_template('signup.html')


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        with sqlite3.connect('users.db') as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM customers WHERE email=? AND password=?", (email, password))
            user = cur.fetchone()

            if user:
                session['user_id'] = user[0]
                return redirect(url_for('home'))
            else:
                return "Invalid email or password. <a href='{}'>Go back to Login</a>".format(url_for('login'))

    return render_template('login.html')

# Route for prediction using Model 1
@app.route('/predict_model1', methods=['POST'])
def predict_model1():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess the image for model 1 (Histopathological)
        img = Image.open(filepath).resize((224, 224))  # Resize image
        img_array = image.img_to_array(img)

        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict using Model 1
        predictions = model1.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = class_labels_model1[predicted_class]

        return render_template('result.html', prediction=predicted_label, image_path=filepath, model="Model 1")

    return redirect(request.url)

# Route for prediction using Model 2
@app.route('/predict_model2', methods=['POST'])
def predict_model2():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess the image for model 2 (CT Scan)
        img = image.load_img(filepath, target_size=(224, 224))  # Resize image
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = resnet_preprocess(img_array)  # Use appropriate preprocessing for model 2

        # Predict using Model 2
        predictions = model2.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = class_labels_model2[predicted_class]

        return render_template('result.html', prediction=predicted_label, image_path=filepath, model="Model 2")

    return redirect(request.url)


if __name__ == '__main__':
    app.run(debug=True)
