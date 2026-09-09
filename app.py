from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import cv2
from PIL import Image
import joblib
from mtcnn import MTCNN
import xgboost as xgb
import os
from skincare_recommendations import skincare_suggestions

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure key
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Initialize Flask-Login
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

# Load user for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create the database
with app.app_context():
    db.create_all()

# Load the trained model
model = joblib.load('ml/recommendation_model.joblib')

# Initialize MTCNN face detector
detector = MTCNN()

# Color palettes
color_palettes = {
    "Pastels": [("#FFB6C1", "Pastel Pink"), ("#D8BFD8", "Thistle")],
    "Earthy Tones": [("#8B4513", "Saddle Brown"), ("#A52A2A", "Brown")],
    "Warm Tones": [("#FF4500", "Orange Red"), ("#FF6347", "Tomato")],
    "Neutrals": [("#808080", "Grey"), ("#A9A9A9", "Dark Grey")],
}

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('home'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')  # Corrected hashing method
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/main2', methods=['GET', 'POST'])
@login_required
def main2():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({"error": "No image file received"}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No image selected"}), 400

        # Read and process the image
        image = Image.open(file.stream)
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Detect faces
        faces = detector.detect_faces(image_np)
        if len(faces) == 0:
            return render_template('main2.html', error="No face detected in the image.")

        # Analyze skin tone
        x, y, w, h = faces[0]['box']
        skin_tone = analyze_skin_tone(image_np, x, y, w, h)

        # Predict color palette
        skin_tone_features = [skin_tone]
        dmatrix_input = xgb.DMatrix(skin_tone_features)
        predicted_index = int(model.predict(dmatrix_input)[0])
        palette_names = list(color_palettes.keys())
        palette_name = palette_names[predicted_index]

        # Get recommended colors
        recommendations = color_palettes.get(palette_name, [])
        return render_template('main2.html', skin_tone=skin_tone, recommendations=recommendations, palette_name=palette_name)

    return render_template('main2.html')

def analyze_skin_tone(image, x, y, w, h):
    face_roi = image[y:y+h, x:x+w]
    face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    reshaped = face_roi.reshape((-1, 3))
    reshaped = np.float32(reshaped)

    # Use K-Means clustering to find dominant color
    num_clusters = 3
    _, _, centers = cv2.kmeans(
        reshaped, num_clusters, None,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS
    )
    dominant_color = centers[0]
    return tuple(map(int, dominant_color))

@app.route('/skincare', methods=['GET', 'POST'])
@login_required
def skincare():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({"error": "No file uploaded."}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected."}), 400

        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Detect face and skin region
        skin_region, error = detect_face_and_skin(file_path)
        if error:
            return render_template('upload.html', error=error)

        # Analyze skin for redness, dryness, and pimples
        redness_score = detect_redness(skin_region)
        dryness_score = detect_dryness(skin_region)
        pimple_count = detect_pimples(skin_region)

        # Interpret analysis results
        redness_level = "High" if redness_score > 0.2 else "Moderate" if redness_score > 0.1 else "Low"
        dryness_level = "Severe" if dryness_score > 150 else "Moderate" if dryness_score > 100 else "Normal"
        pimple_level = f"{pimple_count} pimple(s)" if pimple_count > 0 else "No pimples detected"

        # Generate recommendations
        recommendations = {
            "redness": skincare_suggestions.get("redness", {}).get("severe" if redness_score > 0.2 else "mild", {}),
            "dryness": skincare_suggestions.get("dryness", {}).get("severe" if dryness_score > 150 else "mild", {}),
            "pimples": skincare_suggestions.get("pimples", {}).get("moderate" if pimple_count > 2 else "mild", {})
        }

        return render_template('results.html',
                               redness_level=redness_level,
                               dryness_level=dryness_level,
                               pimple_level=pimple_level,
                               recommendations=recommendations,
                               image_url=file_path)

    return render_template('upload.html')

def detect_face_and_skin(image_path):
    image = cv2.imread(image_path)
    detector = MTCNN()
    results = detector.detect_faces(image)
    if len(results) == 0:
        return None, "No face detected. Please upload a clear image."
    x, y, width, height = results[0]['box']
    cropped_face = image[y:y + height, x:x + width]
    return cropped_face, None

def detect_redness(skin_region):
    hsv = cv2.cvtColor(skin_region, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    redness_score = cv2.countNonZero(mask) / (skin_region.shape[0] * skin_region.shape[1])
    return redness_score

def detect_dryness(skin_region):
    gray = cv2.cvtColor(skin_region, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    dryness_score = np.mean(enhanced)
    return dryness_score

def detect_pimples(skin_region):
    hsv = cv2.cvtColor(skin_region, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pimple_count = sum(50 < cv2.contourArea(c) < 500 for c in contours)
    return pimple_count

if __name__ == "__main__":
    app.run(debug=True)