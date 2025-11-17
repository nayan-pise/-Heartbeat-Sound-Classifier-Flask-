from flask import Flask, request, render_template, redirect, url_for
import os
import numpy as np
import scipy.io.wavfile as wav
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the dataset
def load_audio_files(data_dir):
    """Loads audio files and extracts basic statistical features."""
    features, labels = [], []
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".wav"):
            label = file_name.split('_')[0]  # Assuming label is part of the file name
            file_path = os.path.join(data_dir, file_name)
            
            # Read audio file
            try:
                sr, audio = wav.read(file_path)  # sr = sampling rate
                # Feature extraction: statistical features from the raw signal
                feature = [
                    np.mean(audio),  # Mean amplitude
                    np.std(audio),   # Standard deviation
                    np.max(audio),   # Maximum amplitude
                    np.min(audio),   # Minimum amplitude
                    np.ptp(audio),   # Peak-to-peak range
                ]
                features.append(feature)
                labels.append(label)
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
    return np.array(features), np.array(labels)

# Step 2: Prepare the dataset
def prepare_dataset(data_dir):
    features, labels = load_audio_files(data_dir)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, labels

# Step 3: Train the model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Step 4: Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# Step 5: Predict a new file
def predict_heartbeat(file_path, model, label_encoder):
    sr, audio = wav.read(file_path)
    feature = [
        np.mean(audio),
        np.std(audio),
        np.max(audio),
        np.min(audio),
        np.ptp(audio),
    ]
    feature = np.array(feature).reshape(1, -1)
    prediction = model.predict(feature)
    predicted_label = label_encoder.inverse_transform(prediction)
    return predicted_label[0]

# Main Function
if __name__ == "__main__":
    DATA_DIR = r"C:\Users\mohit\OneDrive\Desktop\Python Programming\Project\set_a copy"  # Replace with your dataset path

    # Step 1: Prepare the dataset
    X_train, X_test, y_train, y_test, Labels = prepare_dataset(DATA_DIR)

    # Step 2: Train the model
    model = train_model(X_train, y_train)

    # Step 3: Evaluate the model
    evaluate_model(model, X_test, y_test)

from flask import Flask, request, render_template, redirect, url_for
import os
import numpy as np
import scipy.io.wavfile as wav
import pickle

app = Flask(__name__)
UPLOAD_FOLDER = 'Uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model and label encoder
with open("Final2322.pkl", 'rb') as f:
    model = pickle.load(f)

with open("label_encoder.pkl", 'rb') as h:
    label_encoder = pickle.load(h)

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to extract features from an audio file
def extract_features(file_path):
    sr, audio = wav.read(file_path)
    feature = [
        np.mean(audio),
        np.std(audio),
        np.max(audio),
        np.min(audio),
        np.ptp(audio),
    ]
    return np.array(feature).reshape(1, -1)

# Predict function
def predict_heartbeat(file_path, model, label_encoder):
    features = extract_features(file_path)
    prediction = model.predict(features)
    predicted_label = label_encoder.inverse_transform(prediction)
    return predicted_label[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            try:
                predicted_label = predict_heartbeat(file_path, model, label_encoder)
            except AttributeError:
                return "Error: Model object is not valid for prediction. Please check the model file."
            except Exception as e:
                return f"Error: {e}"

            return render_template("Frontend1.html", prediction=predicted_label)

    return render_template('Frontend1.html')

if __name__ == '__main__':
    app.run(debug=True)
