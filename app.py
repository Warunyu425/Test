from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.secret_key = 'your_secret_key'  # Required for flashing messages

# Load the model
try:
    model = load_model('model/garbage_classification_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)

# Define the classes
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

def predict_image(image_path):
    try:
        img = load_img(image_path, target_size=(256, 256))  # Change to (300, 300) based on your model's input size
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        predictions = model.predict(img)
        return classes[np.argmax(predictions)]
    except Exception as e:
        print("Error in prediction:", e)
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the file is in the request
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # Save and predict only if a file is selected
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            try:
                file.save(filepath)
                predicted_class = predict_image(filepath)
                if predicted_class:
                    return render_template('result.html', image_url=filepath, prediction=predicted_class)
                else:
                    flash('Error predicting the image')
                    return redirect(request.url)
            except Exception as e:
                print("Error saving or processing the file:", e)
                flash('An error occurred while processing the file')
                return redirect(request.url)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
