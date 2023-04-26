from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
from io import BytesIO
import joblib
import cv2
import numpy as np
import pickle

# load the pre-trained model
model =joblib.load("C:/Users/Dell/Desktop/SaBaHaT/MP/Mini Project 2/svm_model.pkl")

# initialize the Flask application
app = Flask(__name__)

# define a function to preprocess the image
def preprocess_image(image_path):
    # Load the image using OpenCV
    img = cv2.imread(image_path)
    # Resize the image to 256x256 pixels
    img = cv2.resize(img, (256, 256))
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Flatten the image into a 1D array
    img_array = gray.flatten()
    # Return the flattened image array
    return img_array

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        # getting the image from the form in html
        img = request.files['image']

        # saving the image to a temporary file
        img_path = 'tmp.jpg'
        img.save(img_path)

        # preprocessing the image
        img = preprocess_image(img_path)

        # reshape the image
        img = img.reshape(1, -1)  

        # make a prediction
        preds = model.predict(img)
        class_idx = preds[0]
        class_names = ['Non Demented', 'Demented']  
        class_name = class_names[class_idx]

        # return the predicted class to the HTML page
        return render_template('main.html', prediction=class_name)
    else:
        return render_template('main.html')

if __name__ == '__main__':
    app.run(debug=True)
