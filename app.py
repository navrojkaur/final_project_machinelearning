from flask import Flask, request, jsonify, render_template, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Ensure the folder exists for storing images
os.makedirs('./static/images', exist_ok=True)

model = tf.keras.models.load_model('final_model.keras')

CIFAR10_CLASSES = [
    'Airplane',
    'Automobile',
    'Bird',
    'Cat',
    'Deer',
    'Dog',
    'Frog',
    'Horse',
    'Ship',
    'Truck'
]

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = os.path.join('static', 'images', imagefile.filename)
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(32, 32))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    prediction = model.predict(image)
    predicted_index = np.argmax(prediction[0])
    predicted_label = CIFAR10_CLASSES[predicted_index]

    return render_template('index.html', prediction=predicted_label, image_url=url_for('static', filename='images/' + imagefile.filename))

if __name__ == '__main__':
    app.run(debug=True)
