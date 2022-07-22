from __future__ import division, print_function

# coding=utf-8
import os
import numpy as np
from PIL import Image

from decouple import config


# Keras
import tensorflow as tf
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

import base64
import io

# Encryption System
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad


# Firestore Database
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Use the application default credentials
cred = credentials.Certificate("find-cataract-firebase-adminsdk-4wavy-e79c86a2cd.json")
firebase_admin.initialize_app(cred)

db = firestore.client()


# Define a flask app
app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

# Model saved with Keras model.save()
MODEL_PATH = "models/tf_lite_model_64.tflite"

# Load your trained model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# CBC with Fix IV

# data = "I love Medium"

key = config("AES_KEY")
iv = config("AES_IV")
# key = "6CYYqT8cbFcWj7QW2VhjN37ctdqU9Udj"  # 32 char for AES256

# FIX IV
# iv = "7299238835873542".encode("utf-8")  # 16 char for AES128


# def pad(s):
#     block_size = 16
#     remainder = len(s) % block_size
#     padding_needed = block_size - remainder
#     return s + padding_needed * " "


def encrypt(data, key, iv):
    data = pad(data.encode(), AES.block_size)
    cipher = AES.new(key.encode("utf-8"), AES.MODE_CBC, iv.encode("utf-8"))
    return base64.b64encode(cipher.encrypt(data))


def decrypt(enc, key, iv):
    enc = base64.b64decode(enc)
    cipher = AES.new(key.encode("utf-8"), AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(enc), 16)


def convert_rgba_to_rgb(img):
    try:
        img.load()  # needed for split()
        x = Image.new("RGB", img.size, (255, 255, 255))
        x.paste(img, mask=img.split()[3])  # 3 is the alpha channel
        x.save("temp.jpg", "JPEG", quality=100)

        xx = Image.open("temp.jpg")
        return xx

    except Exception as e:
        return None


def model_predict(img, type):
    if type == "file":
        img = image.load_img(img, target_size=(64, 64))

    if img.mode == "RGBA":
        img = convert_rgba_to_rgb(img)

    img = np.array(img.resize((64, 64)))
    a = np.float32(img)

    x = np.true_divide(a, 255)
    # x = np.delete(x, 3, 2)
    x = np.expand_dims(x, axis=0)

    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]["index"])
    return preds


@app.route("/", methods=["GET"])
def index():
    # Main page
    return render_template("index.html")


@app.route("/api", methods=["GET", "POST"])
@cross_origin()
def upload_api():
    if request.method == "POST":
        try:
            # Get the file from post request
            # f = request.files['file']
            userid = request.json["uid"]

            base64_string = request.json["image"]

            # base64 reading behaviour
            try:
                base64_data = base64_string.split(",")
                img_data = base64.b64decode(base64_data[1])
            except:
                img_data = base64.b64decode(base64_string)

            img = Image.open(io.BytesIO(img_data))

            # Make prediction
            preds = model_predict(img, "api")

            # Process your result for human
            pred_class = np.argmax(preds, axis=1)  # Simple argmax

            # Change label to class
            if pred_class[0] == 0:
                result = "Immature"
            elif pred_class[0] == 1:
                result = "Mature"
            else:
                result = "Normal"

            doc_ref = db.collection("users").document(userid)

            result_enc = encrypt(result, key, iv).decode("utf-8", "ignore")

            doc_ref.update({"result": result_enc, "timestamp": firestore.SERVER_TIMESTAMP})

            return ""
        except Exception as e:
            return "Not Recognized - Error: {}".format(str(e))


@app.route("/predict", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        try:
            # Get the file from post request
            f = request.files["file"]

            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath, "uploads", secure_filename(f.filename))
            f.save(file_path)

            # Make prediction
            preds = model_predict(file_path, "file")

            # Process your result for human
            pred_class = np.argmax(preds, axis=1)  # Simple argmax

            doc_ref = db.collection("users").document("aturing")

            # Change label to class
            if pred_class[0] == 0:
                result = "Immature"
            elif pred_class[0] == 1:
                result = "Mature"
            else:
                result = "Normal"

            result_enc = encrypt(result, key, iv).decode("utf-8", "ignore")

            doc_ref.update({"result": result_enc, "timestamp": firestore.SERVER_TIMESTAMP})

            return result
        except Exception as e:
            return "Not Recognized - Error: {}".format(str(e))


if __name__ == "__main__":
    app.run(debug=True, port=3000)
