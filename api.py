from flask import Blueprint, render_template, request
from flask_cors import CORS, cross_origin
import os
import base64
import io
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image

# Model saved with Keras model.save()
MODEL_PATH = "models/tf_lite_model_64.tflite"

# Load your trained model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


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


def model_predict(image):

    if image.mode == "RGBA":
        image = convert_rgba_to_rgb(image)

    img = np.array(image.resize((64, 64)))
    a = np.float32(img)

    x = np.true_divide(a, 255)
    x = np.delete(x, 3, 2)
    x = np.expand_dims(x, axis=0)

    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]["index"])
    return preds


api = Blueprint("api", __name__)


@api.route("/web", methods=["GET", "POST"])
@cross_origin()
def from_web():
    if request.method == "POST":
        # Get the file from post request
        base64_string = request.json["image"]  # base64String is key value
        base64_data = base64_string.split(",")

        img_data = base64.b64decode(base64_data[1])

        img = Image.open(io.BytesIO(img_data))

        # Make prediction
        preds = model_predict(img)

        # Process your result for human
        pred_class = np.argmax(preds, axis=1)  # Simple argmax

        # Change label to class
        if pred_class[0] == 0:
            result = "Immature"
        elif pred_class[0] == 1:
            result = "Mature"
        else:
            result = "Normal"

        # result = str(pred_class[0])                 # Convert to string

        return result
    return None


@api.route("/android", methods=["GET", "POST"])
@cross_origin()
def from_android():
    if request.method == "POST":
        # Get the file from post request
        base64_string = request.json["image"]  # base64String is key value
        base64_data = base64_string.split(",")

        img_data = base64.b64decode(base64_data[1])

        img = Image.open(io.BytesIO(img_data))

        # Make prediction
        preds = model_predict(img)

        # Process your result for human
        pred_class = np.argmax(preds, axis=1)  # Simple argmax

        # Change label to class
        if pred_class[0] == 0:
            result = "Immature"
        elif pred_class[0] == 1:
            result = "Mature"
        else:
            result = "Normal"

        # result = str(pred_class[0])                 # Convert to string

        return result
    return None
