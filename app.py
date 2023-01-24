from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras import models ,layers
import cv2
import numpy as np
import pandas as pd

app = Flask(__name__)

model = tf.keras.models.load_model("potato_model.h5")
print( "Model is loaded")

labels = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']


@app.route('/')
def index():
	return render_template("index.html", data="hey")


@app.route("/prediction", methods=["POST"])
def prediction():

	img = request.files['img']
	img.save("img.jpg")

	image = cv2.imread("img.jpg")
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (256,256))
	image = np.reshape(image, (1,256,256,3))
	pred = model.predict(image)
	pred = np.argmax(pred)
	
	pred = labels[pred]

	return render_template("prdiction.html", data=pred)


if __name__ == "__main__":
	app.run(debug=True)