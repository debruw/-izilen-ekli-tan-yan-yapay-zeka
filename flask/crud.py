from flask import Flask, request, jsonify, render_template
import os

import numpy as np
from keras.models import load_model


app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))


@app.route("/")
def index():
    global model
    model = load_model('C:/my_model.h5')
    return render_template('index.html')


# endpoint to create new user
@app.route("/index", methods=["POST"])
def doodle():
    # Reading json data in web request
    language = request.json['language']
    width = request.json['writing_guide']['width']
    height = request.json['writing_guide']['height']
    ink = request.json["ink"][0]

    # Converting coordinates
    x = np.zeros(len(ink[0]))
    y = np.zeros(len(ink[0]))
    nmpy = np.zeros(28*28)
    oran1 = 28 / width
    oran2 = 28 / height
    for i in range(0, len(ink[0])):
        x[i] = round(ink[0][i] * oran1)
        y[i] = round(ink[1][i] * oran2)

    # Recreating image
    for j in range(0, len(x)):
        ind = x[j] + (y[j] * 28)
        nmpy[int(ind)] = 255

    # Saving image
    np.save('bir.npy', nmpy)

    # Reshaping image
    image_cnn = nmpy.reshape(1, 1, 28, 28).astype('float32')

    label_dict = {0: 'Daire', 1: 'Çizgi', 2: 'Dörtgen', 3: 'Yıldız', 4: 'Üçgen'}

    # Running predict
    pred = model.predict(image_cnn, batch_size=32, verbose=0)
    digit = np.argmax(pred)

    # Sending response
    return jsonify(label_dict[digit])


if __name__ == '__main__':
    app.run(debug=True)
