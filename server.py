#!flask/bin/python
import cv2
import numpy
import tensorflow as tf
from flask import Flask, request, abort
from flask.json import jsonify
from flask_cors import CORS
from keras.models import model_from_json

from common import get_label_by_matrix

app = Flask(__name__)
CORS(app)

@app.route('/classify', methods=['POST'])
def classify():
    if request.method == 'POST' and 'image' in request.files:
        npimg = numpy.fromfile(request.files['image'], numpy.uint8)
        # convert numpy array to image
        img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64)).reshape(1, 64, 64, 1)

        with graph.as_default():
            model_out = model.predict([img])
            response = {'result': get_label_by_matrix(model_out)}
            return jsonify(response)

    return abort(400)


def load_model():
    global model
    with open('trained_model/model_architecture.json', 'r') as f:
        model = model_from_json(f.read())
        model.load_weights('trained_model/model_weights.h5')

    # https://github.com/keras-team/keras/issues/6462
    global graph
    graph = tf.get_default_graph()


if __name__ == '__main__':
    load_model()
    app.run(debug=True)
