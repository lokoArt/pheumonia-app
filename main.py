#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 07:16:04 2018
@author: raghav prabhu
Re-modified TensorFlow classification file according to our need.
"""
import numpy as np
import tensorflow as tf
import sys
import os
import csv

# Disable tensorflow compilation warnings


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
Classify images from test folder and predict dog breeds along with score.
'''


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def classify_image(image_path):
    graph = load_graph('trained_model/retrained_graph.pb')

    wrong_count = 0

    files = os.listdir(image_path)
    with tf.Session(graph=graph) as sess:
        for file in files:
            t = read_tensor_from_image_file(image_path + '/' + file)

            input_operation = graph.get_operation_by_name('import/Placeholder')
            output_operation = graph.get_operation_by_name('import/final_result')

            results = sess.run(output_operation.outputs[0], {
                input_operation.outputs[0]: t
            })

            results = np.squeeze(results)

            top_k = results.argsort()[-5:][::-1]

            labels = load_labels('trained_model/retrained_labels.txt')
            for i in top_k:
                print(labels[i], results[i])

            if results[0] < results[1]:
                wrong_count += 1

    print('Total wrong {}'.format(wrong_count))

def main():
    classify_image('dataset/test/NORMAL')


if __name__ == '__main__':
    main()
