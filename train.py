from random import shuffle

import cv2
import numpy as np
import os

from keras import Sequential
from keras.engine import InputLayer
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.models import model_from_json
from keras import preprocessing

from common import get_label_matrix_by_folder

train_data = 'dataset/train'
test_data = 'dataset/test'


def load_data(base_folder):
    train_images = []

    for folder in ['NORMAL', 'PNEUMONIA']:
        for filename in os.listdir('{}/{}'.format(base_folder, folder)):
            if not filename.endswith(".jpeg"):
                continue

            path = os.path.join(base_folder, folder, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            train_images.append([np.array(img), get_label_matrix_by_folder(folder)])

    shuffle(train_images)
    return train_images


def main():
    training_images = load_data(train_data)

    tr_img_data = np.array([i[0] for i in training_images]).reshape(-1, 64, 64, 1)
    tr_lbl_data = np.array([i[1] for i in training_images])

    model = Sequential()

    model.add(InputLayer(input_shape=[64, 64, 1]))
    model.add(Conv2D(filters=4, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=5, padding='same'))

    model.add(Conv2D(filters=8, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=5, padding='same'))

    model.add(Conv2D(filters=10, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=5, padding='same'))

    model.add(Conv2D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=5, padding='same'))

    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    datagen = preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        shear_range=0.2)

    model.fit_generator(datagen.flow(tr_img_data, tr_lbl_data, batch_size=256), epochs=500)
    model.summary()

    # Save the weights
    model.save_weights('trained_model/model_weights.h5')

    # Save the model architecture
    with open('trained_model/model_architecture.json', 'w') as f:
        f.write(model.to_json())

    test()


def test():
    with open('trained_model/model_architecture.json', 'r') as f:
        model = model_from_json(f.read())
    model.load_weights('trained_model/model_weights.h5')

    test_images = load_data(test_data)
    wrong = 0
    correct = 0
    for image in test_images:
        data = image[0].reshape(1, 64, 64, 1)
        model_out = model.predict([data])

        if np.argmax(model_out) != np.argmax(image[1]):
            wrong += 1
        else:
            correct += 1

    print('Wrong {}, correct {}'.format(wrong, correct))


if __name__ == '__main__':
    main()
