import os
# supress tensorflow logging other than errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.datasets import cifar10
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import LeakyReLU
from keras.callbacks import EarlyStopping
from keras.utils import np_utils

from attacks.fgsm import fgsm



img_rows = 32
img_cols = 32
img_chan = 3
input_shape=(img_rows, img_cols, img_chan)
nb_classes = 10

print('\nLoading cifar10')
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
print('\nX_train shape:', X_train.shape)
print('X_test shape:', X_train.shape)

y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)
print('\ny_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)


sess = tf.InteractiveSession()
K.set_session(sess)


if True:
    print('\nLoading model0')
    model0 = load_model('model/table_2_model0.h5')
else:
    print('\nBuilding model0')
    model0 = Sequential([
        Convolution2D(32, 3, 3, border_mode='same',
                      input_shape=input_shape),
        LeakyReLU(alpha=0.2),
        Convolution2D(32, 3, 3),
        LeakyReLU(alpha=0.2),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.2),
        Convolution2D(64, 3, 3, border_mode='same'),
        LeakyReLU(alpha=0.2),
        Convolution2D(64, 3, 3),
        LeakyReLU(alpha=0.2),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        Convolution2D(128, 3, 3, border_mode='same'),
        LeakyReLU(alpha=0.2),
        Convolution2D(128, 3, 3),
        LeakyReLU(alpha=0.2),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),
        Flatten(),
        Dense(512),
        Activation('relu'),
        Dropout(0.5),
        Dense(nb_classes),
        Activation('softmax')])

    model0.compile(loss='categorical_crossentropy',
                   optimizer='adam', metrics=['accuracy'])

    earlystopping = EarlyStopping(monitor='val_loss', patience=5,
                                  verbose=1)
    model0.fit(X_train, y_train, nb_epoch=100, validation_split=0.1,
               callbacks=[earlystopping])

    print('\nSaving model0')
    os.makedirs('model', exist_ok=True)
    model0.save('model/table_2_model0.h5')


x = tf.placeholder(tf.float32, (None, img_rows, img_cols, img_chan))
eps = tf.placeholder(tf.float32, ())
x_adv = fgsm(model0, x, nb_epoch=9, eps=eps)


print('\nTesting against clean test data')
score = model0.evaluate(X_test, y_test)
print('\nloss: {0:.4f} acc: {1:.4f}'.format(score[0], score[1]))


if False:
    for EPS in [0.01, 0.03, 0.1, 0.3]:
        print('\nBuilding X_train_adv with eps={0:.2f}'.format(EPS))
        nb_sample = X_train.shape[0]
        batch_size = 128
        nb_batch = int(np.ceil(nb_sample/batch_size))
        X_train_adv = np.empty(X_train.shape)
        for batch in range(nb_batch):
            print(' batch {0}/{1}'.format(batch+1, nb_batch),
                  end='\r')
            start = batch * batch_size
            end = min(nb_sample, start+batch_size)
            tmp = sess.run(x_adv, feed_dict={x: X_train[start:end],
                                             eps: EPS,
                                             K.learning_phase(): 0})
            X_train_adv[start:end] = tmp

        print('\nBuilding X_test_adv with eps={0:.2f}'.format(EPS))
        nb_sample = X_test.shape[0]
        nb_batch = int(np.ceil(nb_sample/batch_size))
        X_test_adv = np.empty(X_test.shape)
        for batch in range(nb_batch):
            print(' batch {0}/{1}'.format(batch+1, nb_batch),
                  end='\r')
            start = batch * batch_size
            end = min(nb_sample, start+batch_size)
            tmp = sess.run(x_adv, feed_dict={x: X_test[start:end],
                                             eps: EPS,
                                             K.learning_phase(): 0})
            X_test_adv[start:end] = tmp

        print('\nSaving adversarial images')
        os.makedirs('data/', exist_ok=True)
        np.savez('data/table_2_{0:.2f}.npz'.format(EPS),
                 X_train_adv=X_train_adv, X_test_adv=X_test_adv)


    print('\nTesting against adversarial test data')
    score = model0.evaluate(X_test_adv, y_test)
    print('\nloss: {0:.4f} acc: {1:.4f}'.format(score[0], score[1]))


y0_test = np.zeros((y_test.shape[0], 1))
y1_test = np.ones((y_test.shape[0], 1))


if True:
    print('\nLoading model1')
    model1 = load_model('model/table_2_model1.h5')
else:
    print('\nLoading adversarial data with eps=0.03')
    db = np.load('data/table_2_0.03.npz')
    X_train_adv = db['X_train_adv']
    X_test_adv = db['X_test_adv']

    print('\nPreparing clean/adversarial mixed dataset')
    X_all_train = np.vstack([X_train, X_train_adv])
    y_all_train = np.vstack([np.zeros([X_train.shape[0], 1]),
                             np.ones([X_train_adv.shape[0], 1])])

    ind = np.random.permutation(X_all_train.shape[0])
    X_all_train = X_all_train[ind]
    y_all_train = y_all_train[ind]

    print('\nBuilding model1')
    model1 = Sequential([
        Convolution2D(32, 3, 3, border_mode='same',
                      input_shape=input_shape),
        LeakyReLU(alpha=0.2),
        Convolution2D(32, 3, 3),
        LeakyReLU(alpha=0.2),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.2),
        Convolution2D(64, 3, 3, border_mode='same'),
        LeakyReLU(alpha=0.2),
        Convolution2D(64, 3, 3),
        LeakyReLU(alpha=0.2),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256),
        Activation('relu'),
        Dropout(0.5),
        Dense(1),
        Activation('sigmoid')])

    model1.compile(loss='binary_crossentropy',
                   optimizer='adam', metrics=['accuracy'])

    os.makedirs('model', exist_ok=True)
    model1.fit(X_all_train, y_all_train, nb_epoch=2,
               validation_split=0.1)

    print('\nSaving model1')
    model1.save('model/table_2_model1.h5')


for EPS in [0.01, 0.03, 0.1, 0.3]:
    print('\nLoading adversarial data with eps={0:.2f}'.format(EPS))
    db = np.load('data/table_2_{0:.2f}.npz'.format(EPS))
    X_train_adv = db['X_train_adv']
    X_test_adv = db['X_test_adv']

    print('\nTesting against clean test data')
    score = model1.evaluate(X_test, y0_test)
    print('\nloss: {0:.4f} acc: {1:.4f}'.format(score[0], score[1]))

    print('\nTesting against adversarial test data')
    score = model1.evaluate(X_test_adv, y1_test)
    print('\nloss: {0:.4f} acc: {1:.4f}'.format(score[0], score[1]))
