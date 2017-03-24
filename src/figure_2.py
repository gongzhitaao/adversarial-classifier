import os
# supress tensorflow logging other than errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from attacks.fgsm import fgsm
from attacks.jsma import jsma



def decompose(v, i):
    """Decompose v so that v = a*i+b*j where i is orthogonal to j."""
    i /= np.linalg.norm(i)
    a = np.dot(v, i) / np.dot(i, i)
    j = v - a*i
    b = np.linalg.norm(j)
    j /= b
    return (a, i), (b, j)


img_rows = 28
img_cols = 28
img_chan = 1
nb_classes = 10
input_shape=(img_rows, img_cols, img_chan)


print('\nLoading mnist')
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

X_train = X_train.reshape(-1, img_rows, img_cols, img_chan)
X_test = X_test.reshape(-1, img_rows, img_cols, img_chan)
print('\nX_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)

# one hot encoding
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)


sess = tf.InteractiveSession()
K.set_session(sess)


if True:
    print('\nLoading model0')
    model0 = load_model('model/figure_2_model0.h5')
else:
    print('\nBuilding model0')
    model0 = Sequential([
        Convolution2D(32, 3, 3, input_shape=input_shape),
        Activation('relu'),
        Convolution2D(32, 3, 3),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        # Dropout(0.25),
        Flatten(),
        Dense(128),
        Activation('relu'),
        # Dropout(0.5),
        Dense(10),
        Activation('softmax')])

    model0.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print('\nTraining model0')
    model0.fit(X_train, y_train, nb_epoch=10)

    print('\nSaving model0')
    os.makedirs('model', exist_ok=True)
    model0.save('model/figure_2_model0.h5')


x = tf.placeholder(tf.float32, (None, img_rows, img_cols, img_chan))
y = tf.placeholder(tf.int32, (None, ))
x_adv_fgsm = fgsm(model0, x, eps=0.25, nb_epoch=1)
x_adv_jsma = jsma(model0, x, y, nb_epoch=0.2)


print('\nTesting against clean data')
score = model0.evaluate(X_test, y_test)
print('\nloss: {0:.4f} acc: {1:.4f}'.format(score[0], score[1]))


if False:
    print('\nLoading FGSM/JSMA adversarial datasets')
    db = np.load('data/figure_2.npz')
    X_adv_fgsm, X_adv_jsma = db['X_adv_fgsm'], db['X_adv_jsma']
else:
    print('\nGenerating adversarial via FGSM')
    batch_size = 64
    X_adv_fgsm = np.empty(X_test.shape)
    nb_sample = X_test.shape[0]
    nb_batch = int(np.ceil(nb_sample/batch_size))
    for batch in range(nb_batch):
        print('batch {0}/{1}'.format(batch+1, nb_batch), end='\r')
        start = batch * batch_size
        end = min(nb_sample, start+batch_size)
        tmp = sess.run(x_adv_fgsm, feed_dict={x: X_test[start:end],
                                              K.learning_phase(): 0})
        X_adv_fgsm[start:end] = tmp

    print('\nGenerating adversarial via JSMA')
    batch_size = 64
    X_adv_jsma = np.empty(X_test.shape)
    z_test = np.argmax(y_test, axis=1)
    targets = np.arange(10)
    for i in range(10):
        print('\nModifying digit {0}'.format(i))
        ind, = np.where(z_test==i)
        X_i = X_test[ind]
        nb_sample = X_i.shape[0]
        t_i = np.random.choice(targets[targets!=i], nb_sample)
        nb_batch = int(np.ceil(nb_sample/batch_size))
        for batch in range(nb_batch):
            print('batch {0}/{1}'.format(batch+1, nb_batch), end='\r')
            start = batch * batch_size
            end = min(nb_sample, start+batch_size)
            tmp = sess.run(x_adv_jsma, feed_dict={
                x: X_i[start:end], y: t_i[start:end],
                K.learning_phase(): 0})
            X_adv_jsma[ind[start:end]] = tmp

    print('\nSaving adversarials')
    os.makedirs('data', exist_ok=True)
    np.savez('data/figure_2.npz', X_adv_fgsm=X_adv_fgsm,
             X_adv_jsma=X_adv_jsma)


print('\nTesting against FGSM adversarial data')
score = model0.evaluate(X_adv_fgsm, y_test)
print('\nloss: {0:.4f} acc: {1:.4f}'.format(score[0], score[1]))

print('\nTesting against JSMA adversarial data')
score = model0.evaluate(X_adv_jsma, y_test)
print('\nloss: {0:.4f} acc: {1:.4f}'.format(score[0], score[1]))


if True:
    print('\nLoading model1')
    model1 = load_model('model/figure_2_model1.h5')
else:
    print('\nBuilding model1')
    model1 = Sequential([
        Convolution2D(32, 3, 3, input_shape=input_shape),
        Activation('relu'),
        Convolution2D(32, 3, 3),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128),
        Activation('relu'),
        Dropout(0.5),
        Dense(10),
        Activation('softmax')])

    model1.compile(loss='categorical_crossentropy',
                   optimizer='adam', metrics=['accuracy'])

    x_adv_tmp = fgsm(model1, x, eps=0.25, nb_epoch=1)

    print('\nDummy testing')
    model1.evaluate(X_test[:10], y_test[:10], verbose=0)


    print('\nPreparing training/validation dataset')
    validation_split = 0.1
    N = int(X_train.shape[0]*validation_split)
    X_tmp_train, X_tmp_val = X_train[:-N], X_train[-N:]
    y_tmp_train, y_tmp_val = y_train[:-N], y_train[-N:]


    print('\nTraining model1')
    nb_epoch = 10
    batch_size = 64
    nb_sample = X_tmp_train.shape[0]
    nb_batch = int(np.ceil(nb_sample/batch_size))
    for epoch in range(nb_epoch):
        print('Epoch {0}/{1}'.format(epoch+1, nb_epoch))
        for batch in range(nb_batch):
            print(' batch {0}/{1} '.format(batch+1, nb_batch),
                  end='\r', flush=True)
            start = batch * batch_size
            end = min(nb_sample, start+batch_size)
            X_tmp_adv = sess.run(x_adv_tmp, feed_dict={
                x: X_tmp_train[start:end], K.learning_phase(): 0})
            y_tmp_adv = y_tmp_train[start:end]
            X_batch = np.vstack((X_tmp_train[start:end], X_tmp_adv))
            y_batch = np.vstack((y_tmp_train[start:end], y_tmp_adv))
            score = model1.train_on_batch(X_batch, y_batch)
        score = model1.evaluate(X_tmp_val, y_tmp_val)
        print(' loss: {0:.4f} acc: {1:.4f}'
              .format(score[0], score[1]))

    print('\nSaving model1')
    os.makedirs('model', exist_ok=True)
    model1.save('model/figure_2_model1.h5')


print('\nTesting against FGSM adversarial')
score = model1.evaluate(X_adv_fgsm, y_test)
print('\nloss: {0:.4f} acc: {1:.4f}'.format(score[0], score[1]))

print('\nTesting against JSMA adversarial')
score = model1.evaluate(X_adv_jsma, y_test)
print('\nloss: {0:.4f} acc: {1:.4f}'.format(score[0], score[1]))



print('\nPreparing predictions')
y0_0 = model0.predict(X_test)
y0_1_fgsm = model0.predict(X_adv_fgsm)
y0_1_jsma = model0.predict(X_adv_jsma)

y1_0 = model1.predict(X_test)
y1_1_fgsm = model1.predict(X_adv_fgsm)
y1_1_jsma = model1.predict(X_adv_jsma)

z_test = np.argmax(y_test, axis=1)

z0_0 = np.argmax(y0_0, axis=1)
z0_1_fgsm = np.argmax(y0_1_fgsm, axis=1)
z0_1_jsma = np.argmax(y0_1_jsma, axis=1)

z1_0 = np.argmax(y1_0, axis=1)
z1_1_fgsm = np.argmax(y1_1_fgsm, axis=1)
z1_1_jsma = np.argmax(y1_1_jsma, axis=1)

p0_0 = np.max(y0_0, axis=1)
p0_1_fgsm = np.max(y0_1_fgsm, axis=1)
p0_1_jsma = np.max(y0_1_jsma, axis=1)

p1_0 = np.max(y1_0, axis=1)
p1_1_fgsm = np.max(y1_1_fgsm, axis=1)
p1_1_jsma = np.max(y1_1_jsma, axis=1)

img_rows = 41
img_cols = 41
img_chan = 4

p_filter = np.all([p0_0>0.5,
                   p0_1_fgsm>0.5, p0_1_jsma>0.5,
                   p1_1_fgsm>0.5, p1_1_jsma>0.5],
                  axis=0)

print('\nGenerating figure')
fig = plt.figure(figsize=(40, 5))
gs = gridspec.GridSpec(1, 10, wspace=0.15)

for label in range(10):
    print('Label {0}'.format(label))
    gs0 = gridspec.GridSpecFromSubplotSpec(4, 3,
                                           subplot_spec=gs[label],
                                           wspace=0.05, hspace=0.1)
    ax = fig.add_subplot(gs0[:3, :])
    ax0 = fig.add_subplot(gs0[3, 0])
    ax1 = fig.add_subplot(gs0[3, 1])
    ax2 = fig.add_subplot(gs0[3, 2])

    img = np.empty((img_rows, img_cols, img_chan))
    ind, = np.where(np.all([p_filter,
                            z_test==label, z0_0==label,
                            z0_1_fgsm!=label, z0_1_jsma!=label,
                            z1_1_fgsm==label, z1_1_jsma!=label],
                           axis=0))

    cur = np.random.choice(ind)
    X_i = np.squeeze(X_test[cur])
    X_fgsm_i = np.squeeze(X_adv_fgsm[cur])
    X_jsma_i = np.squeeze(X_adv_jsma[cur])

    i = X_fgsm_i.flatten() - X_i.flatten()
    v = X_jsma_i.flatten() - X_i.flatten()
    (a, i), (b, j) = decompose(v, i)

    D = np.amax([1.5 * np.amax(np.absolute([a, b])),
                 0.5 / np.linalg.norm(np.absolute(i), ord=np.inf)])

    eps_i = np.linspace(-D, D, img_cols)
    eps_j = np.linspace(D, -D, img_rows)

    for r, ej in enumerate(eps_j):
        for c, ei in enumerate(eps_i):

            X_tmp = np.clip(X_i.flatten()+ej*j+ei*i, 0, 1)
            X_tmp = np.reshape(X_tmp, (1, 28, 28, 1))
            y0_tmp = model0.predict(X_tmp)
            z0_tmp = np.argmax(y0_tmp)
            y1_tmp = model1.predict(X_tmp)
            z1_tmp = np.argmax(y1_tmp)

            if z0_tmp == label and z1_tmp == label:
                # correct prediction in both cases
                color = [1, 1, 1, 1]
            elif z0_tmp == label:
                # correct prediction after normal training
                color = [1, 0, 0, 0.1]
            elif z1_tmp == label:
                # correct prediction after adv training
                color = [0, 1, 0, 0.1]
            else:
                # incorrect prediction in both cases
                color = [0.1, 0.1, 0.1, 0.1]
            img[r, c] = color

    # the original datum
    r = img_rows // 2
    c = img_rows // 2
    img[r, c] = [0, 0, 0, 1]

    # fgsm datum
    r = img_rows // 2
    c = int((np.linalg.norm(X_fgsm_i-X_i)-eps_i[0]) / (2*D) *
            img_cols)
    img[r, c] = [1.0, 0.65, 0, 1]

    # jsma datum
    r = int((a-eps_i[0])/(2*D) * img_cols)
    c = int((b-eps_j[-1])/(2*D) * img_rows)
    img[r, c] = [0, 0, 1, 0.6]

    ax.imshow(img, interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])

    ax0.imshow(X_i, cmap='gray', interpolation='none')
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1.imshow(X_fgsm_i, cmap='gray', interpolation='none')
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.imshow(X_jsma_i, cmap='gray', interpolation='none')
    ax2.set_xticks([])
    ax2.set_yticks([])


gs.tight_layout(fig, pad=0)
os.makedirs('img', exist_ok=True)
plt.savefig('img/figure_2.pdf')
