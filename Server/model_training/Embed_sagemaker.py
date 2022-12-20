import argparse
import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import tensorflow_hub as hub

f = open("answers.txt", "r")
answers = f.readlines()
f.close()

answers_byte = []
for a in answers:
    a = a.encode('utf-8')
    answers_byte.append(a)

print(answers_byte[0])

f = open("answers_types.txt", "r")
answers_types = f.readlines()
f.close()

answers_types_int = []
for a_type in answers_types:
    a_type = int(a_type)
    answers_types_int.append(a_type)

print(answers_types_int[:10])

# split into train and test sets
train_size = int(len(answers_byte) * 0.67)
test_size = len(answers_byte) - train_size

trainX, testX = answers_byte[0:train_size], answers_byte[train_size:len(answers_byte)]
print(len(trainX), len(testX))
print(len(answers_byte))

trainY, testY = answers_types_int[0:train_size], answers_types_int[train_size:len(answers_types_int)]
print(len(trainY), len(testY))
print(len(answers_types_int))

import os
train_dir = os.path.join(os.getcwd(), 'train')
os.makedirs(train_dir, exist_ok=True)

test_dir = os.path.join(os.getcwd(), 'test')
os.makedirs(test_dir, exist_ok=True)

np.save(os.path.join(train_dir, 'x_train.npy'), trainX)
np.save(os.path.join(train_dir, 'y_train.npy'), trainY)
np.save(os.path.join(test_dir, 'x_test.npy'), testX)
np.save(os.path.join(test_dir, 'y_test.npy'), testY)

import tensorflow as tf
import numpy as np

answers_tensor = tf.constant(np.array(answers_byte), tf.string, [len(answers_byte)], 'Const')
print(answers_tensor)

answers_types_tensor = tf.constant(np.array(answers_types_int), tf.int64, [len(answers_types_int)], 'Const')
print(answers_types_tensor)

epochs = 10
batch_size = 128
learning_rate = 0.01

def get_train_data():
    x_train = np.load('train/x_train.npy')
    y_train = np.load('train/y_train.npy')
    print('x train', x_train.shape, 'y train', y_train.shape)

    return x_train, y_train


def get_test_data():
    x_test = np.load('test/x_test.npy')
    y_test = np.load('test/y_test.npy')
    print('x test', x_test.shape, 'y test', y_test.shape)

    return x_test, y_test


x_train, y_train = get_train_data()
x_test, y_test = get_test_data()

embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"#gnews-swivel-20dim/1#nnlm-en-dim50/1
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Confirmation
model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

#model = tf.keras.models.load_model('saved_model')
arr = np.array([x_train[0]])
print(arr.shape)

trainPredict = model.predict(arr)
print(trainPredict[0][0])
print(type(trainPredict[0][0]))
print(trainPredict[0][0] >= 0)
testPredict = model.predict(x_test)
print(trainPredict, testPredict)

model.save('yes_no_model')


results = model.evaluate(x_test, verbose=2)

for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))
