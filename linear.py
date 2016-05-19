# Boston House Price Linear Model
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_boston

boston = load_boston()
print(boston.data.shape)
# print(boston.keys())
# print boston.feature_names
# print boston.data

trX = boston.data
# trY = boston.target
X = tf.placeholder("float")
# Y = tf.placeholder("float")
# print trX//

def model(X,w):
    return tf.mul(X,w)

w = tf.Variable(tf.zeros([506,13]))
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
we = sess.run(w)
print we
# tf.Print(w,'p')
# y_model = model(X,w)
# print y_model
# cost = (tf.pow())
# print w
# print(boston)

