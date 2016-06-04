
# coding: utf-8

# Deep Learning
# =============
# 
# Assignment 3
# ------------
# 
# Previously in `2_fullyconnected.ipynb`, you trained a logistic regression and a neural network model.
# 
# The goal of this assignment is to explore regularization techniques.

# In[ ]:

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import os

os.chdir('/home/sud/Downloads')
print(os.getcwd())

# First reload the data we generated in _notmist.ipynb_.

# In[ ]:


pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)


# Reformat into a shape that's more adapted to the models we're going to train:
# - data as a flat matrix,
# - labels as float 1-hot encodings.

# In[ ]:

image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


# In[ ]:

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


# ---
# Problem 1
# ---------
# 
# Introduce and tune L2 regularization for both logistic and neural network models. 
# Remember that L2 amounts to adding a penalty on the norm of the weights to the loss.
#  In TensorFlow, you can compute the L2 loss for a tensor `t` using `nn.l2_loss(t)`. 
#  The right amount of regularization should improve your validation / test accuracy.
# 
# ---
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

"""
## stochastic gradience descent

batch_size = 2000

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  weights = tf.Variable(
    tf.truncated_normal([image_size * image_size, num_labels]))
  biases = tf.Variable(tf.zeros([num_labels]))
  
  beta = 5e-5
  reg = tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
  # Training computation.
  logits = tf.matmul(tf_train_dataset, weights) + biases
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  loss = loss + beta*reg
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
    tf.matmul(tf_valid_dataset, weights) + biases)
  test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)


# Let's run it:

# In[ ]:

num_steps = 3001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


batch_size = 2000
hidden_nodes = 1024
import math

graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape = (batch_size, image_size*image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, num_labels))
    tf_test_dataset = tf.constant(test_dataset)
    tf_valid_dataset = tf.constant(valid_dataset)
    
    weight1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_nodes], stddev= math.sqrt(2.0/ (image_size * image_size))))
    biases1 = tf.Variable(tf.zeros([hidden_nodes]))
    
    weight2 = tf.Variable(tf.truncated_normal([hidden_nodes, num_labels]))
    biases2 = tf.Variable(tf.zeros([num_labels]))
    
    beta = 5e-5
    reg = tf.nn.l2_loss(weight1) + tf.nn.l2_loss(biases1) +tf.nn.l2_loss(weight2) + tf.nn.l2_loss(biases2)

    def getLogits(data):
        logits = tf.nn.relu(tf.matmul(data, weight1)+biases1)
        return tf.matmul(logits,weight2)+biases2
    
    logits = getLogits(tf_train_dataset)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    # predictions
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(getLogits(tf_test_dataset))
    valid_prediction = tf.nn.softmax(getLogits(tf_valid_dataset))
    
num_steps = 3001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

# ---
# Problem 2
# ---------
# Let's demonstrate an extreme case of overfitting. Restrict your training data to just a few batches. What happens?
# 
# ---
"""
# ---
# Problem 3
# ---------
# Introduce Dropout on the hidden layer of the neural network. Remember: Dropout should only be introduced during training, not evaluation, otherwise your evaluation results would be stochastic as well. TensorFlow provides `nn.dropout()` for that, but you have to make sure it's only inserted during training.
# 
# What happens to our extreme overfitting case?
# 
# ---

batch_size = 128
hidden_layer1_size = 1024
hidden_layer2_size = 300
hidden_layer3_size = 70

multilayer = True
regularizatio_meta = 0.03
graph = tf.Graph()

with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape = (batch_size,image_size*image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_test_dataset = tf.constant(test_dataset)
    tf_validate_dataset = tf.constant(valid_dataset)
    
    keep_prob  = tf.placeholder(tf.float32)
    weights_layer1 = tf.Variable(
    tf.truncated_normal([image_size * image_size, hidden_layer1_size], stddev=0.0517))
    biases_layer1 = tf.Variable(tf.zeros([hidden_layer1_size]))
    
    if multilayer:
         weights_layer2 = tf.Variable(
          tf.truncated_normal([hidden_layer1_size, hidden_layer1_size], stddev=0.0441))
         biases_layer2 = tf.Variable(tf.zeros([hidden_layer1_size]))
    
         weights_layer3 = tf.Variable(
          tf.truncated_normal([hidden_layer1_size, hidden_layer2_size], stddev=0.0441))
         biases_layer3 = tf.Variable(tf.zeros([hidden_layer2_size]))
        
         weights_layer4 = tf.Variable(
          tf.truncated_normal([hidden_layer2_size, hidden_layer3_size], stddev=0.0809))
         biases_layer4 = tf.Variable(tf.zeros([hidden_layer3_size]))
    weights = tf.Variable(
    tf.truncated_normal([hidden_layer3_size if multilayer else hidden_layer1_size, num_labels], stddev=0.1632))
    biases = tf.Variable(tf.zeros([num_labels]))
     # get the NN models
    def getNN4Layer(dSet, use_dropout):
        input_to_layer1 = tf.matmul(dSet, weights_layer1) + biases_layer1
        hidden_layer1_output = tf.nn.relu(input_to_layer1)
        
        
        logits_hidden1 = None
        if use_dropout:
           dropout_hidden1 = tf.nn.dropout(hidden_layer1_output, keep_prob)
           logits_hidden1 = tf.matmul(dropout_hidden1, weights_layer2) + biases_layer2
        else:
          logits_hidden1 = tf.matmul(hidden_layer1_output, weights_layer2) + biases_layer2
        
        hidden_layer2_output = tf.nn.relu(logits_hidden1)
        
        logits_hidden2 = None
        if use_dropout:
           dropout_hidden2 = tf.nn.dropout(hidden_layer2_output, keep_prob)
           logits_hidden2 = tf.matmul(dropout_hidden2, weights_layer3) + biases_layer3
        else:
          logits_hidden2 = tf.matmul(hidden_layer2_output, weights_layer3) + biases_layer3
        
        
        hidden_layer3_output = tf.nn.relu(logits_hidden2)
        logits_hidden3 = None
        if use_dropout:
           dropout_hidden3 = tf.nn.dropout(hidden_layer3_output, keep_prob)
           logits_hidden3 = tf.matmul(dropout_hidden3, weights_layer4) + biases_layer4
        else:
          logits_hidden3 = tf.matmul(hidden_layer3_output, weights_layer4) + biases_layer4
        
        
        hidden_layer4_output = tf.nn.relu(logits_hidden3)
        logits = None
        if use_dropout:
           dropout_hidden4 = tf.nn.dropout(hidden_layer4_output, keep_prob)
           logits = tf.matmul(dropout_hidden4, weights) + biases
        else:
          logits = tf.matmul(hidden_layer4_output, weights) + biases
        
        return logits
    # Training computation.
    logits = getNN4Layer(tf_train_dataset, True)  
    logits_valid = getNN4Layer(tf_validate_dataset, False)
    logits_test = getNN4Layer(tf_test_dataset, False)
        
      
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
      #loss_l2 = loss + (regularization_meta * (tf.nn.l2_loss(weights)))
      
    global_step = tf.Variable(0)  # count the number of steps taken.
    learning_rate = tf.train.exponential_decay(0.3, global_step, 3500, 0.86, staircase=True)
      
        
      # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
      
      # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(logits_valid)
    test_prediction = tf.nn.softmax(logits_test)
    
num_steps = 95001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in xrange(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob:0.75}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(train_prediction.eval(feed_dict={tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob:1.0}), batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(feed_dict={keep_prob:1.0}), valid_labels))
      print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(feed_dict={keep_prob:1.0}), test_labels))
# ---
# Problem 4
# ---------
# 
# Try to get the best performance you can using a multi-layer model! The best reported test accuracy using a deep network is [97.1%](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html?showComment=1391023266211#c8758720086795711595).
# 
# One avenue you can explore is to add multiple layers.
# 
# Another one is to use learning rate decay:
# 
#     global_step = tf.Variable(0)  # count the number of steps taken.
#     learning_rate = tf.train.exponential_decay(0.5, global_step, ...)
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
#  
#  ---
# 
