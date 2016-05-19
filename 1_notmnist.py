
# coding: utf-8

# Deep Learning
# =============
# 
# Assignment 1
# ------------
# 
# The objective of this assignment is to learn about simple data curation practices, and familiarize you with some of the data we'll be reusing later.
# 
# This notebook uses the [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) dataset to be used with python experiments. This dataset is designed to look like the classic [MNIST](http://yann.lecun.com/exdb/mnist/) dataset, while looking a little more like real data: it's a harder task, and the data is a lot less 'clean' than MNIST.

# In[6]:

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle


# First, we'll download the dataset to our local machine. The data consists of characters rendered in a variety of fonts on a 28x28 image. The labels are limited to 'A' through 'J' (10 classes). The training set has about 500k and the testset 19000 labelled examples. Given these sizes, it should be possible to train models quickly on any machine.

# In[7]:

url = 'http://commondatastorage.googleapis.com/books1000/'

def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  if force or not os.path.exists(filename):
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)
print('Done')


# Extract the dataset from the compressed .tar.gz file.
# This should give you a set of directories, labelled A through J.

# In[8]:

num_classes = 10
np.random.seed(133)

def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall()
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders
  
train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)


# ---
# Problem 1
# ---------
# 
# Let's take a peek at some of the data to make sure it looks sensible. Each exemplar should be an image of a character A through J rendered in a different font. Display a sample of the images that we just downloaded. Hint: you can use the package IPython.display.
# 
# ---

# In[10]:

os.getcwd()
# for file in os.listdir('/notebooks/notMNIST_large/A'):
#     print(file)
#     print(os.path.abspath(file))
#     display(Image(filename=os.path.abspath(file)))
#     break
folder = '/notebooks/notMNIST_large/A'
for image in os.listdir(folder):
    image_file = os.path.join(folder, image)
    display(Image(filename=image_file))
    break


# In[ ]:

Now let's load the data in a more manageable format. Since, depending on your computer setup you might not be able to fit it all in memory, we'll load each class into a separate dataset, store them on disk and curate them independently. Later we'll merge them into a single dataset of manageable size.

We'll convert the entire dataset into a 3D array (image index, x, y) of floating point values, normalized to have approximately zero mean and standard deviation ~0.5 to make training easier down the road. 

A few images might not be readable, we'll just skip them.


# In[11]:

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  image_index = 0
  print(folder)
  for image in os.listdir(folder):
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[image_index, :, :] = image_data
      image_index += 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  num_images = image_index
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)


# ---
# Problem 2
# ---------
# 
# Let's verify that the data still looks good. Displaying a sample of the labels and images from the ndarray. Hint: you can use matplotlib.pyplot.
# 
# ---

# In[23]:

import random
get_ipython().magic(u'matplotlib inline')
"""
def showLabelAndImage(dataset, labels, n):
    indices = random.sample(range(0,labels.shape[0]),n)
    fig = plt.figure()
    for i in range(n):
      a = fig.add_subplot(1,n,i+1)
      plt.imshow(dataset[indices[i],:,:])
      a.set_title(chr(labels[indices[i]]+ord('A')))
      a.axes.get_xaxis().set_visible(False)
      a.axes.get_yaxis().set_visible(False)
    plt.show()

showLabelAndImage(train_datasets,train_labels,5)
"""
def showProcessedRandom(dataset_name,labels,n): 
    with open(dataset_name, 'rb') as f:
        dataset = pickle.load(f)
        indices=np.random.choice(dataset.shape[0], n)
        fig=plt.figure()
        for i in range(n):
            a=fig.add_subplot(1,n,i+1)
            plt.imshow(dataset[indices[i],:,:])
            a.set_title(chr(labels[indices[i]]+ord('A')))
            a.axes.get_xaxis().set_visible(False)
            a.axes.get_yaxis().set_visible(False)
        plt.show()
    f.close()

for name in train_datasets:
    showProcessedRandom(name,train_labels,5) 
for name in test_datasets:
    showProcessedRandom(name,test_labels,5)


# In[16]:

print(train_datasets)


# ---
# Problem 3
# ---------
# Another check: we expect the data to be balanced across classes. Verify that.
# 
# ---

# Merge and prune the training data as needed. Depending on your computer setup, you might not be able to fit it all in memory, and you can tune `train_size` as needed. The labels will be stored into a separate array of integers 0 through 9.
# 
# Also create a validation dataset for hyperparameter tuning.

# In[28]:

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
                    
        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels
            
            
train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)


# In[29]:

from collections import Counter
Counter(train_labels), Counter(test_labels)
plt.hist(train_labels,bins =20)


# Next, we'll randomize the data. It's important to have the labels well shuffled for the training and test distributions to match.

# In[31]:

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)


# In[32]:

from collections import Counter
Counter(train_labels), Counter(test_labels)
plt.hist(train_labels,bins =20)


# ---
# Problem 4
# ---------
# Convince yourself that the data is still good after shuffling!
# 
# ---

# In[ ]:

from collections import Counter
Counter(train_labels), Counter(test_labels)
plt.hist(train_labels,bins =20)


# Finally, let's save the data for later reuse:

# In[33]:

pickle_file = 'notMNIST.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise


# In[34]:

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)


# ---
# Problem 5
# ---------
# 
# By construction, this dataset might contain a lot of overlapping samples, including training data that's also contained in the validation and test set! Overlap between training and test can skew the results if you expect to use your model in an environment where there is never an overlap, but are actually ok if you expect to see training samples recur when you use it.
# Measure how much overlap there is between training, validation and test samples.
# 
# Optional questions:
# - What about near duplicates between datasets? (images that are almost identical)
# - Create a sanitized validation and test set, and compare your accuracy on those in subsequent assignments.
# ---

# In[42]:

def retrieveDataSet(filename):
    try:
        with open(filename, 'rb') as f:
            dataset_dict = pickle.load(f)
            train_dataset = dataset_dict['train_dataset']
            valid_dataset = dataset_dict['valid_dataset']
            test_dataset = dataset_dict['test_dataset']
        f.close()
    except Exception as e:
        print(str(e))
    return train_dataset,valid_dataset,test_dataset

pickle_file = 'notMNIST.pickle'
train_dataset,valid_dataset,test_dataset = retrieveDataSet(pickle_file)

import hashlib as hlib
import time

t1 = time.time()
train_hashes = [hlib.md5(x).digest() for x in train_dataset]
valid_hashes = [hlib.md5(x).digest() for x in valid_dataset]
test_hashes = [hlib.md5(x).digest() for x in test_dataset]

train_hashes1 = [hlib.sha1(x).digest() for x in train_dataset]
valid_hashes1 = [hlib.sha1(x).digest() for x in valid_dataset]
test_hashes1 = [hlib.sha1(x).digest() for x in test_dataset]

# check whether one dataset is present in another

validInTrain = np.in1d(valid_hashes,train_hashes)
testInTrain = np.in1d(test_hashes, train_hashes)
testInValid = np.in1d(test_hashes, valid_hashes)

validInTrain1 = np.in1d(valid_hashes1,train_hashes1)
testInTrain1 = np.in1d(test_hashes1, train_hashes1)
testInValid1 = np.in1d(test_hashes1, valid_hashes1)
print('Using MD5')
print(validInTrain.sum())
print(testInTrain.sum())
print(testInTrain.sum())
print('Using SHA1')
print(validInTrain1.sum())
print(testInTrain1.sum())
print(testInTrain1.sum())


# ---
# Problem 6
# ---------
# 
# Let's get an idea of what an off-the-shelf classifier can give you on this data. It's always good to check that there is something to learn, and that it's a problem that is not so trivial that a canned solution solves it.
# 
# Train a simple model on this data using 50, 100, 1000 and 5000 training samples. Hint: you can use the LogisticRegression model from sklearn.linear_model.
# 
# Optional question: train an off-the-shelf model on all the data!
# 
# ---

# In[72]:

from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
def retrieveDataSet(filename):
    try:
        with open(filename, 'rb') as f:
            dataset_dict = pickle.load(f)
            train_dataset = dataset_dict['train_dataset']
            valid_dataset = dataset_dict['valid_dataset']
            test_dataset = dataset_dict['test_dataset']
            train_labels = dataset_dict['train_labels']
            valid_labels = dataset_dict['valid_labels']
            test_labels = dataset_dict['test_labels']
        f.close()
    except Exception as e:
        print(str(e))
    return train_dataset,valid_dataset,test_dataset,train_labels,valid_labels,test_labels

pickle_file = 'notMNIST.pickle'
train_dataset,valid_dataset,test_dataset,train_labels,valid_labels,test_labels = retrieveDataSet(pickle_file)
# print(train_labels)
clf = LogisticRegression()
nsamples, nx, ny = train_dataset.shape
# print(nsamples, nx, ny)
d2_train_dataset = train_dataset.reshape((nsamples,nx*ny))

# test Data
nsamples, nx, ny = test_dataset.shape
# print(nsamples, nx, ny)
d2_test_dataset = test_dataset.reshape((nsamples,nx*ny))
# nsamples, nx, ny = test_dataset.shape
# print(nsamples, nx, ny)
# (samples, width, height) = train_dataset.shape
# X = np.reshape(train_dataset,(samples, width*height))[0:1000]
# y = train_labels[0:1000]
# X = d2_train_dataset[0:1000]
# y = train_labels[0:1000]
# (samples, width, height) = test_dataset.shape
# X_test = np.reshape(test_dataset, (samples, width*height))
# y_test = test_labels
# clf.fit(X, y)
# print(clf.score(X_test, y_test))
scores = []
for x in [100,2000,10000,20000]:
    X = d2_train_dataset[0:x]
    y = train_labels[0:x]
    clf.fit(X,y)
    accuracy = clf.score(d2_test_dataset,test_labels)
    scores.append(accuracy)
    print(accuracy)
plt.plot(scores)
plt.show()

print('Done')


# In[ ]:



