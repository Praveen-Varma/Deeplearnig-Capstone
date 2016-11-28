## Load all dependencies
import numpy as np
import tensorflow as tf
import cPickle as pickle
import matplotlib.pyplot as plt
import sys

## Load the pretraining dataset to be trained upon
with open('pretrain.pickle', 'r') as f:
    dataset = pickle.load(f)
    train_dataset = dataset['train']['data']
    test_dataset = dataset['test']['data']
    extra_dataset = dataset['extra']['data']
    train_dataset = np.append(train_dataset, test_dataset)
    train_dataset = np.append(train_dataset ,extra_dataset).reshape(-1, 1024)

## Function to get weights  
def new_weights(shape):
    return tf.get_variable("CONV_WEIGHTS", shape=shape)

## Function to get biases
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]), name = 'CONV_BIAS')

## Function creates a convolution layer
def new_conv_layer(input, weights, biases):
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 2, 2, 1], padding = 'SAME')
    layer += biases
    layer = tf.nn.local_response_normalization(layer)
    layer = tf.nn.relu(layer)
    return layer

## Function creates deconvolution layer
def new_deconv_layer(input, shape_out, weights, biases):
    input -= biases
    layer = tf.nn.conv2d_transpose(value=input, filter=weights, output_shape = shape_out,strides=[1, 2, 2, 1], padding = 'SAME')
    layer = tf.nn.local_response_normalization(layer)
    layer = tf.nn.relu(layer)
    return layer

image_size = 32
input_channels = 1
batch_size = 64

## Hyper parameters
filter_size1 = 5##int(sys.argv[1])
filter_size2 = 5##int(sys.argv[3])
filter_size3 = 4##int(sys.argv[5])
filter_size4 = 3##int(sys.argv[7])
n_filters1 = 20##int(sys.argv[2])
n_filters2 = 40##int(sys.argv[4])
n_filters3 = 80##int(sys.argv[6])
n_filters4 = 140##int(sys.argv[8])
batch_size = 64
parameters_conv = []
parameters_conv.append({'filter_size': filter_size1, 'n_filters':n_filters1, 'n_in_filters': 1})
parameters_conv.append({'filter_size': filter_size2, 'n_filters':n_filters2, 'n_in_filters':n_filters1})
parameters_conv.append({'filter_size': filter_size3, 'n_filters':n_filters3, 'n_in_filters':n_filters2})
parameters_conv.append({'filter_size': filter_size4, 'n_filters':n_filters4, 'n_in_filters':n_filters3})
wbs = []
shapes = []

## Building the CAE graph
graph = tf.Graph()
with graph.as_default():
    x =  tf.placeholder(tf.float32, shape=(batch_size, 1024))
    input = tf.reshape(x, [-1, 32, 32, 1])

## Initialize weights
    for index, params in enumerate(parameters_conv):
        with tf.variable_scope('indices' + str(index)) as scope:
            conv_weights = new_weights([params['filter_size'], params['filter_size'], params['n_in_filters'], params['n_filters']])
            conv_biases = new_biases(length = params['n_filters'])
            wbs.append({'weights': conv_weights, 'biases': conv_biases})

## Encoding step
    for index, params in enumerate(wbs):
        shapes.append(input.get_shape().as_list())
        with tf.variable_scope('conv' + str(index)) as scope:
            conv = new_conv_layer(input, params['weights'], params['biases'])
            input = conv

## Reverse the weights array
    wbs.reverse()
    shapes.reverse()

## Decoding step
    for index, params in enumerate(wbs):
        with tf.variable_scope('deconv' + str(index)) as scope:
            deconv = new_deconv_layer(input, shapes[index], params['weights'], params['biases'])
            input = deconv

## Reshape the output
    y = tf.reshape(input, [-1, 1024])

    cost = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(x, y), 1))
    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    filesave = []
    n_epochs = 5001


    for step in range(n_epochs):
        offset = (step * batch_size) % (train_dataset.shape[0] - batch_size)
        epoch_x = train_dataset[offset:(offset + batch_size), :]
        feed_dict = {x : epoch_x}
        _, l = sess.run([optimizer, cost], feed_dict=feed_dict)
        if (step % 500 == 0):
            print "Cost after {} steps is {}".format(step, l)


    for wb in wbs:
        filesave.append({'weights':wb['weights'].eval(session = sess), 'biases': wb['biases'].eval(session = sess)})

## Pickle the weights to initialize with in CNN classifier
    with open('numpukt.pkl', 'w') as f:
        pickle.dump(filesave, f)
