import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

class Network(object):
    def __init__(self, graph):
        try:
            with open('tensorflow_data.pickle', 'rb') as f:
                dataset = pickle.load(f)
                self.train_dataset = dataset['train']['data'].reshape(-1, 32, 32, 1)
                self.test_dataset = dataset['test']['data'].reshape(-1, 32, 32, 1)
                self.valid_dataset = dataset['valid']['data'].reshape(-1, 32, 32, 1)
                self.extra_dataset = dataset['extra']['data'].reshape(-1, 32, 32, 1)
                self.train_labels = dataset['train']['label']
                self.test_labels = dataset['test']['label']
                self.valid_labels = dataset['valid']['label']
                self.extra_labels = dataset['extra']['label']
        except Exception as e:
            print('Unable to process data from dataset.pickle', ':', e)
            raise
        self.image_size = 32
        self.n_input_channels = 1
        self.batch_size = 64
        filter_size1 = 5
        filter_size2 = 5
        filter_size3 = 5
        filter_size4 = 4
        n_filters1 = 16
        n_filters2 = 32
        n_filters3 = 64
        n_filters4 = 128
        outputs_fc1 = 32
        outputs_fc2 = 11
        n_claf = 5
        parameters_conv = []
        self.conv = []
        self.claf = []
        parameters_conv.append({'filter_size': filter_size1, 'n_filters':n_filters1, 'n_in_filters':self.n_input_channels})
        parameters_conv.append({'filter_size': filter_size2, 'n_filters':n_filters2, 'n_in_filters':n_filters1})
        parameters_conv.append({'filter_size': filter_size3, 'n_filters':n_filters3, 'n_in_filters':n_filters2})
        parameters_conv.append({'filter_size': filter_size4, 'n_filters':n_filters4, 'n_in_filters':n_filters3})
        for index, params in enumerate(parameters_conv):
            with tf.variable_scope('conv' + str(index)) as scope:
                conv_weights = self.new_weights([params['filter_size'], params['filter_size'], params['n_in_filters'], params['n_filters']])
                conv_biases = self.new_biases(length = params['n_filters'])
            self.conv.append({'weights': conv_weights, 'biases': conv_biases})
        for index in range(5):
            with tf.variable_scope('claf'+ str(index)) as scope:
                claf_hidden_weights = tf.get_variable( 'h_weights', shape=[512, outputs_fc1],initializer=tf.contrib.layers.xavier_initializer())
                claf_hidden_biases = tf.get_variable('h_biases', initializer=tf.constant(1.0, shape=[outputs_fc1]))
                claf_out_weights = tf.get_variable( 'o_weights', shape=[outputs_fc1, outputs_fc2],initializer=tf.contrib.layers.xavier_initializer())
                claf_out_biases = tf.get_variable('o_biases', initializer=tf.constant(1.0, shape=[outputs_fc2])) 
            self.claf.append({'h_weights': claf_hidden_weights, 'h_biases': claf_hidden_biases, 'o_weights': claf_out_weights, 'o_biases': claf_out_biases})
    def model(self, data, keep_prob):
        input = data
        logits = []
        #input = self.lcn(input)
        for index, params in enumerate(self.conv):
            with tf.variable_scope('conv' + str(index)) as scope:
                conv = self.new_conv_layer(input, params['weights'], params['biases'])
            input = conv
        input = tf.nn.dropout(input, keep_prob)
        shape = input.get_shape().as_list()
        n_inputs = shape[1] * shape[2] * shape[3]
        layer_flat = tf.reshape(input, [shape[0], n_inputs])
        for index, params in enumerate(self.claf):
            with tf.variable_scope('claf' + str(index)) as scope:
                fc = self.new_fc_layer(layer_flat, params['h_weights'], params['h_biases'], use_relu= True )
                logit = self.new_fc_layer(fc, params['o_weights'], params['o_biases'], use_relu= False)
            logits.append(logit)
        return logits
    def gaussian_filter(self, kernel_shape):
        x = np.zeros(kernel_shape, dtype = float)
        mid = np.floor(kernel_shape[0] / 2.)
        for kernel_idx in xrange(0, kernel_shape[2]):
            for i in xrange(0, kernel_shape[0]):
                for j in xrange(0, kernel_shape[1]):
                    x[i, j, kernel_idx, 0] = self.gauss(i - mid, j - mid)
    
        return tf.convert_to_tensor(x / np.sum(x), dtype=tf.float32)
    def gauss(self, x, y, sigma=3.0):
        Z = 2 * np.pi * sigma ** 2
        return  1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

    def lcn(self, X):
        radius=7
        threshold=1e-4
        filter_shape = (radius, radius, 1, 1)
        filters = self.gaussian_filter(filter_shape)
        convout = tf.nn.conv2d(X, filters, [1,1,1,1], 'SAME')
        mid = int(np.floor(filter_shape[1] / 2.))
        centered_X = tf.sub(X, convout)
        sum_sqr_XX = tf.nn.conv2d(tf.square(centered_X), filters, [1,1,1,1], 'SAME')
        denom = tf.sqrt(sum_sqr_XX)
        per_img_mean = tf.reduce_mean(denom)
        divisor = tf.maximum(per_img_mean, denom)
        new_X = tf.truediv(centered_X, tf.maximum(divisor, threshold))
        return new_X
    def new_conv_layer(self, input, weights, biases):
        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding = 'SAME')
        layer += biases
        layer = tf.nn.local_response_normalization(layer)
        layer = tf.nn.relu(layer)
        layer = tf.nn.max_pool(value = layer, ksize = [1, 2, 2, 1], strides = [1,2, 2, 1], padding = 'SAME')
        tf.image_summary('convultion3', layer[:, :, :, 0:1], 5)
        return layer
    def new_fc_layer(self, input, weights, biases,use_relu):
        layer = tf.matmul(input, weights) + biases
        if use_relu:
            layer = tf.nn.relu(layer)
        return layer
    def new_weights(self,shape):
        return tf.get_variable("CONV_WEIGHTS", shape=shape,initializer=tf.contrib.layers.xavier_initializer_conv2d())
    def new_biases(self,length):
        return tf.Variable(tf.constant(0.05, shape=[length]), name = 'CONV_BIAS')
    def predict(self, graph):
        x = tf.placeholder(tf.float32, shape=(10, self.image_size, self.image_size, self.n_input_channels))
        with tf.variable_scope('test_pred') as scope:
            logits = self.model(x, 1.0)
            test_prediction = tf.pack([tf.nn.softmax(logits[0]),\
                                       tf.nn.softmax(logits[1]),\
                                       tf.nn.softmax(logits[2]),\
                                       tf.nn.softmax(logits[3]),\
                                       tf.nn.softmax(logits[4])])
            test_prediction = tf.transpose(tf.argmax(test_prediction, 2))

        saver = tf.train.Saver()
        with tf.Session(graph=graph) as session:
            tf.initialize_all_variables().run()
            saver.restore(session, "SVHN_MODEL.ckpt")
            test_prediction = session.run(test_prediction, feed_dict={x: self.test_dataset[0:10,:,:,:],})
            print test_prediction
            print self.test_labels[0:10]
graph = tf.Graph()
with graph.as_default():
    network = Network(graph)
    network.predict(graph)
