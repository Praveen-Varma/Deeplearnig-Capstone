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
        
        self.writer = tf.train.SummaryWriter('log', graph=graph)
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
#       tf.image_summary('input', input, 5)
        input = self.lcn(input)
#        tf.image_summary('input_lcn', input, 5)
        for index, params in enumerate(self.conv):
            with tf.variable_scope('conv' + str(index)) as scope:
                conv = self.new_conv_layer(input, params['weights'], params['biases'])
#                tf.histogram_summary('conv' + str(index) + 'weights', params['weights'])
#                tf.image_summary('conv' + str(index) + 'im', conv[:, :, :, 0:1], 5)
            input = conv
        input = tf.nn.dropout(input, keep_prob)
        shape = input.get_shape().as_list()
        n_inputs = shape[1] * shape[2] * shape[3]
        layer_flat = tf.reshape(input, [shape[0], n_inputs])
        for index, params in enumerate(self.claf):
            with tf.variable_scope('claf' + str(index)) as scope:
                fc = self.new_fc_layer(layer_flat, params['h_weights'], params['h_biases'], use_relu= True )
                logit = self.new_fc_layer(fc, params['o_weights'], params['o_biases'], use_relu= False)
#                tf.histogram_summary('claf_h' + str(index) + 'weights', params['h_weights'])
#                tf.histogram_summary('claf_out' + str(index) + 'weights', params['o_weights'])
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
    def accuracy(self,predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 2).T == labels) / predictions.shape[1] / predictions.shape[0])
    def new_weights(self,shape):
        return tf.get_variable("CONV_WEIGHTS", shape=shape,initializer=tf.contrib.layers.xavier_initializer_conv2d())
    def new_biases(self,length):
        return tf.Variable(tf.constant(0.05, shape=[length]), name = 'CONV_BIAS')
    def train(self, graph):
        x = tf.placeholder(tf.float32, shape=(self.batch_size, self.image_size, self.image_size, self.n_input_channels))
        y = tf.placeholder(tf.int32, shape=(self.batch_size, 6))
        [logit1, logit2, logit3, logit4, logit5] = self.model(x, 0.92)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logit1, y[:,1]))
        loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logit2, y[:,2]))
        loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logit3, y[:,3]))    
        loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logit4, y[:,4]))
        loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logit5, y[:,5]))
        tf.scalar_summary('loss', loss)
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.95)
        optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)
        with tf.variable_scope('train_pred') as scope:
            train_preds = self.model(x, 1.0)
            train_prediction = tf.pack([tf.nn.softmax(train_preds[0]),\
                                        tf.nn.softmax(train_preds[1]),\
                                        tf.nn.softmax(train_preds[2]),\
                                        tf.nn.softmax(train_preds[3]),\
                                        tf.nn.softmax(train_preds[4])])
        with tf.variable_scope('valid_pred') as scope:
            valid_preds = self.model(self.valid_dataset, 1.0)
            valid_prediction = tf.pack([tf.nn.softmax(valid_preds[0]),\
                                        tf.nn.softmax(valid_preds[1]),\
                                        tf.nn.softmax(valid_preds[2]),\
                                        tf.nn.softmax(valid_preds[3]),\
                                        tf.nn.softmax(valid_preds[4])])
        with tf.variable_scope('test_pred') as scope:
            test_preds = self.model(self.test_dataset, 1.0)
            test_prediction = tf.pack([tf.nn.softmax(test_preds[0]),\
                                       tf.nn.softmax(test_preds[1]),\
                                       tf.nn.softmax(test_preds[2]),\
                                       tf.nn.softmax(test_preds[3]),\
                                       tf.nn.softmax(test_preds[4])])

        saver = tf.train.Saver()
        num_steps = 10000
        summary_op = tf.merge_all_summaries()
        with tf.Session(graph=graph) as session:
            tf.initialize_all_variables().run()
            for step in range(num_steps):
                offset = (step * self.batch_size) % (self.train_labels.shape[0] - self.batch_size)
                epoch_x = self.train_dataset[offset:(offset + self.batch_size), :, :, :]
                epoch_y = self.train_labels[offset:(offset + self.batch_size),:]
                feed_dict = {x : epoch_x, y : epoch_y}
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
                if (step % 500 == 0): 
                    print('Batch loss at step %d: %f' % (step, l))
                    print('Batch accuracy: %.1f%%' % self.accuracy(predictions, epoch_y[:,1:6]))
                    print('Validation accuracy: %.1f%%' % self.accuracy(valid_prediction.eval(), self.valid_labels[:,1:6]))
                    #self.writer.add_summary(summary, step)
                    
            ##print('Test accuracy: %.1f%%' % self.accuracy(test_prediction.eval(), self.test_labels[:,1:6]))
            save_path = saver.save(session, "SVHN_MODEL.ckpt")
graph = tf.Graph()
with graph.as_default():
    network = Network(graph)
    network.train(graph)
