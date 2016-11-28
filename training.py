## Load dependent libraries
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import sys
from utils import lcn, accuracy, acc

# Class for the CNN network 
class Network(object):
    def __init__(self, graph):
        try:
            # Import the training dataset
            with open('tensorflow_data.pickle', 'rb') as f:
                dataset = pickle.load(f)
                self.train_dataset = dataset['train']['data'].reshape(-1, 32, 32, 1).astype(np.float32)
                self.test_dataset = dataset['test']['data'].reshape(-1, 32, 32, 1).astype(np.float32)
                self.valid_dataset = dataset['valid']['data'].reshape(-1, 32, 32, 1).astype(np.float32)
                self.train_labels = dataset['train']['label']
                self.test_labels = dataset['test']['label']
                self.valid_labels = dataset['valid']['label']
        except Exception as e:
            print('Unable to process data from dataset.pickle', ':', e)
            raise
        
        self.image_size = 32
        self.n_input_channels = 1
        self.batch_size = 64
        
        ## Hyper parameters for the CNN
        filter_size1 = 5##int(sys.argv[1])
        filter_size2 = 5##int(sys.argv[3])
        filter_size3 = 4##int(sys.argv[5])
        filter_size4 = 3##int(sys.argv[7])
        n_filters1 = 20##int(sys.argv[2])
        n_filters2 = 40##int(sys.argv[4])
        n_filters3 = 80##int(sys.argv[6])
        n_filters4 = 140##int(sys.argv[8])
        outputs_fc1 = 128
        outputs_fc2 = 32
        outputs_fc3 = 11
        parameters_conv = []
        self.conv = []
        self.claf = []

        ## Store parameters in an array
        parameters_conv.append({'filter_size': filter_size1, 'n_filters':n_filters1, 'n_in_filters':self.n_input_channels})
        parameters_conv.append({'filter_size': filter_size2, 'n_filters':n_filters2, 'n_in_filters':n_filters1})
        parameters_conv.append({'filter_size': filter_size3, 'n_filters':n_filters3, 'n_in_filters':n_filters2})
        parameters_conv.append({'filter_size': filter_size4, 'n_filters':n_filters4, 'n_in_filters':n_filters3})

        ## load the pretrained weights
        numpukt = []
        with open('numpukt.pkl', 'r') as f:
            numpukt = pickle.load(f)

        ## weights are stored in reverse so arrange them again
        numpukt.reverse()

        ## Create weights and bias variables with values from CAE
        for i, v in enumerate(numpukt):
            with tf.variable_scope('indices' + str(i)) as scope:
                self.conv.append({'weights': tf.Variable(v['weights'], name = 'CONV_WEIGHTS'), 'biases': tf.Variable(v['biases'], name = 'CONV_BIAS')})

        ## Create weights and biases for  4 fully connected layers
        for index in range(4):  
            with tf.variable_scope('claf'+ str(index)) as scope:
                claf_hidden_weight1 = tf.get_variable( 'h1_weights', shape=[2 * 2 * n_filters4, outputs_fc1],initializer=tf.contrib.layers.xavier_initializer())
                claf_hidden_biases1 = tf.get_variable('h1_biases', initializer=tf.constant(1.0, shape=[outputs_fc1]))
                claf_hidden_weight2 = tf.get_variable( 'h2_weights', shape=[outputs_fc1, outputs_fc2],initializer=tf.contrib.layers.xavier_initializer())
                claf_hidden_biases2 = tf.get_variable('h2_biases', initializer=tf.constant(1.0, shape=[outputs_fc2]))
                claf_out_weights    = tf.get_variable( 'o_weights', shape=[outputs_fc2, outputs_fc3],initializer=tf.contrib.layers.xavier_initializer())
                claf_out_biases     = tf.get_variable('o_biases', initializer=tf.constant(1.0, shape=[outputs_fc3])) 
            self.claf.append({'h1_weights': claf_hidden_weight1, 'h1_biases': claf_hidden_biases1,
                              'h2_weights': claf_hidden_weight2, 'h2_biases': claf_hidden_biases2,
                              'o_weights': claf_out_weights, 'o_biases': claf_out_biases})

    ## Method to get logits
    def model(self, data, keep_prob):
        input = data
        logits = []
        # Perform local contrast normalization
        input = lcn(input)
        # Create Convolutional layers
        for index, params in enumerate(self.conv):
            with tf.variable_scope('conv' + str(index)) as scope:
                conv = self.new_conv_layer(input, params['weights'], params['biases'])
            input = conv
        input = tf.nn.dropout(input, keep_prob)
        shape = input.get_shape().as_list()
        n_inputs = shape[1] * shape[2] * shape[3]
        # flatten the final convolutional layer
        layer_flat = tf.reshape(input, [shape[0], n_inputs])
        regularizers = []
        # build fully connected layers
        for index, params in enumerate(self.claf):
            with tf.variable_scope('claf' + str(index)) as scope:
                fc = self.new_fc_layer(layer_flat, params['h1_weights'], params['h1_biases'], use_relu= True )
                fc = tf.nn.dropout(fc, 0.8)
                fc = self.new_fc_layer(fc, params['h2_weights'], params['h2_biases'], use_relu= True )
                fc = tf.nn.dropout(fc, 0.8)
                regularizers.append( tf.nn.l2_loss(params['h2_weights']) + tf.nn.l2_loss(params['h2_biases']))
                logit = self.new_fc_layer(fc, params['o_weights'], params['o_biases'], use_relu= False)
            logits.append(logit)
        return logits, regularizers

    ## method for covolutional layer
    def new_conv_layer(self, input, weights, biases):
        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 2, 2, 1], padding = 'SAME')
        layer += biases
        layer = tf.nn.local_response_normalization(layer)
        layer = tf.nn.relu(layer)
        return layer
    
    ## method for fully connected layer
    def new_fc_layer(self, input, weights, biases,use_relu):
        layer = tf.matmul(input, weights) + biases
        if use_relu:
            layer = tf.nn.relu(layer)
        return layer

    ## method for performing the network training
    def train(self, graph):

        ## input data placeholder for training data
        x = tf.placeholder(tf.float32, shape=(self.batch_size, self.image_size, self.image_size, self.n_input_channels))
        y = tf.placeholder(tf.int32, shape=(self.batch_size, 6))

        ## Get logits for training
        [logit1, logit2, logit3, logit4], regularizers = self.model(x, 0.68)

        
        l = [tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logit1, y[:,0])),
               tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logit2, y[:,1])),
               tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logit3, y[:,2])),
               tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logit4, y[:,3]))]
        loss = 0
        for i in range(0, 4):
            loss += l[i] + 4e-3 * regularizers[i]

        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(0.02, global_step, 10000, 0.95)
        optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)

        ## predictions
        with tf.variable_scope('train_pred') as scope:
            train_preds, _ = self.model(x, 1.0)
            train_prediction = tf.pack([tf.nn.softmax(train_preds[0]),\
                                        tf.nn.softmax(train_preds[1]),\
                                        tf.nn.softmax(train_preds[2]),\
                                       tf.nn.softmax(train_preds[3])])

        with tf.variable_scope('test_pred') as scope:
            test_preds, _ = self.model(self.test_dataset, 1.0)
            test_prediction = tf.pack([tf.nn.softmax(test_preds[0]),\
                                       tf.nn.softmax(test_preds[1]),\
                                       tf.nn.softmax(test_preds[2]),\
                                       tf.nn.softmax(test_preds[3])])

        with tf.variable_scope('test_pred') as scope:
            valid_preds, _ = self.model(self.valid_dataset, 1.0)
            valid_prediction = tf.pack([tf.nn.softmax(valid_preds[0]),\
                                       tf.nn.softmax(valid_preds[1]),\
                                       tf.nn.softmax(valid_preds[2]),\
                                       tf.nn.softmax(valid_preds[3])])

        saver = tf.train.Saver()
        num_steps = 20001

        losses = {}
        acc_p = {}
        acc_v = {}
        with tf.Session(graph=graph) as session:
            tf.initialize_all_variables().run()
            for step in range(num_steps):
                # mini batch logic
                offset = (step * self.batch_size) % (self.train_labels.shape[0] - self.batch_size)
                epoch_x = self.train_dataset[offset:(offset + self.batch_size), :, :, :]
                epoch_y = self.train_labels[offset:(offset + self.batch_size),:]
                feed_dict = {x : epoch_x, y : epoch_y}
                
                if (step % 500 == 0): 
                    _, l, p, v = session.run([optimizer, loss, train_prediction, valid_prediction], feed_dict=feed_dict)
                    acc_p[step] = acc(p, epoch_y[:,0:4])
                    print ('Batch accuracy at step %d: %.1f%%' % (step, acc_p[step]))
                    acc_v[step] = acc(v, self.valid_labels[:,0:4])
                    print ('Validation accuracy at step %d: %.1f%%' % (step, acc_v[step]))
                else:
                    _, l = session.run([optimizer, loss], feed_dict=feed_dict)
                losses[step] = l
            acc1 =  acc(test_prediction.eval(), self.test_labels[:,0:4])   
            
            print('Test accuracy: %.1f%%' % acc1)
            save_path = saver.save(session, "SVHN_MODEL.ckpt")
            return losses, acc_p, acc_v

## Set the graph        
graph = tf.Graph()
with graph.as_default():
    network = Network(graph)
    losses, acc_p, acc_v = network.train(graph)

## Pickling accuracy for Bayesian OPtimization
with open('plot.pkl', 'w') as f:
    pickle.dump([losses, acc_p, acc_v], f)
