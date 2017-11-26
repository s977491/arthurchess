'''
The neural network architecture is some mix of AlphaGo's input features and
Cazenave's resnet architecture.

See features.py for a list of input features used. All colors are flipped so
that it is always "black to play". Thus, the same policy is used to estimate
both white and black moves. However, in the case of the value network, the
value of komi, or whose turn to play, must also be passed in, because there
is an asymmetry between black and white there.

The policy and value networks share a majority of their architecture.
This helps the intermediate layers extract concepts that are relevant to both 
move prediction and score estimation.

Within the DNN, the layer width is configurable, but 128 is a good compromise
between network size and compute time. All layers use ReLu nonlinearities and
zero-padding for convolutions.

The policy and value networks can be evaluated independently or together;
if executed together, the shared part of the network only needs to be computed
once. When training, you must either alternate training both halves, or freeze
the shared part of the network, or else the half that isn't being trained will
start producing inaccurate outputs.
'''

import math
import os
import sys
import tensorflow as tf

import features
import cc
import utils
import numpy as np
EPSILON = 1e-35

class PolicyNetwork(object):
    temper = False
    def __init__(self, k=128, num_int_conv_layers=11, use_cpu=False):
        self.num_input_planes = sum(f.planes for f in features.DEFAULT_FEATURES)
        self.k = k
        self.num_int_conv_layers = num_int_conv_layers
        self.test_summary_writer = None
        self.training_summary_writer = None
        self.test_stats = StatisticsCollector()
        self.training_stats = StatisticsCollector()
        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = 8
        config.inter_op_parallelism_threads = 8
        self.session = tf.Session(config=config)
        self.weightList = []
        if use_cpu:
            with tf.device("/cpu:0"):
                self.set_up_network()
        else:
            self.set_up_network()

    def set_up_network(self):
        # a global_step variable allows epoch counts to persist through multiple training sessions
        global_step = tf.Variable(0, name="global_step", trainable=False)
        RL_global_step = tf.Variable(0, name="RL_global_step", trainable=False)
        x = tf.placeholder(tf.float32, [None, cc.Ny, cc.Nx, self.num_input_planes], name="x")
        yV = tf.placeholder(tf.float32, shape=[None, 1], name="yV")
        yPos = tf.placeholder(tf.float32, shape=[None, cc.Ny * cc.Nx * cc.Ny * cc.Nx], name="yPos")
        # whether this example should be positively or negatively reinforced.
        # Set to 1 for positive, -1 for negative.
        reinforce_direction = tf.placeholder(tf.float32, shape=[])

        #convenience functions for initializing weights and biases
        def _weight_variable(shape, name):
            # If shape is [5, 5, 20, 32], then each of the 32 output planes
            # has 5 * 5 * 20 inputs.
            # initial = tf.truncated_normal(shape, stddev=0.1)
            # return tf.Variable(initial, name=name)

            number_inputs_added = utils.product(shape[:-1])
            stddev = 1 / math.sqrt(number_inputs_added) / 50
            # http://neuralnetworksanddeeplearning.com/chap3.html#weight_initialization
            w = tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)
            self.weightList.append(w)
            return w

        def _conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

        # initial conv layer is 5x5
        W_conv_init55 = _weight_variable([5, 5, self.num_input_planes, self.k], name="W_conv_init55")
        W_conv_init11 = _weight_variable([1, 1, self.num_input_planes, self.k], name="W_conv_init11")
        h_conv_init = tf.nn.relu(_conv2d(x, W_conv_init55) + _conv2d(x, W_conv_init11), name="h_conv_init")

        # followed by a series of resnet 3x3 conv layers
        W_conv_intermediate = []
        h_conv_intermediate = []
        _current_h_conv = h_conv_init
        for i in range(self.num_int_conv_layers):
            with tf.name_scope("layer"+str(i)):
                _resnet_weights1 = _weight_variable([3, 3, self.k, self.k], name="W_conv_resnet1")
                _resnet_weights2 = _weight_variable([3, 3, self.k, self.k], name="W_conv_resnet2")
                _int_conv = tf.nn.relu(_conv2d(_current_h_conv, _resnet_weights1), name="h_conv_intermediate")
                _output_conv = tf.nn.relu(
                    _current_h_conv +
                    _conv2d(_int_conv, _resnet_weights2),
                    name="h_conv")
                W_conv_intermediate.extend([_resnet_weights1, _resnet_weights2])
                h_conv_intermediate.append(_output_conv)
                _current_h_conv = _output_conv
        W_conv_init11FinalPNet = _weight_variable([1, 1, self.k, self.k], name="W_conv_init11FinalPNet")
        b_conv_init11FinalPNet = tf.Variable(tf.constant(0, shape=[self.k], dtype=tf.float32), name="b_conv_init11FinalPNet")
        h_conv_initFinalPNet = _conv2d(h_conv_intermediate[-1], W_conv_init11FinalPNet) + b_conv_init11FinalPNet

        W_conv_init11FinalPNet2 = _weight_variable([1, 1, self.k, self.k], name="W_conv_init11FinalPNet2")
        b_conv_init11FinalPNet2 = tf.Variable(tf.constant(0, shape=[self.k], dtype=tf.float32),
                                             name="b_conv_init11FinalPNet2")
        h_conv_initFinalPNet2 = tf.nn.relu(
            _conv2d(h_conv_initFinalPNet, W_conv_init11FinalPNet2) + b_conv_init11FinalPNet2,
            name="h_conv_initFinalPNet2")

        W_conv_finalPnet =_weight_variable([1, 1, self.k, cc.Nx * cc.Ny], name="W_conv_finalPnet")
        b_conv_finalPnet = tf.Variable(tf.constant(0, shape=[cc.Nx * cc.Ny * cc.Nx * cc.Ny ], dtype=tf.float32), name="b_conv_finalPnet")
        h_conv_finalPnet = _conv2d(h_conv_initFinalPNet2, W_conv_finalPnet)
        fc1Pnet = tf.nn.softmax(tf.reshape(h_conv_finalPnet, [-1, cc.Nx * cc.Ny * cc.Nx * cc.Ny]) + b_conv_finalPnet)


        W_conv_init11FinalValue = _weight_variable([1, 1, self.k, self.k], name="W_conv_init11FinalValue")
        b_conv_init11FinalValue = tf.Variable(tf.constant(0, shape=[self.k], dtype=tf.float32),name="b_conv_init11FinalValue")
        h_conv_initFinalValue = tf.nn.relu(
            _conv2d(h_conv_intermediate[-1], W_conv_init11FinalValue) + b_conv_init11FinalValue,
            name="h_conv_initFinalValue")

        W_conv_finalValue = _weight_variable([1, 1, self.k, 1], name="W_conv_finalValue")
        b_conv_finalValue = tf.Variable(tf.constant(0, shape=[cc.Nx * cc.Ny ], dtype=tf.float32),
                                       name="b_conv_finalValue")
        h_conv_finalValue = _conv2d(h_conv_initFinalValue, W_conv_finalValue)
        fc1Value = tf.nn.relu(tf.reshape(h_conv_finalValue, [-1, cc.Nx * cc.Ny ]) + b_conv_finalValue)

        W_conv_finalSinglValue = _weight_variable([cc.Ny*cc.Nx, 1],
                                                   name="W_conv_finalSinglValue")
        b_conv_finalSinglValue = tf.Variable(tf.constant(0, shape=[1], dtype=tf.float32),
                                              name="b_conv_finalSinglValue")

        fc2ValueRaw = tf.matmul(fc1Value, W_conv_finalSinglValue) + b_conv_finalSinglValue
        fc2Value = tf.tanh(fc2ValueRaw)

        #y_conv = tf.concat([fc2Value, fc1Pnet], 1)

        outputV = fc2Value
        outputPos = fc1Pnet
        output = (outputV, outputPos)


        log_likelihood_costV = tf.losses.mean_squared_error(predictions=outputV, labels=yV)
        log_likelihood_costPos = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc1Pnet, labels=yPos))

        regLoss = tf.nn.l2_loss(self.weightList[0])
        for w in self.weightList[1:]:
            regLoss = regLoss + tf.nn.l2_loss(w)

        log_likelihood_cost = tf.reduce_mean(log_likelihood_costV + log_likelihood_costPos + 1e-4 * regLoss)
        # AdamOptimizer is faster at start but gets really spiky after 2-3 million steps.
        # train_step = tf.train.AdamOptimizer(1e-4).minimize(log_likelihood_cost, global_step=global_step)
        #learning_rate = tf.train.exponential_decay(1e-2, global_step, 4 * 10 ** 6, 0.5)
        train_step = tf.train.MomentumOptimizer(1e-2, 0.9).minimize(log_likelihood_cost, global_step=global_step)
        #train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(log_likelihood_cost, global_step=global_step)

        was_correct = tf.equal(tf.argmax(outputPos, 1), tf.argmax(yPos, 1))
        accuracy = tf.reduce_mean(tf.cast(was_correct, tf.float32))

        reinforce_step = tf.train.GradientDescentOptimizer(1e-2).minimize(
            log_likelihood_cost * reinforce_direction, global_step=RL_global_step)

        # weight_summaries = tf.summary.merge([
        #     tf.summary.histogram(weight_var.name, weight_var)
        #     for weight_var in [W_conv_init55, W_conv_init11] +  W_conv_intermediate + [W_conv_final, b_conv_final]],
        #     name="weight_summaries"
        # )
        # activation_summaries = tf.summary.merge([
        #     tf.summary.histogram(act_var.name, act_var)
        #     for act_var in [h_conv_init] + h_conv_intermediate + [h_conv_final]],
        #     name="activation_summaries"
        # )
        saver = tf.train.Saver()

        # save everything to self.
        for name, thing in list(locals().items()):
            if not name.startswith('_'):
                setattr(self, name, thing)

    def initialize_logging(self, tensorboard_logdir):
        self.test_summary_writer = tf.summary.FileWriter(os.path.join(tensorboard_logdir, "test"), self.session.graph)
        self.training_summary_writer = tf.summary.FileWriter(os.path.join(tensorboard_logdir, "training"), self.session.graph)

    def initialize_variables(self, save_file=None):
        self.save_file = save_file
        self.session.run(tf.global_variables_initializer())
        if save_file is not None:
            try:
                self.saver.restore(self.session, save_file)
            except:
                # some wizardry here... basically, only restore variables
                # that are in the save file; otherwise, initialize them normally.
                from tensorflow.python.framework import meta_graph
                meta_graph_def = meta_graph.read_meta_graph_file(save_file + '.meta')
                stored_var_names = set([n.name
                    for n in meta_graph_def.graph_def.node
                    if n.op == 'VariableV2'])
                print(stored_var_names)
                var_list = [v for v in tf.global_variables()
                    if v.op.name in stored_var_names]
                # initialize all of the variables
                self.session.run(tf.global_variables_initializer())
                # then overwrite the ones we have in the save file
                # by using a throwaway saver, saved models are automatically
                # "upgraded" to the latest graph definition.
                throwaway_saver = tf.train.Saver(var_list=var_list)
                throwaway_saver.restore(self.session, save_file)


    def get_global_step(self):
        return self.session.run(self.global_step)

    def save_variables(self, save_file = None):
        if save_file is not None:
            self.save_file = save_file
        print("Saving checkpoint to %s" % self.save_file, file=sys.stderr)
        self.saver.save(self.session, self.save_file)

    def trainOne(self, training_data, reinforce=1, punish=False):
        batch_x, batch_yPos, batch_yV = training_data.get_batch(training_data.data_size)

        _, accuracy, cost = self.session.run(
            [self.train_step, self.accuracy, self.log_likelihood_cost],
            feed_dict={self.x: batch_x,
                       self.yV: batch_yV,
                       self.yPos: batch_yPos.reshape(-1, 8100),
                       self.reinforce_direction: reinforce})
        print(cost)

    def train(self, training_data, batch_size=32):
        num_minibatches = training_data.data_size // batch_size
        for i in range(num_minibatches+1):
            if i == num_minibatches:
                batch_size = training_data.data_size % batch_size
            if batch_size <= 0:
                break
            batch_x, batch_yPos, batch_yV = training_data.get_batch(batch_size)
            _, accuracy, cost = self.session.run(
                [self.train_step, self.accuracy, self.log_likelihood_cost],
                feed_dict={self.x: batch_x,
                           self.yV: batch_yV,
                           self.yPos: batch_yPos.reshape(-1, 8100),
                           self.reinforce_direction: 1})
            self.training_stats.report(accuracy,cost)

        avg_accuracy, avg_cost, accuracy_summaries = self.training_stats.collect()
        global_step = self.get_global_step()
        print("Step %d training data accuracy: %g; cost: %g" % (global_step, avg_accuracy, avg_cost))

    def reinforce(self, dataset, direction=1, batch_size=32):
        num_minibatches = dataset.data_size // batch_size
        for i in range(num_minibatches):
            batch_x, batch_y = dataset.get_batch(batch_size)
            self.session.run(
                self.reinforce_step,
                feed_dict={self.x: batch_x, self.y: batch_y, self.reinforce_direction: direction})

    def run(self, position):
        'Return a sorted list of (probability, move) tuples'
        if PolicyNetwork.temper:

            return 0, np.ones([cc.Ny, cc.Nx,cc.Ny, cc.Nx], dtype=np.float32)/cc.Ny/ cc.Nx/cc.Ny/ cc.Nx
        processed_position = features.extract_features(position)
        probabilities = self.session.run([self.output], feed_dict={self.x: processed_position[None, :]})[0]

        ret = probabilities[1][0].reshape(10, 9, 10, 9)

        return probabilities[0][0][0], ret
#        return probabilities.reshape([cc.Ny, cc.Nx, cc.Ny, cc.Nx])

    def run_many(self, positions):
        processed_positions = features.bulk_extract_features(positions)
        probabilities = self.session.run(self.output, feed_dict={self.x:processed_positions})
        return probabilities.reshape([-1, cc.Ny, cc.Nx])

    def check_accuracy(self, test_data, batch_size=128):
        num_minibatches = test_data.data_size // batch_size
        weight_summaries = self.session.run(self.weight_summaries)

        for i in range(num_minibatches):
            batch_x, batch_y = test_data.get_batch(batch_size)
            accuracy, cost = self.session.run(
                [self.accuracy, self.log_likelihood_cost],
                feed_dict={self.x: batch_x, self.y: batch_y, self.reinforce_direction: 1})
            self.test_stats.report(accuracy, cost)

        avg_accuracy, avg_cost, accuracy_summaries = self.test_stats.collect()
        global_step = self.get_global_step()
        print("Step %s test data accuracy: %g; cost: %g" % (global_step, avg_accuracy, avg_cost))

        if self.test_summary_writer is not None:
            self.test_summary_writer.add_summary(weight_summaries, global_step)
            self.test_summary_writer.add_summary(accuracy_summaries, global_step)

class StatisticsCollector(object):
    '''
    Accuracy and cost cannot be calculated with the full test dataset
    in one pass, so they must be computed in batches. Unfortunately,
    the built-in TF summary nodes cannot be told to aggregate multiple
    executions. Therefore, we aggregate the accuracy/cost ourselves at
    the python level, and then s.reshape(10,9)hove it through the accuracy/cost summary
    nodes to generate the appropriate summary protobufs for writing.
    '''
    graph = tf.Graph()
    with tf.device("/cpu:0"), graph.as_default():
        accuracy = tf.placeholder(tf.float32, [])
        cost = tf.placeholder(tf.float32, [])
        accuracy_summary = tf.summary.scalar("accuracy", accuracy)
        cost_summary = tf.summary.scalar("log_likelihood_cost", cost)
        accuracy_summaries = tf.summary.merge([accuracy_summary, cost_summary], name="accuracy_summaries")
    session = tf.Session(graph=graph)

    def __init__(self):
        self.accuracies = []
        self.costs = []

    def report(self, accuracy, cost):
        self.accuracies.append(accuracy)
        self.costs.append(cost)

    def collect(self):
        avg_acc = sum(self.accuracies) / len(self.accuracies)
        avg_cost = sum(self.costs) / len(self.costs)
        self.accuracies = []
        self.costs = []
        summary = self.session.run(self.accuracy_summaries,
            feed_dict={self.accuracy:avg_acc, self.cost: avg_cost})
        return avg_acc, avg_cost, summary
