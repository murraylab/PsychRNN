from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import sys
from time import time
from os import makedirs, path

from psychrnn.backend.regularizations import Regularizer
from psychrnn.backend.loss_functions import LossFunction
from psychrnn.backend.initializations import WeightInitializer, GaussianSpectralRadius


class RNN(object):
    def __init__(self, params):
        self.params = params

        # --------------------------------------------
        # Unique name used to determine variable scope
        # --------------------------------------------
        try:
            self.name = params['name']
        except KeyError:
            print("You must pass a  'name' to RNN")
            raise

        # ----------------------------------
        # Network sizes (tensor dimensions)
        # ----------------------------------
        try:
            N_in = self.N_in = params['N_in']
        except KeyError:
            print("You must pass 'N_in' to RNN")
            raise
        try:
            N_rec = self.N_rec = params['N_rec']
        except KeyError:
            print("You must pass 'N_rec' to RNN")
            raise
        try:
            N_out = self.N_out = params['N_out']
        except KeyError:
            print("You must pass 'N_out' to RNN")
            raise
        try:
            N_steps = self.N_steps = params['N_steps']
        except KeyError:
            print("You must pass 'N_steps' to RNN")
            raise

        # ----------------------------------
        # Physical parameters
        # ----------------------------------
        try:
            self.dt = params['dt']
        except KeyError:
            print("You must pass 'dt' to RNN")
            raise
        try:
            self.tau = params['tau']
        except KeyError:
            print("You must pass 'dt' to RNN")
            raise
        try:
            self.N_batch = params['N_batch']
        except KeyError:
            print("You must pass 'N_batch' to RNN")
            raise
            
        self.alpha = (1.0 * self.dt) / self.tau
        self.dale_ratio = params.get('dale_ratio', None)
        self.rec_noise = params.get('rec_noise', 0.0)

        # ----------------------------------
        # Dale's law matrix
        # ----------------------------------
        dale_vec = np.ones(N_rec)
        if self.dale_ratio is not None:
            dale_vec[int(self.dale_ratio * N_rec):] = -1
            self.dale_rec = np.diag(dale_vec)
            dale_vec[int(self.dale_ratio * N_rec):] = 0
            self.dale_out = np.diag(dale_vec)
        else:
            self.dale_rec = np.diag(dale_vec)
            self.dale_out = np.diag(dale_vec)

        # ----------------------------------
        # Trainable features
        # ----------------------------------
        self.W_in_train = params.get('W_in_train', True)
        self.W_rec_train = params.get('W_rec_train', True)
        self.W_out_train = params.get('W_out_train', True)
        self.b_rec_train = params.get('b_rec_train', True)
        self.b_out_train = params.get('b_out_train', True)
        self.init_state_train = params.get('init_state_train', True)

        # ----------------------------------
        # Load weights path
        # ----------------------------------
        self.load_weights_path = params.get('load_weights_path', None)

        # ------------------------------------------------
        # Define initializer for TensorFlow variables
        # ------------------------------------------------
        if self.load_weights_path is not None:
            self.initializer = WeightInitializer(load_weights_path=self.load_weights_path)
        else:
            self.initializer = params.get('initializer',
                                          GaussianSpectralRadius(N_in=N_in,
                                                                 N_rec=N_rec, N_out=N_out,
                                                                 autapses=True, spec_rad=1.1))

        # --------------------------------------------------
        # Tensorflow input/output placeholder initializations
        # ---------------------------------------------------
        self.x = tf.placeholder("float", [None, N_steps, N_in])
        self.y = tf.placeholder("float", [None, N_steps, N_out])
        self.output_mask = tf.placeholder("float", [None, N_steps, N_out])

        # --------------------------------------------------
        # Initialize variables in proper scope
        # ---------------------------------------------------
        with tf.variable_scope(self.name) as scope:
            # ------------------------------------------------
            # Trainable variables:
            # Initial State, weight matrices and biases
            # ------------------------------------------------

            self.init_state = tf.get_variable('init_state', [1, N_rec],
                                              initializer=self.initializer.get('init_state'),
                                              trainable=self.init_state_train)
            self.init_state = tf.tile(self.init_state, [self.N_batch, 1])

            # Input weight matrix:
            self.W_in = \
                tf.get_variable('W_in', [N_rec, N_in],
                                initializer=self.initializer.get('W_in'),
                                trainable=self.W_in_train)

            # Recurrent weight matrix:
            self.W_rec = \
                tf.get_variable(
                    'W_rec',
                    [N_rec, N_rec],
                    initializer=self.initializer.get('W_rec'),
                    trainable=self.W_rec_train)

            # Output weight matrix:
            self.W_out = tf.get_variable('W_out', [N_out, N_rec],
                                         initializer=self.initializer.get('W_out'),
                                         trainable=self.W_out_train)

            # Recurrent bias:
            self.b_rec = tf.get_variable('b_rec', [N_rec], initializer=self.initializer.get('b_rec'),
                                         trainable=self.b_rec_train)
            # Output bias:
            self.b_out = tf.get_variable('b_out', [N_out], initializer=self.initializer.get('b_out'),
                                         trainable=self.b_out_train)

            # ------------------------------------------------
            # Non-trainable variables:
            # Overall connectivity and Dale's law matrices
            # ------------------------------------------------

            # Recurrent Dale's law weight matrix:
            self.Dale_rec = tf.get_variable('Dale_rec', [N_rec, N_rec],
                                            initializer=tf.constant_initializer(self.dale_rec),
                                            trainable=False)

            # Output Dale's law weight matrix:
            self.Dale_out = tf.get_variable('Dale_out', [N_rec, N_rec],
                                            initializer=tf.constant_initializer(self.dale_out),
                                            trainable=False)

            # Connectivity weight matrices:
            self.input_connectivity = tf.get_variable('input_connectivity', [N_rec, N_in],
                                                      initializer=self.initializer.get('input_connectivity'),
                                                      trainable=False)
            self.rec_connectivity = tf.get_variable('rec_connectivity', [N_rec, N_rec],
                                                    initializer=self.initializer.get('rec_connectivity'),
                                                    trainable=False)
            self.output_connectivity = tf.get_variable('output_connectivity', [N_out, N_rec],
                                                       initializer=self.initializer.get('output_connectivity'),
                                                       trainable=False)

        # --------------------------------------------------
        # Flag to check if variables initialized, model built
        # ---------------------------------------------------
        self.is_initialized = False
        self.is_built = False

    def build(self):
        # --------------------------------------------------
        # Define the predictions
        # --------------------------------------------------
        self.predictions, self.states = self.forward_pass()

        # --------------------------------------------------
        # Define the loss (based on the predictions)
        # --------------------------------------------------
        self.loss = LossFunction(self.params).set_model_loss(self)

        # --------------------------------------------------
        # Define the regularization
        # --------------------------------------------------
        self.reg = Regularizer(self.params).set_model_regularization(self)

        # --------------------------------------------------
        # Define the total regularized loss
        # --------------------------------------------------
        self.reg_loss = self.loss + self.reg

        # --------------------------------------------------
        # Open a session
        # --------------------------------------------------
        self.sess = tf.Session()

        # --------------------------------------------------
        # Record successful build
        # --------------------------------------------------
        self.is_built = True

        return

    def destruct(self):
        # --------------------------------------------------
        # Close the session. Delete the graph.
        # --------------------------------------------------
        if self.is_built:
            self.sess.close()
        tf.reset_default_graph()
        return

    def recurrent_timestep(self, rnn_in, state):
        raise UserWarning("recurrent_timestep must be implemented in child class. See Basic for example.")

    def output_timestep(self, state):
        raise UserWarning("output_timestep must be implemented in child class. See Basic for example.")

    def forward_pass(self):
        raise UserWarning("forward_pass must be implemented in child class. See Basic for example.")

    def get_weights(self):
        if not self.is_initialized or not self.is_built:
            raise UserWarning("No weights to return yet -- model has not yet been initialized.")
        else:
            weights_dict = dict()

            for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                # avoid saving duplicates
                if var.name.endswith(':0') and var.name.startswith(self.name):
                    name = var.name[len(self.name)+1:-2]
                    weights_dict.update({name: var.eval(session=self.sess)})
            return weights_dict

    def save(self, save_path):

        weights_dict = self.get_weights()

        np.savez(save_path, **weights_dict)

        return

    def train(self, trial_batch_generator, train_params={}):
        if not self.is_built:
            raise UserWarning("Must build network before training. Call build() before calling train().")

        t0 = time()
        # --------------------------------------------------
        # Extract params
        # --------------------------------------------------
        learning_rate = train_params.get('learning_rate', .001)
        training_iters = train_params.get('training_iters', 10000)
        loss_epoch = train_params.get('loss_epoch', 10)
        verbosity = train_params.get('verbosity', True)
        save_weights_path = train_params.get('save_weights_path', None)
        save_training_weights_epoch = train_params.get('save_training_weights_epoch', 100)
        training_weights_path = train_params.get('training_weights_path', None)
        curriculum = train_params.get('curriculum', None)
        optimizer = train_params.get('optimizer',
                                     tf.train.AdamOptimizer(learning_rate=learning_rate))
        clip_grads = train_params.get('clip_grads', True)
        performance_cutoff = train_params.get('performance_cutoff', None)
        performance_measure = train_params.get('performance_measure', None)

        if (performance_cutoff is not None and performance_measure is None) or (performance_cutoff is None and performance_measure is not None):
                raise UserWarning("training will not be cutoff based on performance. Make sure both performance_measure and performance_cutoff are defined")

        if curriculum is not None:
            trial_batch_generator = curriculum.get_generator_function()

        # --------------------------------------------------
        # Make weights folder if it doesn't already exist.
        # --------------------------------------------------
        if save_weights_path != None:
            if path.dirname(save_weights_path) != "" and not path.exists(path.dirname(save_weights_path)):
                makedirs(path.dirname(save_weights_path))

        # --------------------------------------------------
        # Make train weights folder if it doesn't already exist.
        # --------------------------------------------------
        if training_weights_path != None:
            if path.dirname(training_weights_path) != "" and not path.exists(path.dirname(training_weights_path)):
                makedirs(path.dirname(_weights_path))

        # --------------------------------------------------
        # Compute gradients
        # --------------------------------------------------
        grads = optimizer.compute_gradients(self.reg_loss)

        # --------------------------------------------------
        # Clip gradients
        # --------------------------------------------------
        if clip_grads:
            grads = [(tf.clip_by_norm(grad, 1.0), var)
                     if grad is not None else (grad, var)
                     for grad, var in grads]

        # --------------------------------------------------
        # Call the optimizer and initialize variables
        # --------------------------------------------------
        optimize = optimizer.apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())
        self.is_initialized = True

        # --------------------------------------------------
        # Record training time for performance benchmarks
        # --------------------------------------------------
        t1 = time()

        # --------------------------------------------------
        # Training loop
        # --------------------------------------------------
        epoch = 1
        batch_size = next(trial_batch_generator)[0].shape[0]
        losses = []
        if performance_cutoff is not None:
            performance = performance_cutoff - 1

        while epoch * batch_size < training_iters and (performance_cutoff is None or performance < performance_cutoff):
            batch_x, batch_y, output_mask = next(trial_batch_generator)
            self.sess.run(optimize, feed_dict={self.x: batch_x, self.y: batch_y, self.output_mask: output_mask})
            # --------------------------------------------------
            # Output batch loss
            # --------------------------------------------------
            if epoch % loss_epoch == 0:
                reg_loss = self.sess.run(self.reg_loss,
                                feed_dict={self.x: batch_x, self.y: batch_y, self.output_mask: output_mask})
                losses.append(reg_loss)
                if verbosity:
                    print("Iter " + str(epoch * batch_size) + ", Minibatch Loss= " + \
                          "{:.6f}".format(reg_loss))

            # --------------------------------------------------
            # Allow for curriculum learning
            # --------------------------------------------------
            if curriculum is not None and epoch % curriculum.metric_epoch == 0:
                trial_batch, trial_y, output_mask = next(trial_batch_generator)
                output, _ = self.test(trial_batch)
                if curriculum.metric_test(trial_batch, trial_y, output_mask, output, epoch, losses, verbosity):
                    if curriculum.stopTraining:
                        break
                    trial_batch_generator = curriculum.get_generator_function()

            # --------------------------------------------------
            # Save intermediary weights
            # --------------------------------------------------
            if epoch % save_training_weights_epoch == 0:
                if training_weights_path is not None:
                    self.save(training_weights_path + str(epoch))
                    if verbosity:
                        print("Training weights saved in file: %s" % training_weights_path + str(epoch))
            
            # ---------------------------------------------------
            # Update performance value if necessary
            # ---------------------------------------------------
            if performance_measure is not None:
                trial_batch, trial_y, output_mask = next(trial_batch_generator)
                output, _ = self.test(trial_batch)
                performance = performance_measure(trial_batch, trial_y, output_mask, output, epoch, losses, verbosity)
                if verbosity:
                    print("performance: " + str(performance))
            epoch += 1

        t2 = time()
        if verbosity:
            print("Optimization finished!")

        # --------------------------------------------------
        # Save final weights
        # --------------------------------------------------
        if save_weights_path is not None:
            self.save(save_weights_path)
            if verbosity:
                print("Model saved in file: %s" % save_weights_path)

        # --------------------------------------------------
        # Return losses, training time, initialization time
        # --------------------------------------------------
        return losses, (t2 - t1), (t1 - t0)

    def test(self, trial_batch):
        if not self.is_built:
            raise UserWarning("Must build network before training. Call build() before calling train().")

        if not self.is_initialized:
            self.sess.run(tf.global_variables_initializer())

        # --------------------------------------------------
        # Run the forward pass on trial_batch
        # --------------------------------------------------
        outputs, states = self.sess.run([self.predictions, self.states],
                                        feed_dict={self.x: trial_batch})

        return outputs, states