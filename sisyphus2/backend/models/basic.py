from sisyphus2.backend.rnn import RNN
import tensorflow as tf


class Basic(RNN):

    def recurrent_timestep(self, rnn_in, state):

        if self.dale_ratio:
            new_state = (1-self.alpha) * state \
                        + self.alpha * (
                            tf.matmul(
                                tf.nn.relu(state),
                                tf.matmul(
                                    tf.abs(self.W_rec) * self.rec_connectivity,
                                    self.Dale_rec, name="in_1"),
                                transpose_b=True, name="1")
                            + tf.matmul(
                                rnn_in,
                                tf.abs(self.W_in) * self.input_connectivity,
                                transpose_b=True, name="2")
                            + self.b_rec)\
                        + tf.sqrt(2.0 * self.alpha * self.rec_noise * self.rec_noise)\
                          * tf.random_normal(tf.shape(state), mean=0.0, stddev=1.0)

        else:
            new_state = ((1-self.alpha) * state) \
                        + self.alpha * (
                            tf.matmul(
                                tf.nn.relu(state),
                                self.W_rec * self.rec_connectivity,
                                transpose_b=True, name="1")
                            + tf.matmul(
                                rnn_in,
                                self.W_in * self.input_connectivity,
                                transpose_b=True, name="2")
                            + self.b_rec)\
                        + tf.sqrt(2.0 * self.alpha * self.rec_noise * self.rec_noise)\
                          * tf.random_normal(tf.shape(state), mean=0.0, stddev=1.0)

        return new_state

    def output_timestep(self, state):

        if self.dale_ratio:
            output = tf.matmul(tf.nn.relu(state),
                                    tf.matmul(tf.abs(self.W_out) * self.output_connectivity,
                                            self.Dale_out, name="in_2"), transpose_b=True, name="3") \
                        + self.b_out
        else:
            output = tf.matmul(tf.nn.relu(state),
                                self.W_out * self.output_connectivity, transpose_b=True, name="3") \
                        + self.b_out

        return output

    def forward_pass(self):

        rnn_inputs = tf.unstack(self.x, axis=1)
        state = self.init_state
        rnn_outputs = []
        rnn_states = []
        for rnn_input in rnn_inputs:
            state = self.recurrent_timestep(rnn_input, state)
            output = self.output_timestep(state)
            rnn_outputs.append(output)
            rnn_states.append(state)
        return tf.transpose(rnn_outputs, [1, 0, 2]), tf.transpose(rnn_states, [1, 0, 2])


class BasicSigmoid(Basic):

    def recurrent_timestep(self, rnn_in, state):

        if self.dale_ratio:
            new_state = (1-self.alpha) * state \
                        + self.alpha * (
                            tf.matmul(
                                tf.nn.sigmoid(state),
                                tf.matmul(
                                    tf.abs(self.W_rec) * self.rec_connectivity,
                                    self.Dale_rec, name="in_1"),
                                transpose_b=True, name="1")
                            + tf.matmul(
                                rnn_in,
                                tf.abs(self.W_in) * self.input_connectivity,
                                transpose_b=True, name="2")
                            + self.b_rec)\
                        + tf.sqrt(2.0 * self.alpha * self.rec_noise * self.rec_noise)\
                          * tf.random_normal(tf.shape(state), mean=0.0, stddev=1.0)

        else:
            new_state = ((1-self.alpha) * state) \
                        + self.alpha * (
                            tf.matmul(
                                tf.nn.sigmoid(state),
                                self.W_rec * self.rec_connectivity,
                                transpose_b=True, name="1")
                            + tf.matmul(
                                rnn_in,
                                self.W_in * self.input_connectivity,
                                transpose_b=True, name="2")
                            + self.b_rec)\
                        + tf.sqrt(2.0 * self.alpha * self.rec_noise * self.rec_noise)\
                          * tf.random_normal(tf.shape(state), mean=0.0, stddev=1.0)

        return new_state


class BasicScan(Basic):

    def recurrent_timestep_scan(self, state, rnn_in):
        return self.recurrent_timestep(rnn_in, state)

    def output_timestep_scan(self, dummy, state):
        return self.output_timestep(state)

    def forward_pass(self):
        state = self.init_state
        rnn_states = \
            tf.scan(
                self.recurrent_timestep_scan,
                tf.transpose(self.x, [1, 0, 2]),
                initializer=state,
                parallel_iterations=1)
        rnn_outputs = \
            tf.scan(
                self.output_timestep_scan,
                rnn_states,
                initializer=tf.zeros([self.N_batch, self.N_out]),
                parallel_iterations=1)
        return tf.transpose(rnn_outputs, [1, 0, 2]), tf.transpose(rnn_states, [1, 0, 2])