import tensorflow as tf


class Regularizer(object):

    def __init__(self, params):
        # ----------------------------------
        # regularization coefficients
        # ----------------------------------
        self.L1_in = params.get('L1_in', 0)
        self.L1_rec = params.get('L1_rec', 0)
        self.L1_out = params.get('L1_out', 0)

        self.L2_in = params.get('L2_in', 0)
        self.L2_rec = params.get('L2_rec', 0)
        self.L2_out = params.get('L2_out', 0)

        self.L2_firing_rate = params.get('L2_firing_rate', 0)
        self.sussillo_constant = params.get('sussillo_constant', 0)

    def set_model_regularization(self, model):
        reg = 0

        # ----------------------------------
        # L1 weight regularization
        # ----------------------------------
        reg += self.L1_weight_reg(model)

        # ----------------------------------
        # L2 weight regularization
        # ----------------------------------
        reg += self.L2_weight_reg(model)

        # ----------------------------------
        # L2 firing rate regularization
        # ----------------------------------
        reg += self.L2_firing_rate_reg(model)

        # ----------------------------------
        # Susillo regularization
        # ----------------------------------
        reg += self.sussillo_reg(model)

        return reg

    def L1_weight_reg(self, model):

        reg = 0

        reg += self.L1_in * tf.reduce_mean(tf.abs(model.W_in) * model.input_connectivity)
        reg += self.L1_rec * tf.reduce_mean(tf.abs(model.W_rec) * model.rec_connectivity)
        if model.dale_ratio:
            reg += self.L1_out * tf.reduce_mean(
                tf.matmul(tf.abs(model.W_out) * model.output_connectivity, model.Dale_out))
        else:
            reg += self.L1_out * tf.reduce_mean(tf.abs(model.W_out) * model.output_connectivity)

        return reg

    def L2_weight_reg(self, model):

        reg = 0

        reg += self.L2_in * tf.reduce_mean(tf.square(tf.abs(model.W_in) * model.input_connectivity))
        reg += self.L2_rec * tf.reduce_mean(tf.square(tf.abs(model.W_rec) * model.rec_connectivity))
        if model.dale_ratio:
            reg += model.L2_out * tf.reduce_mean(tf.square(
                tf.matmul(tf.abs(model.W_out) * model.output_connectivity, model.Dale_out)))
        else:
            reg += self.L2_out * tf.reduce_mean(tf.square(tf.abs(model.W_out) * model.output_connectivity))

        return reg

    def L2_firing_rate_reg(self, model):

        reg = self.L2_firing_rate * tf.reduce_mean(tf.square(tf.nn.relu(model.states)))

        return reg

    def sussillo_reg(self, model):
        states = tf.unstack(tf.transpose(model.states, [1, 0, 2]))

        reg = 0

        for state in states:
            dJr = tf.matmul(tf.nn.relu(state),
                            tf.matmul(tf.abs(model.W_rec) * model.rec_connectivity, model.Dale_rec))
            reg += tf.reduce_mean(tf.square(dJr))

        return self.sussillo_constant * (reg)