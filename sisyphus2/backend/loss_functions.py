import tensorflow as tf


class LossFunction(object):

    def __init__(self, params):
        self.type = params.get("loss_function", "mean_squared_error")

    def set_model_loss(self, model):

        loss = 0

        if self.type == "mean_squared_error":
            loss = self.mean_squared_error(model.predictions, model.y, model.output_mask)

        # TODO TEST THIS
        if self.type == "binary_cross_entropy":
            loss = self.binary_cross_entropy(model.prediction, model.y, model.output_mask)

        return loss

    def mean_squared_error(self, predictions, y, output_mask):
        return tf.reduce_mean(tf.square(output_mask * (predictions - y)))

    def binary_cross_entropy(self, predictions, y, output_mask):
        return tf.reduce_mean( output_mask *
                               -(y * tf.log(predictions) + (1 - y) * tf.log(1 - predictions)))



