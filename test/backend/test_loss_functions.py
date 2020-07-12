import pytest
import tensorflow as tf
from psychrnn.backend.rnn import RNN
from pytest_mock import mocker

import sys

if sys.version_info[0] == 2:
    from mock import patch
else:
    from unittest.mock import patch


# clears tf graph after each test.
@pytest.fixture()
def tf_graph():
    yield
    tf.compat.v1.reset_default_graph()

def get_params():
	params = {}
	params['name'] = "test"
	params['N_in'] = 2
	params['N_rec'] = 50
	params['N_out'] = 2
	params['N_steps'] = 200
	params['dt'] = 10
	params['tau'] = 100
	params['N_batch'] = 50
	return params

def mean_squared_error(predictions, y, output_mask):
        """ Mean squared error.

        ``loss = mean(square(output_mask * (predictions - y)))``

        Args:
            predictions (*tf.Tensor(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Network output.
            y (*tf.Tensor(dtype=float, shape =(*?, :attr:`N_steps`, :attr:`N_out` *))*): Target output.
            output_mask (*tf.Tensor(dtype=float, shape =(*?, :attr:`N_steps`, :attr:`N_out` *))*): Output mask for :attr:`N_batch` trials. True when the network should aim to match the target output, False when the target output can be ignored.

        Returns:
            tf.Tensor(dtype=float): Mean squared error.

        """

        return tf.reduce_mean(input_tensor=tf.square(output_mask * (predictions - y)))

@patch.object(RNN, '__abstractmethods__', set())
def test_custom_loss(tf_graph, mocker):

	params = get_params()
	params['loss_function'] = 'my_mean_squared_error'
	rnn = RNN(params)

	mocker.patch.object(RNN, 'forward_pass')
	RNN.forward_pass.return_value = tf.fill([params['N_batch'], params['N_steps'], params['N_out']], float('nan')), tf.fill([params['N_batch'], params['N_steps'], params['N_rec']], float('nan'))

	with pytest.raises(UserWarning) as excinfo:
		rnn.build()
	assert 'my_mean_squared_error' in str(excinfo.value)
	rnn.destruct()

	params['my_mean_squared_error'] = mean_squared_error
	rnn = RNN(params)

	mocker.patch.object(RNN, 'forward_pass')
	RNN.forward_pass.return_value = tf.fill([params['N_batch'], params['N_steps'], params['N_out']], float('nan')), tf.fill([params['N_batch'], params['N_steps'], params['N_rec']], float('nan'))

	rnn.build()
