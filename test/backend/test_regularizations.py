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

def my_custom_regularization(model, params):
        return 0

@patch.object(RNN, '__abstractmethods__', set())
def test_custom_loss(tf_graph, mocker):

	params = get_params()
	params['custom_regularization'] = my_custom_regularization
	rnn = RNN(params)

	mocker.patch.object(RNN, 'forward_pass')
	RNN.forward_pass.return_value = tf.fill([params['N_batch'], params['N_steps'], params['N_out']], float('nan')), tf.fill([params['N_batch'], params['N_steps'], params['N_rec']], float('nan'))
	rnn.build()
