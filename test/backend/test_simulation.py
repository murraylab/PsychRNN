import pytest
import tensorflow as tf
from psychrnn.backend.models.basic import Basic
from psychrnn.backend.models.lstm import LSTM
from psychrnn.tasks.match_to_category import MatchToCategory
from psychrnn.backend.simulation import BasicSimulator, LSTMSimulator
import numpy as np
import random

# clears tf graph after each test.
@pytest.fixture()
def tf_graph():
    yield
    tf.compat.v1.reset_default_graph()

def reset_seeds(seed):
	tf.compat.v1.reset_default_graph()
	tf.compat.v1.set_random_seed(seed)
	random.seed(seed)
	np.random.seed(seed)

def test_load_from_file(tf_graph, tmpdir, capfd):
	reset_seeds(19846)

	mtc = MatchToCategory(dt = 10, tau = 100, T= 2000, N_batch = 50)

	params = mtc.get_task_params()
	params['name'] = 'test'
	params['N_rec'] = 49
	rnn_model = Basic(params)
	rnn_model.save(str(tmpdir.dirpath("save_weights.npz")))
	x,_,_,_ = mtc.get_trial_batch()

	sim_model = BasicSimulator(params = params, weights_path=str(tmpdir.dirpath("save_weights.npz")))
	tmpdir.dirpath("save_weights.npz").remove()


	tf_output, tf_state = rnn_model.test(x)
	sim_output, sim_state = sim_model.run_trials(x)

	assert(tf_output.shape == sim_output.shape)
	assert(tf_state.shape == sim_state.shape)
	assert(np.allclose(tf_state, sim_state, atol=1e-06))
	assert(np.allclose(tf_output, sim_output, atol=1e-06))

	rnn_model.destruct()



def test_load_from_alpha_params(tf_graph):
	reset_seeds(19846)

	mtc = MatchToCategory(dt = 10, tau = 100, T= 2000, N_batch = 50)

	params = mtc.get_task_params()
	params['name'] = 'test'
	params['N_rec'] = 49
	rnn_model = Basic(params)
	weights = rnn_model.get_weights()
	x,_,_,_ = mtc.get_trial_batch()

	sim_model = BasicSimulator(params = {'alpha': params['alpha']}, weights=weights)


	tf_output, tf_state = rnn_model.test(x)
	sim_output, sim_state = sim_model.run_trials(x)

	assert(tf_output.shape == sim_output.shape)
	assert(tf_state.shape == sim_state.shape)
	assert(np.allclose(tf_state, sim_state, atol=1e-06))
	assert(np.allclose(tf_output, sim_output, atol=1e-06))

	rnn_model.destruct()

def test_load_from_dt_tau_params(tf_graph):
	reset_seeds(19846)

	mtc = MatchToCategory(dt = 10, tau = 100, T= 2000, N_batch = 50)

	params = mtc.get_task_params()
	params['name'] = 'test'
	params['N_rec'] = 49
	rnn_model = Basic(params)
	weights = rnn_model.get_weights()
	x,_,_,_ = mtc.get_trial_batch()

	sim_model = BasicSimulator(params = {'dt': params['dt'], 'tau': params['tau']}, weights=weights)


	tf_output, tf_state = rnn_model.test(x)
	sim_output, sim_state = sim_model.run_trials(x)

	assert(tf_output.shape == sim_output.shape)
	assert(tf_state.shape == sim_state.shape)
	assert(np.allclose(tf_state, sim_state, atol=1e-06))
	assert(np.allclose(tf_output, sim_output, atol=1e-06))
	rnn_model.destruct()

def test_load_from_params(tf_graph):
	reset_seeds(19846)

	mtc = MatchToCategory(dt = 10, tau = 100, T= 2000, N_batch = 50)

	params = mtc.get_task_params()
	params['name'] = 'test'
	params['N_rec'] = 49
	rnn_model = Basic(params)
	weights = rnn_model.get_weights()
	x,_,_,_ = mtc.get_trial_batch()

	sim_model = BasicSimulator(params = params, weights=weights)


	tf_output, tf_state = rnn_model.test(x)
	sim_output, sim_state = sim_model.run_trials(x)

	assert(tf_output.shape == sim_output.shape)
	assert(tf_state.shape == sim_state.shape)
	assert(np.allclose(tf_state, sim_state, atol=1e-06))
	assert(np.allclose(tf_output, sim_output, atol=1e-06))
	rnn_model.destruct()


def test_load_from_rnn_model(tf_graph):
	reset_seeds(19846)

	mtc = MatchToCategory(dt = 10, tau = 100, T= 2000, N_batch = 50)

	params = mtc.get_task_params()
	params['name'] = 'test'
	params['N_rec'] = 49
	rnn_model = Basic(params)
	x,_,_,_ = mtc.get_trial_batch()

	sim_model = BasicSimulator(rnn_model = rnn_model)


	tf_output, tf_state = rnn_model.test(x)
	sim_output, sim_state = sim_model.run_trials(x)

	assert(tf_output.shape == sim_output.shape)
	assert(tf_state.shape == sim_state.shape)
	assert(np.allclose(tf_state, sim_state, atol=1e-06))
	assert(np.allclose(tf_output, sim_output, atol=1e-06))

	rnn_model.destruct()


def test_transfer_function(tf_graph):
	def my_relu(X):
		return np.maximum(X, 0)

	reset_seeds(19846)

	mtc = MatchToCategory(dt = 10, tau = 100, T= 2000, N_batch = 50)

	params = mtc.get_task_params()
	params['name'] = 'test'
	params['N_rec'] = 49
	rnn_model = Basic(params)
	x,_,_,_ = mtc.get_trial_batch()

	with pytest.raises(UserWarning) as excinfo:
		sim_model = BasicSimulator(rnn_model = rnn_model, transfer_function=my_relu)
	assert 'my_relu' in str(excinfo.value)

	rnn_model.destruct()


def test_rec_noise_rnn_model(tf_graph):
	reset_seeds(19846)

	mtc = MatchToCategory(dt = 10, tau = 100, T= 2000, N_batch = 50)

	params = mtc.get_task_params()
	params['name'] = 'test'
	params['N_rec'] = 49
	params['rec_noise'] = .1
	rnn_model = Basic(params)
	x,_,_,_ = mtc.get_trial_batch()

	sim_model = BasicSimulator(rnn_model = rnn_model)
	assert(sim_model.rec_noise == rnn_model.rec_noise)
	assert(sim_model.rec_noise == .1)

	rnn_model.destruct()

def test_rec_noise_params(tf_graph):
	reset_seeds(19846)

	mtc = MatchToCategory(dt = 10, tau = 100, T= 2000, N_batch = 50)

	params = mtc.get_task_params()
	params['name'] = 'test'
	params['N_rec'] = 49
	params['rec_noise'] = .1
	rnn_model = Basic(params)
	weights = rnn_model.get_weights()
	x,_,_,_ = mtc.get_trial_batch()

	sim_model = BasicSimulator(params = params, weights = weights)
	assert(sim_model.rec_noise == params['rec_noise'])
	rnn_model.destruct()

def test_warnings(tf_graph, tmpdir, capfd):
	reset_seeds(19846)

	mtc = MatchToCategory(dt = 10, tau = 100, T= 2000, N_batch = 50)

	params = mtc.get_task_params()
	params['name'] = 'test'
	params['N_rec'] = 49
	params['rec_noise'] = .1
	rnn_model = Basic(params)
	weights = rnn_model.get_weights()
	rnn_model.save(str(tmpdir.dirpath("save_weights.npz")))
	x,_,_,_ = mtc.get_trial_batch()

	with pytest.raises(UserWarning) as excinfo:
		sim_model = BasicSimulator(params = params, rnn_model = rnn_model)
	assert 'rnn_model takes precedence' in str(excinfo.value)

	with pytest.raises(UserWarning) as excinfo:
		sim_model = BasicSimulator(weights = weights, rnn_model = rnn_model)
	assert 'Weights from rnn_model and weights_path will be ignored' in str(excinfo.value)

	with pytest.raises(UserWarning) as excinfo:
		sim_model = BasicSimulator(weights_path = str(tmpdir.dirpath("save_weights.npz")), weights = weights, params=params)
	assert 'Weights from rnn_model and weights_path will be ignored' in str(excinfo.value)

	with pytest.raises(UserWarning) as excinfo:
		sim_model = BasicSimulator(weights_path = str(tmpdir.dirpath("save_weights.npz")), rnn_model = rnn_model)
	assert 'Weights from rnn_model will be ignored.' in str(excinfo.value)

	with pytest.raises(UserWarning) as excinfo:
		sim_model = BasicSimulator(params=params)
	assert 'Either weights, rnn_model, or weights_path must be passed in.' in str(excinfo.value)

	rnn_model.destruct()
	tmpdir.dirpath("save_weights.npz").remove()


def test_lstm_simulator_load_from_rnn_model(tf_graph):
	reset_seeds(19846)
	tf.compat.v1.keras.backend.set_floatx('float64')

	mtc = MatchToCategory(dt = 10, tau = 100, T= 2000, N_batch = 50)

	params = mtc.get_task_params()
	params['name'] = 'test'
	params['N_rec'] = 49
	rnn_model = LSTM(params)
	x,_,_,_ = mtc.get_trial_batch()

	sim_model = LSTMSimulator(rnn_model = rnn_model)


	tf_output, tf_state = rnn_model.test(x)
	sim_output, sim_state = sim_model.run_trials(x)

	assert(tf_output.shape == sim_output.shape)
	assert(tf_state.shape == sim_state.shape)
	assert(np.allclose(tf_state, sim_state, atol=1e-01,rtol=1e-01))
	assert(np.allclose(tf_output, sim_output, atol=1e-01,rtol=1e-01))

	rnn_model.destruct()

	rnn_model = LSTM(params)
	tf_output, tf_state = rnn_model.test(x)
	assert(not np.allclose(tf_state, sim_state, atol=1e-01,rtol=1e-01))
	assert(not np.allclose(tf_output, sim_output, atol=1e-01,rtol=1e-01))


