import pytest
import tensorflow as tf
from psychrnn.backend.rnn import RNN
from psychrnn.backend.initializations import GaussianSpectralRadius
from psychrnn.tasks import rdm as rd  
from pytest_mock import mocker

# clears tf graph after each test.
@pytest.fixture()
def tf_graph():
    yield
    tf.reset_default_graph()

def get_params():
	params = {}
	params['name'] = "test"
	params['N_in'] = 3
	params['N_rec'] = 51
	params['N_out'] = 2
	params['N_steps'] = 200
	params['dt'] = 10
	params['tau'] = 100
	params['N_batch'] = 50
	return params

def extend_params(params):
	params['dale_ratio'] = .2
	params['rec_noise'] = .01
	params['W_in_train'] = False
	params['W_rec_train'] = False
	params['b_rec_train'] = False
	params['b_out_train'] = False
	params['init_state_train'] = False
	return params

def test_minimal_info_rnn(tf_graph):
	params = get_params()
	del params['name']
	with pytest.raises(KeyError) as excinfo:
		RNN(params)
	assert 'name' in str(excinfo.value)
	
	params = get_params()
	del params['N_in']
	with pytest.raises(KeyError) as excinfo:
		RNN(params)
	assert 'N_in' in str(excinfo.value)

	params = get_params()
	del params['N_rec']
	with pytest.raises(KeyError) as excinfo:
		RNN(params)
	assert 'N_rec' in str(excinfo.value)

	params = get_params()
	del params['N_out']
	with pytest.raises(KeyError) as excinfo:
		RNN(params)
	assert 'N_out' in str(excinfo.value)

	params = get_params()
	del params['N_steps']
	with pytest.raises(KeyError) as excinfo:
		RNN(params)
	assert 'N_steps' in str(excinfo.value)

	params = get_params()
	del params['dt']
	with pytest.raises(KeyError) as excinfo:
		RNN(params)
	assert 'dt' in str(excinfo.value)

	params = get_params()
	del params['tau']
	with pytest.raises(KeyError) as excinfo:
		RNN(params)
	assert 'tau' in str(excinfo.value)

	params = get_params()
	del params['N_batch']
	with pytest.raises(KeyError) as excinfo:
		RNN(params)
	assert 'N_batch' in str(excinfo.value)

	# Test RNN works works if minimal info given.
	params = get_params()
	RNN(params)

def test_extra_info_rnn(tf_graph):
	params = get_params()
	params = extend_params(params)
	RNN(params)

#TODO(Jasmine): test load weights after testing save weights in train
def test_load_weights_path_rnn():
	pass

def test_initializer_rnn(tf_graph):
	params = get_params()
	params = extend_params(params)
	params['initializer'] = GaussianSpectralRadius(N_in=params['N_in'], N_rec=params['N_rec'], N_out=params['N_out'],autapses=True, spec_rad=1.1)
	RNN(params)

def test_build(tf_graph):
	pass
	#TODO(Jasmine): doesn't work with forward pass unitialized -- should I fake initialize it or do something as a catch for ppl writing code? Otherwise should RNN be an abstract class of sorts?

def test_destruct(tf_graph):
	params = get_params()
	rnn1 = RNN(params)
	rnn1.destruct()
	rnn2 = RNN(params)
	#TODO(Jasmine): also test when built

	params = get_params()
	rnn = RNN(params)
	with pytest.raises(UserWarning) as excinfo:
		rnn.recurrent_timestep(1,2)
	assert 'recurrent_timestep' in str(excinfo.value)

def test_output_timestep(tf_graph):
	params = get_params()
	rnn = RNN(params)
	with pytest.raises(UserWarning) as excinfo:
		rnn.output_timestep(1)
	assert 'output_timestep' in str(excinfo.value)

def test_forward_pass(tf_graph):
	params = get_params()
	rnn = RNN(params)
	with pytest.raises(UserWarning) as excinfo:
		rnn.forward_pass()
	assert 'forward_pass' in str(excinfo.value)

def test_get_weights(tf_graph):
	params = get_params()
	rnn = RNN(params)
	with pytest.raises(UserWarning) as excinfo:
		rnn.get_weights()
	assert 'No weights to return yet -- model has not yet been initialized.' in str(excinfo.value)
	#TODO(jasmine): also test once actual weights exist

def test_save(tf_graph):
	params = get_params()
	rnn = RNN(params)
	with pytest.raises(UserWarning) as excinfo:
		rnn.save("save_weights_path")
	#TODO(Jasmine): also test with actual weights

def test_train(tf_graph, mocker):
	rdm = rd.RDM(dt = 10, tau = 100, T = 2000, N_batch = 128)  
	gen = rdm.batch_generator()

	params = get_params()
	rnn = RNN(params)
	with pytest.raises(UserWarning) as excinfo:
		rnn.train(gen)
	assert 'build' in str(excinfo.value)

	mocker.patch.object(RNN, 'forward_pass')
	RNN.forward_pass.return_value = tf.fill([params['N_batch'], params['N_steps'], params['N_out']], float('nan')), tf.fill([params['N_batch'], params['N_steps'], params['N_rec']], float('nan'))
	rnn.build()
	#TODO(jasmine): Also test when built



def test_test():
	rdm = rd.RDM(dt = 10, tau = 100, T = 2000, N_batch = 128)  
	gen = rdm.batch_generator()
	x,y,m = next(gen)

	params = get_params()
	rnn = RNN(params)
	with pytest.raises(UserWarning) as excinfo:
		rnn.test(x)
	assert 'build' in str(excinfo.value)
	#TODO(Jasmine): Also test when built


