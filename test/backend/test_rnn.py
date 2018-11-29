import pytest
import tensorflow as tf
from psychrnn.backend.rnn import RNN
from psychrnn.backend.initializations import GaussianSpectralRadius
from psychrnn.tasks import rdm as rd  

# clears tf graph after each test.
@pytest.fixture()
def tf_graph():
    yield
    tf.reset_default_graph()

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
# The next 3 functions currently undefined in RNN. should I just make sure they pass? Or have it print a warning, unimplemented, along with message of type it should return?
def test_recurrent_timestep():
	pass

def test_output_timestep():
	pass

def test_forward_pass():
	pass

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

def test_train(tf_graph):
	rdm = rd.RDM(dt = 10, tau = 100, T = 2000, N_batch = 128)  
	gen = rdm.batch_generator()

	params = get_params()
	rnn = RNN(params)
	with pytest.raises(UserWarning) as excinfo:
		rnn.train(gen)
	assert 'build' in str(excinfo.value)
	#Also test when built



def test_test():
	pass


