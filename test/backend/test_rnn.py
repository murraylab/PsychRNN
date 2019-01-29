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
def test_load_weights_path_rnn(tf_graph,mocker,tmpdir, capfd):
	params = get_params()

	rdm1 = rd.RDM(dt = params['dt'], tau = params['tau'], T = 2000, N_batch = params['N_batch'])  
	gen1 = rdm1.batch_generator()

	mocker.patch.object(RNN, 'forward_pass')
	RNN.forward_pass.return_value = tf.fill([params['N_batch'], params['N_steps'], params['N_out']], float('nan')), tf.fill([params['N_batch'], params['N_steps'], params['N_rec']], float('nan'))
	
	rnn =RNN(params)
	rnn.build()

	train_params = {}
	train_params['save_weights_path'] =  str(tmpdir.dirpath("save_weights.npz")) # Where to save the model after training. Default: None
	train_params['verbosity'] = False

	### save out some weights to test with and destroy the rnn that created them
	assert not tmpdir.dirpath("save_weights.npz").check(exists=1)
	rnn.train(gen1, train_params)

	assert rnn.is_initialized is True
	out, _ = capfd.readouterr()
	print(out)
	assert out==""
	assert tmpdir.dirpath("save_weights.npz").check(exists=1)
	rnn.destruct()

	### Make sure loading weights fails on nonexistent file
	params['load_weights_path'] = "nonexistent"
	with pytest.raises(EnvironmentError) as excinfo:
		rnn = RNN(params)
	assert "No such file" in str(excinfo.value)
	rnn.destruct()

	### Ensure success when loading weights created previously
	params['load_weights_path'] = str(tmpdir.dirpath("save_weights.npz"))
	rnn = RNN(params)

	tmpdir.dirpath("save_weights.npz").remove()
	

def test_initializer_rnn(tf_graph):
	params = get_params()
	params = extend_params(params)
	params['initializer'] = GaussianSpectralRadius(N_in=params['N_in'], N_rec=params['N_rec'], N_out=params['N_out'],autapses=True, spec_rad=1.1)
	RNN(params)

def test_build(tf_graph, mocker):
	rdm = rd.RDM(dt = 10, tau = 100, T = 2000, N_batch = 128)  
	gen = rdm.batch_generator()

	params = get_params()
	rnn = RNN(params)
	mocker.patch.object(RNN, 'forward_pass')
	RNN.forward_pass.return_value = tf.fill([params['N_batch'], params['N_steps'], params['N_out']], float('nan')), tf.fill([params['N_batch'], params['N_steps'], params['N_rec']], float('nan'))
	rnn.build()

def test_destruct(tf_graph, mocker):
	params = get_params()
	rnn1 = RNN(params)
	rnn1.destruct()
	rnn2 = RNN(params)
	mocker.patch.object(RNN, 'forward_pass')
	RNN.forward_pass.return_value = tf.fill([params['N_batch'], params['N_steps'], params['N_out']], float('nan')), tf.fill([params['N_batch'], params['N_steps'], params['N_rec']], float('nan'))
	rnn2.build()
	rnn2.destruct()
	rnn3 = RNN(params)

def test_recurrent_timestep(tf_graph):
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

def test_get_weights(tf_graph,mocker):
	params = get_params()
	rnn = RNN(params)
	with pytest.raises(UserWarning) as excinfo:
		rnn.get_weights()
	assert 'No weights to return yet -- model has not yet been initialized.' in str(excinfo.value)
	mocker.patch.object(RNN, 'forward_pass')
	RNN.forward_pass.return_value = tf.fill([params['N_batch'], params['N_steps'], params['N_out']], float('nan')), tf.fill([params['N_batch'], params['N_steps'], params['N_rec']], float('nan'))
	rnn.build()
	with pytest.raises(UserWarning) as excinfo:
		rnn.get_weights()
	assert 'No weights to return yet -- model has not yet been initialized.' in str(excinfo.value)
	rdm1 = rd.RDM(dt = params['dt'], tau = params['tau'], T = 2000, N_batch = params['N_batch'])  
	gen1 = rdm1.batch_generator()
	rnn.train(gen1)
	assert type(rnn.get_weights()) is dict


	#TODO(jasmine): also test once actual weights exist

def test_save(tf_graph):
	params = get_params()
	rnn = RNN(params)
	with pytest.raises(UserWarning) as excinfo:
		rnn.save("save_weights_path")
	#TODO(Jasmine): also test with actual weights

def test_train(tf_graph, mocker, capfd):
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

	rdm1 = rd.RDM(dt = params['dt'], tau = params['tau'], T = 2000, N_batch = params['N_batch'])  
	gen1 = rdm1.batch_generator()
	assert rnn.is_initialized is False
	rnn.train(gen1)
	assert rnn.is_initialized is True
	out, _ = capfd.readouterr()
	assert out != ""

def test_train_train_params_file_creation(tf_graph, mocker, tmpdir, capfd):
	params = get_params()

	rdm1 = rd.RDM(dt = params['dt'], tau = params['tau'], T = 2000, N_batch = params['N_batch'])  
	gen1 = rdm1.batch_generator()
	rdm2 = rd.RDM(dt = params['dt'], tau = params['tau'], T = 1000, N_batch = params['N_batch'])
	gen2 = rdm2.batch_generator()

	mocker.patch.object(RNN, 'forward_pass')
	RNN.forward_pass.return_value = tf.fill([params['N_batch'], params['N_steps'], params['N_out']], float('nan')), tf.fill([params['N_batch'], params['N_steps'], params['N_rec']], float('nan'))
	
	rnn =RNN(params)
	rnn.build()

	train_params = {}
	train_params['save_weights_path'] =  str(tmpdir.dirpath("save_weights.npz")) # Where to save the model after training. Default: None
	train_params['training_iters'] = 1000 # number of iterations to train for Default: 10000
	train_params['learning_rate'] = .01 # Sets learning rate if use default optimizer Default: .001
	train_params['loss_epoch'] = 20 # Compute and record loss every 'loss_epoch' epochs. Default: 10
	train_params['verbosity'] = False
	train_params['save_training_weights_epoch'] = 10 # save training weights every 'save_training_weights_epoch' epochs. Default: 100
	train_params['training_weights_path'] = str(tmpdir.dirpath("training_weights")) # where to save training weights as training progresses. Default: None
	train_params['generator_function'] = gen2 # replaces trial_batch_generator with the generator_function when not none. Default: None
	train_params['optimizer'] = tf.train.AdamOptimizer(learning_rate=train_params['learning_rate']) # What optimizer to use to compute gradients. Default: tf.train.AdamOptimizer(learning_rate=train_params['learning_rate'])
	train_params['clip_grads'] = False # If true, clip gradients by norm 1. Default: True
	
	assert not tmpdir.dirpath("save_weights.npz").check(exists=1)
	assert not tmpdir.dirpath("training_weights" + str(train_params['save_training_weights_epoch'])).check(exists=1)
	rnn.train(gen1, train_params)


	assert rnn.is_initialized is True
	out, _ = capfd.readouterr()
	print(out)
	assert out==""
	assert tmpdir.dirpath("save_weights.npz").check(exists=1)
	assert tmpdir.dirpath("training_weights" + str(train_params['save_training_weights_epoch'])+ ".npz").check(exists=1)



def test_test(mocker):
	rdm = rd.RDM(dt = 10, tau = 100, T = 2000, N_batch = 128)  
	gen = rdm.batch_generator()
	x,y,m = next(gen)

	params = get_params()
	rnn = RNN(params)
	with pytest.raises(UserWarning) as excinfo:
		rnn.test(x)
	assert 'build' in str(excinfo.value)
	
	mocker.patch.object(RNN, 'forward_pass')
	RNN.forward_pass.return_value = tf.fill([params['N_batch'], params['N_steps'], params['N_out']], float('nan')), tf.fill([params['N_batch'], params['N_steps'], params['N_rec']], float('nan'))
	rnn.build()

	rnn.test(x)



