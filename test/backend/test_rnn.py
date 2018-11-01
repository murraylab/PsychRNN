import pytest
from psychrnn.backend.rnn import RNN

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

def test_rnn():
	# params = {}
	# with pytest.raises(KeyError) as excinfo:
	# 	RNN(params)
	params = get_params()
	RNN(params)

	# test throws errors if insufficient info given
	

def test_build():
	pass

def test_destruct():
	pass

def test_recurrent_timestep():
	pass

def test_output_timestep():
	pass

def test_forward_pass():
	pass

def test_save():
	pass

def test_train():
	pass

def test_test():
	pass


