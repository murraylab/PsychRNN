# PsychRNN
[![Build Status](https://api.travis-ci.com/dbehrlich/PsychRNN.svg?branch=master)](https://api.travis-ci.com/dbehrlich/PsychRNN)

This package is intended to help cognitive scientist easily translate task designs from human or primate behavioral experiments into a form capable of being used as training data for a recurrent neural network.


We have isolated the front-end task design, in which users can intuitively describe the conditional logic of their task from the backend where gradient descent based optimization occurs. This is intended to facilitate researchers who might otherwise not have an easy implementation available to design and test hypothesis regarding the behavior of recurrent neural networks in different task environements.


Code is written and upkept by: @davidbrandfonbrener @dbehrlic @ABAtanasov @syncrostone 

## Install

### Dependencies

- Numpy
- Tensorflow
- Python=2.7 or 3.6

For Demos:
- Jupyter
- Ipython
- Matplotlib

### Installation

git clone https://github.com/dbehrlich/PsychRNN.git  
cd PsychRNN   
python setup.py install

#### Alternative Install

pip install PsychRNN


## 17 Lines Introduction

A minimal introduction to our package. In this simple introduction you can generate a new recurrent neural network model, train that model on the random dot motion discrimination task, and plot out an example output in just 17 lines.

	import psychrnn  
	from psychrnn.tasks import rdm as rd  
	from psychrnn.backend.models.basic import Basic  
	import tensorflow as tf  

	from matplotlib import pyplot as plt  
	%matplotlib inline

	rdm = rd.RDM(dt = 10, tau = 100, T = 2000, N_batch = 128)  
	gen = rdm.batch_generator()

	params = rdm.__dict__  
	params['name'] = 'model'  
	params['N_rec'] = 50  

	model = Basic(params)  
	model.build()  
	model.train(gen)

	x,_,_ = next(gen)

	plt.plot(model.test(x)[0][0,:,:])

	model.destruct()

Code for this example can be found in "Minimal_Example.ipynb"

## Demonstration Notebook

For a more complete tour of training and model parameters see the "RDM.ipynb" notebook.


## Writing a New Task

You can easily begin running your own tasks by writing a new task subclass with the two functions (generate_trial_params, trial_function) specified below, or by modifying one of our existing task files such as "rdm.py" or "romo.py".

	Class your_new_class(Task):

		def __init__(self, N_in, N_out, dt, tau, T, N_batch):

			super(RDM,self).__init__(N_in, N_out, dt, tau, T, N_batch)

				'''

				Args:
					N_in: number of network inputs
					N_out: number of network output
					dt: simulation time step
					tau: unit time constant
					T: trial length
					N_batch: number of trials per training update

				'''

		def generate_trial_params(self,batch,trial):

			''' function that produces trial specific params for your task (e.g. coherence for the 
				random dot motion discrimination task)

			Args:
				batch: # of batch for training (for internal use)
				trial: # of trial within a batch (for internal use)

			Returns:
				params: A dictionary of necessary params for trial_function

				'''

		def trial_function(self,t,params):

			'''function that specifies conditional network input, target output and loss mask for your task at a given time (e.g. if t>stim_onset x_t=1).

			Args:
				t: time
				params: params dictionary from generate_trial_params

			Returns:
				x_t: input vector of length N_in at time t
				y_t: target output vector of length N_out at time t
				mask_t: loss function mask vector of length N_out at time t

				'''

## Building a New Model


New models can be added by extending the RNN superclass, as in our examples of "basic.py" and "lstm.py". Each new model class requires three functions (recurrent_timestep, output time_step and forward_pass).

	Class your_new_model(RNN):

		def recurrent_timestep(self, rnn_in, state):

			'''function that updates the recurrent state of your network one timestep

			Args:
				rnn_in: network input vector of length N_in at t
				state: network state at t

			Returns:
				new_state: network state at t+1

				'''

		def output_timestep(self, state):

			'''function that produces output for the current state of your network at one timestep

			Args:
				state: network state at t

			Returns:
				output: output vector of length N_out at t

				'''

		def forward_pass(self):

			'''function that contains the loop of calls to recurrent_timestep and output_timestep
			to run the evolution of your state through a trial 


				'''


## Further Extensibility

If you wish to modify weight initializations, loss functions or regularizations it is as simple as adding an additional class to "initializations.py" describing your preferred initial weight patterns or a single function to "loss_functions.py" or "regularizations.py".

### Backend

- initializations
- loss_functions
- regularizations
- rnn
- simulation

