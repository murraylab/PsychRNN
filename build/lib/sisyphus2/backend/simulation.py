import numpy as np


class Simulator(object):
    def __init__(self,  params, weights_path=None, weights=None):
        # ----------------------------------
        # Extract params
        # ----------------------------------
        N_in = self.N_in = params['N_in']
        N_rec = self.N_rec = params['N_rec']
        N_out = self.N_out = params['N_out']

        self.dt = params['dt']
        self.tau = params['tau']
        self.alpha = (1.0* self.dt) / self.tau
        self.dale_ratio = params['dale_ratio']
        self.rec_noise = params['rec_noise']

        # ----------------------------------
        # Load Dale matrix
        # ----------------------------------
        dale_vec = np.ones(N_rec)
        if self.dale_ratio:
            dale_vec[int(self.dale_ratio * N_rec):] = -1
            self.dale_rec = np.diag(dale_vec)
            dale_vec[int(self.dale_ratio * N_rec):] = 0
            self.dale_out = np.diag(dale_vec)
        else:
            self.dale_rec = np.diag(dale_vec)
            self.dale_out = np.diag(dale_vec)

        # ----------------------------------
        # Initialize weights
        # ----------------------------------
        if weights_path is not None:
            weights = np.load(weights_path)
        self.W_in = weights['W_in'] * weights['input_Connectivity']
        self.W_rec = weights['W_rec'] * weights['rec_Connectivity']
        self.W_out = weights['W_out'] * weights['output_Connectivity']

        self.b_rec = weights['b_rec']
        self.b_out = weights['b_out']

        self.init_state = weights['init_state']

    # ----------------------------------------------
    # t_connectivity allows for ablation experiments
    # ----------------------------------------------
    def rnn_step(self, state, rnn_in, t_connectivity, use_input):
        pass

    def run_trials(self, trial_input, t_connectivity=None, use_input=True):
        pass


class BasicSimulator(Simulator):

    def rnn_step(self, state, rnn_in, t_connectivity, use_input):
        if self.dale_ratio:
            new_state = (1-self.alpha) * state \
                        + self.alpha * (np.matmul(np.maximum(state, np.zeros(state.shape)),
                            np.transpose(np.matmul(np.absolute(self.W_rec) * t_connectivity, self.dale_rec)))
                            + self.b_rec)\
                        + np.sqrt(2.0 * self.alpha * self.rec_noise * self.rec_noise) * \
                        np.random.normal(loc=0.0, scale=1.0, size=state.shape)
            if use_input:
                new_state += self.alpha * np.matmul(rnn_in, np.transpose(np.absolute(self.W_in)))

            new_output = np.matmul(
                            np.maximum(new_state, np.zeros(state.shape)),
                            np.transpose(np.matmul(
                                np.absolute(self.W_out),
                                self.dale_out))) + self.b_out
                        
        else:
            new_state = (1-self.alpha) * state \
                        + self.alpha * (np.matmul(np.maximum(state, np.zeros(state.shape)),
                                np.transpose(self.W_rec * t_connectivity))
                            + self.b_rec)\
                        + np.sqrt(2.0 * self.alpha * self.rec_noise * self.rec_noise) * \
                          np.random.normal(loc=0.0, scale=1.0, size=state.shape)
            if use_input:
                new_state += self.alpha * np.matmul(rnn_in, np.transpose(self.W_in))

            new_output = np.matmul(
                            np.maximum(new_state, np.zeros(state.shape)),
                            np.transpose(self.W_out)) + self.b_out

        return new_output, new_state

    def run_trials(self, trial_input, t_connectivity=None, use_input=True):

        batch_size = trial_input.shape[0]
        rnn_inputs = np.squeeze(np.split(trial_input, trial_input.shape[1], axis=1))
        state = np.expand_dims(self.init_state[0, :], 0)
        state = np.repeat(state, batch_size, 0)
        rnn_outputs = []
        rnn_states = []
        for i, rnn_input in enumerate(rnn_inputs):
            if t_connectivity is not None:
                output, state = self.rnn_step(state, rnn_input, t_connectivity[i], use_input)
            else:
                output, state = self.rnn_step(state, rnn_input, np.ones_like(self.W_rec), use_input)

            rnn_outputs.append(output)
            rnn_states.append(state)

        return np.swapaxes(np.array(rnn_outputs), 0, 1), np.swapaxes(np.array(rnn_states), 0, 1)
