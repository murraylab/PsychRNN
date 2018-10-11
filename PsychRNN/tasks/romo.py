from task import Task
import numpy as np

"""
    Romo, R., Brody, C. D., Hernández, A., & Lemus, L. (1999). Neuronal correlates of 
    parametric working memory in the prefrontal cortex. Nature, 399(6735), 470.

    https://www.nature.com/articles/20939
"""

class Romo(Task):

    def __init__(self, dt, tau, T, N_batch):
        super(RDM,self).__init__(2, 2, dt, tau, T, N_batch)

    def scale_p(self, f):
        return 0.4 + 0.8 * (f - 10) / (34 - 10)

    def scale_n(self, f):
        return 0.4 + 0.8 * (34 - f) / (34 - 10)

    frequency_pairs = [(18, 10), (22, 14), (26, 18), (30, 22), (34, 26)]
    decision_options = ['>', '<']

    lo = 0.2
    hi = 1.0

    def generate_trial_params(self, batch, trial):
        params = dict()

        onset_time = np.random.uniform(0, 1) * self.T / 8.0
        stim_duration_1 = np.random.uniform(0, 1) * self.T / 4.0
        delay_duration = np.random.uniform(0, 1) * self.T / 4.0
        stim_duration_2 = np.random.uniform(0, 1) * self.T / 4.0
        decision_duration = np.random.uniform(0, 1) * self.T / 8.0

        params['stimulus_1'] = onset_time
        params['delay'] = onset_time + stim_duration_1
        params['stimulus_2'] = onset_time + stim_duration_1 + delay_duration
        params['decision'] = onset_time + stim_duration_1 + delay_duration + stim_duration_2
        params['end'] = onset_time + stim_duration_1 + delay_duration + stim_duration_2 + decision_duration

        params['stim_noise'] = 0.1

        fpair = np.random.choice(len(self.frequency_pairs))
        gt_lt = np.random.choice(self.decision_options)
        if gt_lt == '>':
            f1, f2 = fpair
            choice = 0
        else:
            f2, f1 = fpair
            choice = 1
        params['f1'] = f1
        params['f2'] = f2
        params['choice'] = choice

        return params

    def trial_function(self, t, params):
        # ----------------------------------
        # Initialize with noise
        # ----------------------------------
        x_t = np.sqrt(2*self.alpha*params['stim_noise']*params['stim_noise'])*np.random.randn(self.N_in)
        y_t = np.zeros(self.N_out)
        mask_t = np.ones(self.N_out)

        # ----------------------------------
        # Retrieve parameters
        # ----------------------------------
        stimulus_1 = params['stimulus_1']
        delay = params['delay']
        stimulus_2 = params['stimulus_2']
        decision = params['decision']
        end = params['end']

        f1 = params['f1']
        f2 = params['f2']
        choice = params['choice']

        # ----------------------------------
        # Compute values
        # ----------------------------------
        if 0 <= t < stimulus_1:
            mask_t = np.zeros(self.N_out)

        if stimulus_1 <= t < delay:
            x_t[0] = self.scale_p(f1)
            x_t[1] = self.scale_n(f1)
            mask_t = np.zeros(self.N_out)

        if delay <= t < stimulus_2:
            mask_t = np.zeros(self.N_out)

        if stimulus_2 <= t < decision:
            x_t[0] = self.scale_p(f2)
            x_t[1] = self.scale_n(f2)
            mask_t = np.zeros(self.N_out)

        if decision <= t < end:
            y_t[choice] = self.hi
            y_t[1-choice] = self.lo

        if end <= t:
            mask_t = np.zeros(self.N_out)

        return x_t, y_t, mask_t
