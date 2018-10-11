from task import Task
import numpy as np


class FlipFlop(Task):

    def generate_trial_params(self, batch, trial):
        params = dict()

        n_turns = np.random.choice([1, 2, 3, 4, 5])

        onset_time = np.random.uniform(0, 1) * self.T / (2 * float(n_turns + 1))
        stim_duration = np.random.uniform(0, 1) * self.T / (2 * float(n_turns + 1))
        echo_duration = np.random.uniform(0, 1) * self.T / (2 * float(n_turns + 1))

        params['onset_time'] = onset_time
        params['stim_duration'] = stim_duration
        params['echo_duration'] = echo_duration

        params['stim_noise'] = 0.1

        turn_time = stim_duration + echo_duration
        input_times = [onset_time + i * turn_time for i in range(n_turns)]
        echo_times = [onset_time + i * turn_time + stim_duration for i in range(n_turns)]

        params['input_times'] = input_times
        params['echo_times'] = echo_times

        params['end'] = echo_times[-1] + echo_duration

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
        onset_time = params["onset_time"]
        input_times = params["input_times"]
        echo_times = params["echo_times"]
        echo_duration = params['echo_duration']

        if 0 <= t < onset_time:
            mask_t = np.zeros(self.N_out)

        for i, (input_start, echo_start) in enumerate(zip(input_times, echo_times)):

            if input_start <= t < input_start + echo_start:
                x_t = 1.0
                mask_t = np.zeros(self.N_out)

            elif echo_start <= t < echo_start + echo_duration:
                y_t = 1.0

        if echo_times[-1] + echo_duration <= t:
            mask_t = np.zeros(self.N_out)

        return x_t, y_t, mask_t


