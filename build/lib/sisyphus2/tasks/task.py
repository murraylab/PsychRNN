import numpy as np


class Task(object):
    def __init__(self, N_in, N_out, dt, tau, T, N_batch):

        # ----------------------------------
        # Initialize required parameters
        # ----------------------------------
        self.N_batch = N_batch
        self.N_in = N_in
        self.N_out = N_out
        self.dt = dt
        self.tau = tau
        self.T = T

        # ----------------------------------
        # Calculate implied parameters
        # ----------------------------------
        self.alpha = (1.0 * self.dt) / self.tau
        self.N_steps = int(np.ceil(self.T / self.dt))

    def generate_trial_params(self, batch, trial):
        pass

    def trial_function(self, time, params):
        pass

    def generate_trial(self, params):

        # ----------------------------------
        # Loop to generate a single trial
        # ----------------------------------
        x_data = np.zeros([self.N_steps, self.N_in])
        y_data = np.zeros([self.N_steps, self.N_out])
        mask = np.zeros([self.N_steps, self.N_out])

        for t in range(self.N_steps):
            x_data[t, :], y_data[t, :], mask[t, :] = self.trial_function(t * self.dt, params)

        return x_data, y_data, mask

    def batch_generator(self):
        batch = 1
        while batch > 0:

            x_data = []
            y_data = []
            mask = []
            # ----------------------------------
            # Loop over trials in batch
            # ----------------------------------
            for trial in range(self.N_batch):
                # ---------------------------------------
                # Generate each trial based on its params
                # ---------------------------------------
                x,y,m = self.generate_trial(self.generate_trial_params(batch, trial))
                x_data.append(x)
                y_data.append(y)
                mask.append(m)

            batch += 1

            yield np.array(x_data), np.array(y_data), np.array(mask)

