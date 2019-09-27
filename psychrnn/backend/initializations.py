from __future__ import division

import numpy as np
import tensorflow as tf


class WeightInitializer(object):

    def __init__(self, **kwargs):

        print(kwargs)
        # ----------------------------------
        # Required parameters
        # ----------------------------------
        self.load_weights_path = kwargs.get('load_weights_path', None)
        N_in = self.N_in = kwargs.get('N_in')
        N_rec = self.N_rec = kwargs.get('N_rec')
        N_out = self.N_out = kwargs.get('N_out')
        self.autapses = kwargs.get('autapses', True)

        self.initializations = dict()

        if self.load_weights_path is not None:
            # ----------------------------------
            # Load saved weights
            # ----------------------------------
            self.initializations = np.load(self.load_weights_path)
        else:
            # ----------------------------------
            # Default initializations / optional loading from params
            # ----------------------------------
            self.initializations['W_in'] = kwargs.get('W_in', .2 * np.random.rand(N_rec, N_in) - .1)
            assert(self.initializations['W_in'].shape == (N_rec, N_in))
            self.initializations['W_out'] = kwargs.get('W_out', .2 * np.random.rand(N_out, N_rec) - .1)
            assert(self.initializations['W_out'].shape == (N_out, N_rec))
            self.initializations['W_rec'] = kwargs.get('W_rec', np.random.randn(N_rec, N_rec))
            assert(self.initializations['W_rec'].shape == (N_rec, N_rec))

            self.initializations['b_rec'] = kwargs.get('b_rec',np.zeros(N_rec))
            assert(self.initializations['b_rec'].shape == (N_rec,))
            self.initializations['b_out'] = kwargs.get('b_out',np.zeros(N_out))
            assert(self.initializations['b_out'].shape == (N_out,))

            self.initializations['init_state'] = kwargs.get('init_state', .1 + .01 * np.random.randn(N_rec))
            assert(self.initializations['init_state'].shape == (N_rec,))

            self.initializations['input_connectivity'] = kwargs.get('input_connectivity',np.ones([N_rec, N_in]))
            assert(self.initializations['input_connectivity'].shape == (N_rec, N_in))
            self.initializations['rec_connectivity'] = kwargs.get('rec_connectivity',np.ones([N_rec, N_rec]))
            assert(self.initializations['rec_connectivity'].shape == (N_rec, N_rec))
            self.initializations['output_connectivity'] = kwargs.get('output_connectivity', np.ones([N_out, N_rec]))
            assert(self.initializations['output_connectivity'].shape == (N_out, N_rec))

            if not self.autapses:
                self.initializations['W_rec'][np.eye(N_rec) == 1] = 0
                self.initializations['rec_connectivity'][np.eye(N_rec) == 1] = 0

        return

    def get(self, tensor_name):

        return tf.constant_initializer(self.initializations[tensor_name])

    def save(self, save_path):

        np.savez(save_path, **self.initializations)
        return







class GaussianSpectralRadius(WeightInitializer):
    '''Generate random gaussian weights with specified spectral radius'''

    def __init__(self, **kwargs):

        super(GaussianSpectralRadius, self).__init__(**kwargs)

        self.spec_rad = kwargs.get('spec_rad', 1.1)

        W_rec = np.random.randn(self.N_rec, self.N_rec)
        self.initializations['W_rec'] = self.spec_rad * W_rec / np.max(np.abs(np.linalg.eig(W_rec)[0]))

        if not self.autapses:
            self.initializations['W_rec'][np.eye(self.N_rec) == 1] = 0

        return



class AlphaIdentity(WeightInitializer):
    '''Generate recurrent weights w(i,i) = alpha, w(i,j) = 0'''

    def __init__(self, **kwargs):

        super(AlphaIdentity, self).__init__(**kwargs)

        self.alpha = kwargs.get('alpha')

        self.initializations['W_rec'] = np.eye(self.N_rec) * self.alpha

        if not self.autapses:
            self.initializations['W_rec'][np.eye(self.N_rec) == 1] = 0

        return
