import numpy as np
import tensorflow as tf


class WeightInitializer(object):

    def __init__(self, **kwargs):

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
            # Default initializations
            # ----------------------------------
            self.initializations['W_in'] = .2 * np.random.rand(N_rec, N_in) - .1
            self.initializations['W_out'] = .2 * np.random.rand(N_out, N_rec) - .1
            self.initializations['W_rec'] = np.random.randn(N_rec, N_rec)

            self.initializations['b_rec'] = np.zeros(N_rec)
            self.initializations['b_out'] = np.zeros(N_out)

            self.initializations['init_state'] = .1 + .01 * np.random.randn(N_rec)

            self.initializations['input_connectivity'] = np.ones([N_rec, N_in])
            self.initializations['rec_connectivity'] = np.ones([N_rec, N_rec])
            self.initializations['output_connectivity'] = np.ones([N_out, N_rec])

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

        self.spec_rad = kwargs.get('spec_rad')

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


class FeedForward(WeightInitializer):
    '''Generate random feedforward wrec (lower triangular)'''

    def __init__(self, **kwargs):

        super(FeedForward, self).__init__(**kwargs)

        self.sigma = kwargs.get('sigma')

        self.initializations['W_rec'] = np.tril(self.sigma * np.random.randn(self.N_rec, self.N_rec), -1)

        if not self.autapses:
            self.initializations['W_rec'][np.eye(self.N_rec) == 1] = 0

        return


class StrictFeedForward(WeightInitializer):

    def __init__(self, **kwargs):

        super(StrictFeedForward, self).__init__(**kwargs)

        self.alpha = kwargs.get('alpha')

        W_rec = np.zeros([self.N_rec, self.N_rec])
        for ii in range(self.N_rec-1):
            W_rec[ii+1,ii] = self.alpha
        self.initializations['W_rec'] = W_rec

        if not self.autapses:
            self.initializations['W_rec'][np.eye(self.N_rec)==1] = 0


class ZeroMatrix(WeightInitializer):
     '''Generate wrec of all zeros'''

     def __init__(self, **kwargs):

        super(ZeroMatrix, self).__init__(**kwargs)

        self.initializations['W_rec'] = np.zeros([self.N_rec, self.N_rec])



class BlockFeedForward(WeightInitializer):

    def __init__(self, **kwargs):
        super(BlockFeedForward, self).__init__(**kwargs)

        self.spec_rad = kwargs.get('spec_rad')

        W_rec = np.random.randn(self.N_rec, self.N_rec)
        W_rec = self.spec_rad * W_rec / np.max(np.abs(np.linalg.eig(W_rec)[0]))
        W_rec[:self.N_rec / 2, self.N_rec / 2:] = 0
        self.initializations['W_rec'] = W_rec

        input_connectivity = np.ones([self.N_rec, self.N_in])
        input_connectivity[self.N_rec / 2:, :] = 0
        self.initializations['input_connectivity'] = input_connectivity

        rec_connectivity = np.ones([self.N_rec, self.N_rec])
        rec_connectivity[:self.N_rec / 2, self.N_rec / 2:] = 0
        self.initializations['rec_connectivity'] = rec_connectivity

        output_connectivity = np.ones([self.N_out, self.N_rec])
        output_connectivity[:, :self.N_rec / 2] = 0
        self.initializations['output_connectivity'] = output_connectivity

        if not self.autapses:
            self.initializations['W_rec'][np.eye(self.N_rec) == 1] = 0
            self.initializations['rec_connectivity'][np.eye(self.N_rec) == 1] = 0


class FlipFlops(WeightInitializer):

    def __init__(self, **kwargs):
        super(FlipFlops, self).__init__(**kwargs)

        assert np.mod(self.N_rec, 2) == 0, "N_rec must be even"

        self.initializations['b_rec'] = .5 * np.ones(self.N_rec)

        # flip flop circuit
        w11 = w22 = .5
        w12 = w21 = -2.
        W = np.array([[w11, w12], [w21, w22]])

        W_rec = np.zeros([self.N_rec, self.N_rec])
        for ii in range(0, self.N_rec, 2):
            W_rec[ii:ii + 2, ii:ii + 2] = W
        self.initializations['W_rec'] = W_rec

        if not self.autapses:
            self.initializations['W_rec'][np.eye(self.N_rec) == 1] = 0


class HalfFlipFlops(WeightInitializer):

    def __init__(self, **kwargs):
        super(HalfFlipFlops, self).__init__(**kwargs)

        assert np.mod(self.N_rec, 2) == 0, "N_rec must be even"

        self.initializations['b_rec'] = .5 * np.ones(self.N_rec)

        # flip flop circuit
        w11 = w22 = .5
        w12 = w21 = -2.
        W = np.array([[w11, w12], [w21, w22]])

        W_rec = np.zeros([self.N_rec, self.N_rec])
        for ii in range(0, self.N_rec//2, 2):
            W_rec[ii:ii + 2, ii:ii + 2] = W

        W_rec[self.N_rec // 2:, self.N_rec // 2:] = np.linalg.qr(
                                    np.random.randn(self.N_rec // 2, self.N_rec // 2))[0]

        self.initializations['W_rec'] = W_rec

        if not self.autapses:
            self.initializations['W_rec'][np.eye(self.N_rec) == 1] = 0