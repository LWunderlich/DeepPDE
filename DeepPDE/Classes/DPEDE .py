import numpy as np

from tensorflow import keras
from scipy.stats import norm
from numpy.polynomial.hermite import hermgauss

np.random.seed(42)


class DPDEGenerator(keras.utils.Sequence):
    """ Create batches of random points for the network training. """

    def __init__(self, batch_size, normalised_min, normalised_max, dimension):
        """ Initialise the generator by saving the batch size. """
        self.batch_size = batch_size
        self.normalised_min = normalised_min
        self.normalised_max = normalised_max
        self.dimension_state = dimension[0]
        self.dimension_parameter = dimension[1]
        self.dimension_total = dimension[2]

    def __len__(self):
        """ Describes the number of points to create """
        return self.batch_size

    def __getitem__(self, idx):
        """ Get one batch of random points in the interior of the domain to 
        train the PDE residual and with initial time to train the initial value.
        """
        data_train_interior = np.random.uniform(self.normalised_min, self.normalised_max,
                                                [self.batch_size, self.dimension_total])

        t_train_initial = self.normalised_min * np.ones((self.batch_size, 1))
        s_and_p_train_initial = np.random.uniform(self.normalised_min, self.normalised_max,
                                                  [self.batch_size, self.dimension_state + self.dimension_parameter])

        data_train_initial = np.concatenate((t_train_initial, s_and_p_train_initial), axis=1)

        return [data_train_interior, data_train_initial]
