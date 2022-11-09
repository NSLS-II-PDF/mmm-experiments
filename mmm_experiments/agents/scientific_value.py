from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from botorch.acquisition import UpperConfidenceBound, qUpperConfidenceBound
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.spatial import distance_matrix


def next_closest_raster_scan_point(proposed_points, observed_points, possible_coordinates, eps=1e-8):
    """A helper function which determines the closest grid point for every
    proposed points, under the constraint that the proposed point is not
    present in the currently observed points, given possible coordinates.

    Parameters
    ----------
    proposed_points : array_like
        The proposed points. Should be of shape N x d, where d is the dimension
        of the space (e.g. 2-dimensional for a 2d raster). N is the number of
        proposed points (i.e. the batch size).
    observed_points : array_like
        Points that have been previously observed. N1 x d, where N1 is the
        number of previously observed points.
    possible_coordinates : array_like
        A grid of possible coordinates, options to choose from. N2 x d, where
        N2 is the number of coordinates on the grid.
    eps : float, optional
        The cutoff for determining that two points are the same, as computed
        by the L2 norm via scipy's ``distance_matrix``.

    Returns
    -------
    numpy.ndarray
        The new proposed points.
    """

    # If there are no observed points, put a point at infinity just to make
    # the calculations work the same way. Kinda lazy but this is really fast
    # and works just fine.
    if observed_points is None or len(observed_points) == 0:
        observed_points = np.ones(shape=(1, possible_coordinates.shape[1]))
        observed_points = observed_points * np.inf

    assert proposed_points.shape[1] == observed_points.shape[1]
    assert proposed_points.shape[1] == possible_coordinates.shape[1]

    D2 = distance_matrix(observed_points, possible_coordinates) > eps
    D2 = np.all(D2, axis=0)

    actual_points = []
    for possible_point in proposed_points:
        p = possible_point.reshape(1, -1)
        D = distance_matrix(p, possible_coordinates).squeeze()
        argsorted = np.argsort(D)
        for index in argsorted:
            if D2[index]:
                actual_points.append(possible_coordinates[index])
                break

    return np.array(actual_points)


def scientific_value_function(X, Y, sd=None, multiplier=1.0, y_distance_function=None):
    """The value of two datasets, X and Y. Both X and Y must have the same
    number of rows. The returned result is a value of value for each of the
    data points.

    Parameters
    ----------
    X : numpy.ndarray
        The input data of shape N x d.
    Y : numpy.ndarray
        The output data of shape N x d'. Note that d and d' can be different
        and they also do not have to be 1.
    sd : float, optional
        Controls the length scale decay. We recommend this be set to ``None``
        to allow for automatic detection of the decay length scale(s).
    multiplier : float, optional
        Multiplies the automatically derived length scale if ``sd`` is
        ``None``.
    y_distance_function : callable, optional
        A callable function which takes the array ``Y`` as input and returns
        an N x N array in which the ith row and jth column is the distance
        measure between points i and j. Defaults to
        ``scipy.spatial.distance_matrix`` with its default kwargs (i.e. it is
        the L2 norm).

    Returns
    -------
    array_like
        The value for each data point.
    """

    X_dist = distance_matrix(X, X)

    if sd is None:
        distance = X_dist.copy()
        distance[distance == 0.0] = np.inf
        sd = distance.min(axis=1).reshape(1, -1) * multiplier

    # We can make this more pythonic but it makes sense in this case to keep
    # the default behavior explicit
    if y_distance_function is None:
        Y_dist = distance_matrix(Y, Y)
    else:
        Y_dist = y_distance_function(Y)

    v = Y_dist * np.exp(-(X_dist**2) / sd**2 / 2.0)

    return v.mean(axis=1)


class ScientificValueAgentMixin:
    """This is a jack-of-all-trades agent which is general to any type of
    observation (i.e. any beamline). This is a _parameter free_ method, meaning
    that the user does not need to set any parameters explicitly, though one
    can manually set a length scale if desired.

    Assumes that ``relative_bounds`` and ``measurement_origin`` are
    defined as attributes.
    """

    def __init__(
        self,
        *,
        device: Union[torch.device, str],
        beta=20.0,
        length_scale: Optional[float] = None,
        y_distance_function: Optional[callable] = None,
        optimize_acqf_kwargs: Optional[dict] = None,
        possible_coordinates: Optional[np.ndarray] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._beta = beta
        self.device = device
        self._length_scale = length_scale
        self._y_distance_function = y_distance_function
        self._optimize_acqf_kwargs = self.default_acqf_kwargs()
        self._optimize_acqf_kwargs.update({} if optimize_acqf_kwargs is None else optimize_acqf_kwargs)

        # Possible coordinatse should be an array of shape L x d, where L is
        # the number of possible coordinates and d is the dimension of the
        # space (i.e. it's 2 if we have a 2-dimensional scanning surface)
        self._possible_coordinates = possible_coordinates

        # Caches for the results
        self._observations_cache = []
        self._positions_cache = []
        self._relative_positions_cache = []
        self._value_cache = []

    @staticmethod
    def default_acqf_kwargs():
        return {"num_restarts": 5, "raw_samples": 20}

    def _value_function(self, X, Y):
        return scientific_value_function(X, Y, sd=self._length_scale)

    def update_acqf_kwargs(self, **kwargs):
        """User exposed function to update acquisition function kwargs"""
        self._optimize_acqf_kwargs.update(kwargs)

    def tell(self, position, observation):
        """Takes the position of the motor and an arbitrary observation which
        is a function of that position, and appends the proper caches.

        Parameters
        ----------
        position : array_like
            The position of the motor. Must match the dimensionality of that
            provided in the ``__init__``.
        observation : array_like
            The observation, such as a spectrum, PDF, XRD, etc. Must match the
            dimensionality of that provided in the ``__init__``.

        Returns
        -------
        dict
        """

        # Using the relative position for all learning
        relative_position = position - self.measurement_origin

        self._positions_cache.append(position)
        self._relative_positions_cache.append(relative_position)
        self._observations_cache.append(observation)

        return dict(
            position=[position],
            rel_position=[relative_position],
            observation=[observation],
            cache_len=[len(self._relative_positions_cache)],
        )

    def report(self):
        # The value is a scalar
        value = self._value_function(
            np.array(self._relative_positions_cache).reshape(-1, 1), np.stack(self._observations_cache)
        )
        value = value.reshape(-1, 1)
        return dict(
            position=[self._positions_cache],
            rel_position=[self._relative_positions_cache],
            observation=[self._observations_cache],
            cache_len=[len(self._relative_positions_cache)],
            value=[value.squeeze()],
        )

    def ask(self, batch_size: int = 1) -> Tuple[dict, Sequence]:

        value = self._value_function(
            np.array(self._relative_positions_cache).reshape(-1, 1), np.stack(self._observations_cache)
        )
        value = value.reshape(-1, 1)

        train_x = torch.tensor(self._relative_positions_cache, dtype=torch.float).view(-1, 1)
        train_x = train_x.to(self.device)
        train_y = torch.tensor(value, dtype=torch.float)
        train_y = train_y.to(self.device)

        gp = SingleTaskGP(train_x, train_y).to(self.device)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(self.device)
        fit_gpytorch_model(mll)
        acq = (
            UpperConfidenceBound(gp, beta=self._beta)
            if batch_size == 1
            else qUpperConfidenceBound(gp, beta=self._beta)
        )

        next_points, acq_value = optimize_acqf(
            acq,
            bounds=torch.tensor(self.relative_bounds, dtype=torch.float).view(2, 1),
            **self._optimize_acqf_kwargs,
        )

        if self._possible_coordinates is not None:
            next_points = next_closest_raster_scan_point(
                next_points, train_x.detach().numpy(), self._possible_coordinates, eps=1e-8
            )

        if batch_size == 1:
            next_points = [float(next_points.to("cpu"))]
        else:
            next_points = [float(x.to("cpu")) for x in next_points]

        doc = dict(
            batch_size=[batch_size],
            next_points=[next_points],
            acq_value=[float(x.to("cpu")) for x in acq_value] if batch_size > 1 else [float(acq_value.to("cpu"))],
        )
        return doc, next_points
