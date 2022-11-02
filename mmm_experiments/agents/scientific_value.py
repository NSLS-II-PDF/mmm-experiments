from abc import ABC
from typing import Optional

import numpy as np
import torch
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.spatial import distance_matrix

from .base import Agent


def next_closest_raster_scan_point(
    proposed_points,
    observed_points,
    possible_coordinates,
    eps=1e-8
):
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


def scientific_value_function(
    X,
    Y,
    sd=None,
    multiplier=1.0,
    y_distance_function=None
):
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


class ScientificValueAgent(Agent, ABC):
    """This is a jack of all trades agent which is general to any type of
    observation (i.e. any beamline). This is a _parameter free_ method, meaning
    that the user does not need to set any parameters explicitly, though one
    can manually set a length scale if desired."""

    def __init__(
        self,
        bounds: list,
        in_dim: int,
        out_dim: int,
        beamline_tla: str,
        length_scale: Optional[float] = None,
        metadata: Optional[dict] = None,
        y_distance_function: Optional[callable] = None,
        optimize_acqf_kwargs: dict = {
            "q": 1, "num_restarts": 5, "raw_samples": 20
        },
        possible_coordinates: Optional[np.ndarray] = None,
        device="cpu",
    ):
        super().__init__(beamline_tla=beamline_tla, metadata=metadata)
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._length_scale = length_scale
        self._device = device
        self._observations_cache = []
        self._positions_cache = []
        self._value_cache = []
        self._bounds = torch.tensor(bounds).to(self._device).float()
        self._y_distance_function = y_distance_function
        self._optimize_acqf_kwargs = optimize_acqf_kwargs

        # Possible coordinatse should be an array of shape L x d, where L is
        # the number of possible coordinates and d is the dimension of the
        # space (i.e. it's 2 if we have a 2-dimensional scanning surface)
        self._possible_coordinates = possible_coordinates

    def tell(self, position, observation):
        """Takes the position of the motor and an arbitrary observation which
        is a function of that position, computes the value of this site, and
        appends the proper caches.

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

        self._positions_cache.append(position)
        self._observations_cache.append(observation)

        # Compute the value
        X = np.array(self._positions_cache).reshape(-1, self._in_dim)
        Y = np.array(self._observations_cache).reshape(-1, self._out_dim)
        V = scientific_value_function(X, Y, sd=self._length_scale)

        # The value is a scalar
        V = V.reshape(-1, 1)

        return dict(position=position, observation=observation, value=V.squeeze()[-1])

    def ask(self, optimize_acqf_kwargs=None):
        train_x = torch.tensor(self._positions_cache, dtype=torch.float)
        train_x = train_x.view(-1, self._in_dim)
        train_x = train_x.to(self._device)
        train_y = torch.tensor(self._observations_cache, dtype=torch.float)
        train_y = train_y.view(-1, self._out_dim)
        train_y = train_y.to(self._device)

        gp = SingleTaskGP(train_x, train_y).to(self.device)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(self.device)
        fit_gpytorch_model(mll)

        if optimize_acqf_kwargs is None:
            optimize_acqf_kwargs = self._optimize_acqf_kwargs

        if optimize_acqf_kwargs["q"] == 1:
            acq = ExpectedImprovement(gp, best_f=torch.max(train_y).item())
        else:
            acq = qExpectedImprovement(gp, best_f=torch.max(train_y).item())

        next_points, acq_value = optimize_acqf(
            acq,
            bounds=self._bounds,
            **optimize_acqf_kwargs,
        )

        if self._possible_coordinates is not None:
            next_points = next_closest_raster_scan_point(
                next_points,
                train_x.detach().numpy(),
                self._possible_coordinates,
                eps=1e-8
            )

        if optimize_acqf_kwargs["q"] == 1:
            next_points = [float(next_points.to("cpu"))]
        else:
            next_points = [float(x.to("cpu")) for x in next_points]

        doc = dict(
            batch_size=optimize_acqf_kwargs["q"],
            next_points=next_points,
            acq_value=[float(x.to("cpu")) for x in acq_value]
            if optimize_acqf_kwargs["q"] > 1
            else [float(acq_value.to("cpu"))],
        )
        return doc, next_points
