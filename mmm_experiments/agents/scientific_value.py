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


def scientific_value_function(X, Y, sd=None, multiplier=1.0):
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

    Returns
    -------
    array_like
        The value for each data point.
    """

    X_dist = distance_matrix(X, X)

    if sd is None:
        # Automatic determination
        distance = X_dist.copy()
        distance[distance == 0.0] = np.inf
        sd = distance.min(axis=1).reshape(1, -1) * multiplier

    Y_dist = distance_matrix(Y, Y)

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

    def ask(
        self,
        optimize_acqf_kwargs={"q": 1, "num_restarts": 5, "raw_samples": 20},
    ):
        train_x = torch.tensor(self._positions_cache, dtype=torch.float)
        train_x = train_x.view(-1, self._in_dim)
        train_x = train_x.to(self._device)
        train_y = torch.tensor(self._observations_cache, dtype=torch.float)
        train_y = train_y.view(-1, self._out_dim)
        train_y = train_y.to(self._device)

        gp = SingleTaskGP(train_x, train_y).to(self.device)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(self.device)
        fit_gpytorch_model(mll)

        if optimize_acqf_kwargs["q"] == 1:
            acq = ExpectedImprovement(gp, best_f=torch.max(train_y).item())
        else:
            acq = qExpectedImprovement(gp, best_f=torch.max(train_y).item())

        next_points, acq_value = optimize_acqf(
            acq,
            bounds=self._bounds,
            **optimize_acqf_kwargs,
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
