from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Union

import requests


class Agent(ABC):
    """
    Single Plan Agent. These agents should consume data, decide where to measure next,
    and execute a single type of plan (maybe, move and count).
    """

    @property
    @abstractmethod
    def server_host(self):
        """
        Host to POST requests to. Declare as property or as class level attribute.
        Something akin to 'http://localhost:60610'
        """
        ...

    @property
    @abstractmethod
    def measurement_plan_name(self) -> str:
        """String name of registered plan"""
        ...

    @abstractmethod
    def measurement_plan_args(self, *args) -> list:
        """List of arguments to pass to plan"""
        ...

    @abstractmethod
    def tell(self, x, y):
        """
        Tell the agent about some new data
        Parameters
        ----------
        x :
            Independent variable for data observed
        y :
            Dependent variable for data observed

        Returns
        -------

        """
        ...

    @abstractmethod
    def ask(self, batch_size: int):
        """
        Ask the agent for a new batch of points to measure.

        Parameters
        ----------
        batch_size : int
            Number of new points to measure

        Returns
        -------
        Set of independent variables of length batch size

        """
        ...

    def report(self):
        """
        Create a report given the data observed by the agent.
        This could be potentially implemented in the base class to write document stream.
        """
        raise NotImplementedError

    def tell_many(self, xs, ys):
        """
        Tell the agent about some new data. It is likely that there is a more efficient approach to
        handling multiple observations for an agent. The default behavior is to iterate over all
        observations and call the ``tell`` method.

        Parameters
        ----------
        xs : list, array
            Array of independent variables for observations
        ys : list, array
            Array of dependent variables for observations

        Returns
        -------

        """
        for x, y in zip(xs, ys):
            self.tell(x, y)

    @property
    def _add_position(self) -> Union[int, Literal["front", "back"]]:
        return "back"

    def _add_to_queue(self, batch_size: int = 1):
        """
        Soft wrapper for the `ask` method that puts the agent's
        proposed next points on the queue.

        Parameters
        ----------
        batch_size : int
            Number of new points to measure

        Returns
        -------

        """
        next_points = self.ask(batch_size)
        url = Path(self.server_host) / "api" / "queue" / "item" / "add"
        responses = {}
        for point in next_points:
            data = dict(
                pos=self._add_position,
                item=dict(
                    name=self.measurement_plan_name, args=self.measurement_plan_args(point), item_type="plan"
                ),
            )
            r = requests.post(str(url), data=data)
            responses[point] = r
        return responses
