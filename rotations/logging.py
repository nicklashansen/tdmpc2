from typing import Any

from array_api_compat import array_namespace
from lsy_rl.core.logger import Collector


class AngleSuccessCollector(Collector):
    def __init__(self, angle_key: str, success_key: str, tol: float):
        self._angle = None
        self._xp = None
        self._tol = tol
        self._angle_key = angle_key
        self._success_key = success_key

    def collect(self, **kwargs: Any):
        if "info" not in kwargs:
            return
        if self._xp is None:
            self._xp = array_namespace(kwargs["info"]["angle"])
        self._angle = kwargs["info"]["angle"]

    def log(self, mask):
        if self._angle is None:
            return {}
        return {
            self._angle_key: float(self._xp.mean(self._angle[mask])),
            self._success_key: float(self._xp.mean(1.0 * (self._angle[mask] < self._tol))),
        }

    def clear(self, mask):
        if self._angle is not None:
            self._angle[mask] = 0


class SuccessCollector(Collector):
    def __init__(self, success_key: str):
        self._xp = None
        self._success_key = success_key
        self._success = None

    def collect(self, **kwargs: Any):
        if "reward" not in kwargs:
            return
        if self._xp is None:
            self._xp = array_namespace(kwargs["reward"])
        self._success = kwargs["reward"] == 0.0

    def log(self, mask):
        if self._success is None:
            return {}
        return {self._success_key: float(self._xp.mean(1.0 * (self._success[mask])))}

    def clear(self, mask):
        if self._success is not None:
            self._success[mask] = 0.0
