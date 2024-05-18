from __future__ import annotations

from copy import copy
from typing import Tuple

import torch

from tensordict.utils import NestedKey

from torchrl.data.replay_buffers.storages import Storage, TensorStorage
from torchrl.data.replay_buffers.samplers import Sampler


# Source: https://pytorch.org/rl/reference/generated/torchrl.data.replay_buffers.SliceSampler.html
# This copy will live here until it has been included in a few torchrl stable releases


class SliceSampler(Sampler):
    """Samples slices of data along the first dimension, given start and stop signals.

    This class samples sub-trajectories with replacement. For a version without
    replacement, see :class:`~torchrl.data.replay_buffers.samplers.SliceSamplerWithoutReplacement`.

    Keyword Args:
        num_slices (int): the number of slices to be sampled. The batch-size
            must be greater or equal to the ``num_slices`` argument. Exclusive
            with ``slice_len``.
        slice_len (int): the length of the slices to be sampled. The batch-size
            must be greater or equal to the ``slice_len`` argument and divisible
            by it. Exclusive with ``num_slices``.
        end_key (NestedKey, optional): the key indicating the end of a
            trajectory (or episode). Defaults to ``("next", "done")``.
        traj_key (NestedKey, optional): the key indicating the trajectories.
            Defaults to ``"episode"`` (commonly used across datasets in TorchRL).
        cache_values (bool, optional): to be used with static datasets.
            Will cache the start and end signal of the trajectory.
        truncated_key (NestedKey, optional): If not ``None``, this argument
            indicates where a truncated signal should be written in the output
            data. This is used to indicate to value estimators where the provided
            trajectory breaks. Defaults to ``("next", "truncated")``.
            This feature only works with :class:`~torchrl.data.replay_buffers.TensorDictReplayBuffer`
            instances (otherwise the truncated key is returned in the info dictionary
            returned by the :meth:`~torchrl.data.replay_buffers.ReplayBuffer.sample` method).
        strict_length (bool, optional): if ``False``, trajectories of length
            shorter than `slice_len` (or `batch_size // num_slices`) will be
            allowed to appear in the batch.
            Be mindful that this can result in effective `batch_size`  shorter
            than the one asked for! Trajectories can be split using
            :func:`torchrl.collectors.split_trajectories`. Defaults to ``True``.

    .. note:: To recover the trajectory splits in the storage,
        :class:`~torchrl.data.replay_buffers.samplers.SliceSampler` will first
        attempt to find the ``traj_key`` entry in the storage. If it cannot be
        found, the ``end_key`` will be used to reconstruct the episodes.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data.replay_buffers import LazyMemmapStorage, TensorDictReplayBuffer
        >>> from torchrl.data.replay_buffers.samplers import SliceSampler
        >>> torch.manual_seed(0)
        >>> rb = TensorDictReplayBuffer(
        ...     storage=LazyMemmapStorage(1_000_000),
        ...     sampler=SliceSampler(cache_values=True, num_slices=10),
        ...     batch_size=320,
        ... )
        >>> episode = torch.zeros(1000, dtype=torch.int)
        >>> episode[:300] = 1
        >>> episode[300:550] = 2
        >>> episode[550:700] = 3
        >>> episode[700:] = 4
        >>> data = TensorDict(
        ...     {
        ...         "episode": episode,
        ...         "obs": torch.randn((3, 4, 5)).expand(1000, 3, 4, 5),
        ...         "act": torch.randn((20,)).expand(1000, 20),
        ...         "other": torch.randn((20, 50)).expand(1000, 20, 50),
        ...     }, [1000]
        ... )
        >>> rb.extend(data)
        >>> sample = rb.sample()
        >>> print("sample:", sample)
        >>> print("episodes", sample.get("episode").unique())
        episodes tensor([1, 2, 3, 4], dtype=torch.int32)

    :class:`torchrl.data.replay_buffers.SliceSampler` is default-compatible with
    most of TorchRL's datasets:

    Examples:
        >>> import torch
        >>>
        >>> from torchrl.data.datasets import RobosetExperienceReplay
        >>> from torchrl.data import SliceSampler
        >>>
        >>> torch.manual_seed(0)
        >>> num_slices = 10
        >>> dataid = list(RobosetExperienceReplay.available_datasets)[0]
        >>> data = RobosetExperienceReplay(dataid, batch_size=320, sampler=SliceSampler(num_slices=num_slices))
        >>> for batch in data:
        ...     batch = batch.reshape(num_slices, -1)
        ...     break
        >>> print("check that each batch only has one episode:", batch["episode"].unique(dim=1))
        check that each batch only has one episode: tensor([[19],
                [14],
                [ 8],
                [10],
                [13],
                [ 4],
                [ 2],
                [ 3],
                [22],
                [ 8]])

    """

    def __init__(
        self,
        *,
        num_slices: int = None,
        slice_len: int = None,
        end_key: NestedKey | None = None,
        traj_key: NestedKey | None = None,
        cache_values: bool = False,
        truncated_key: NestedKey | None = ("next", "truncated"),
        strict_length: bool = True,
    ) -> object:
        if end_key is None:
            end_key = ("next", "done")
        if traj_key is None:
            traj_key = "episode"
        if not ((num_slices is None) ^ (slice_len is None)):
            raise TypeError(
                "Either num_slices or slice_len must be not None, and not both. "
                f"Got num_slices={num_slices} and slice_len={slice_len}."
            )
        self.num_slices = num_slices
        self.slice_len = slice_len
        self.end_key = end_key
        self.traj_key = traj_key
        self.truncated_key = truncated_key
        self.cache_values = cache_values
        self._fetch_traj = True
        self._uses_data_prefix = False
        self.strict_length = strict_length
        self._cache = {}

    @staticmethod
    def _find_start_stop_traj(*, trajectory=None, end=None):
        if trajectory is not None:
            # slower
            # _, stop_idx = torch.unique_consecutive(trajectory, return_counts=True)
            # stop_idx = stop_idx.cumsum(0) - 1

            # even slower
            # t = trajectory.unsqueeze(0)
            # w = torch.tensor([1, -1], dtype=torch.int).view(1, 1, 2)
            # stop_idx = torch.conv1d(t, w).nonzero()

            # faster
            end = trajectory[:-1] != trajectory[1:]
            end = torch.cat([end, torch.ones_like(end[:1])], 0)
        else:
            end = torch.index_fill(
                end,
                index=torch.tensor(-1, device=end.device, dtype=torch.long),
                dim=0,
                value=1,
            )
        if end.ndim != 1:
            raise RuntimeError(
                f"Expected the end-of-trajectory signal to be 1-dimensional. Got a {end.ndim} tensor instead."
            )
        stop_idx = end.view(-1).nonzero().view(-1)
        start_idx = torch.cat([torch.zeros_like(stop_idx[:1]), stop_idx[:-1] + 1])
        lengths = stop_idx - start_idx + 1
        return start_idx, stop_idx, lengths

    def _tensor_slices_from_startend(self, seq_length, start):
        if isinstance(seq_length, int):
            return (
                torch.arange(
                    seq_length, device=start.device, dtype=start.dtype
                ).unsqueeze(0)
                + start.unsqueeze(1)
            ).view(-1)
        else:
            # when padding is needed
            return torch.cat(
                [
                    _start
                    + torch.arange(_seq_len, device=start.device, dtype=start.dtype)
                    for _start, _seq_len in zip(start, seq_length)
                ]
            )

    def _get_stop_and_length(self, storage, fallback=True):
        if self.cache_values and "stop-and-length" in self._cache:
            return self._cache.get("stop-and-length")

        if self._fetch_traj:
            # We first try with the traj_key
            try:
                # In some cases, the storage hides the data behind "_data".
                # In the future, this may be deprecated, and we don't want to mess
                # with the keys provided by the user so we fall back on a proxy to
                # the traj key.
                try:
                    trajectory = storage._storage.get(self._used_traj_key)
                except KeyError:
                    trajectory = storage._storage.get(("_data", self.traj_key))
                    # cache that value for future use
                    self._used_traj_key = ("_data", self.traj_key)
                self._uses_data_prefix = (
                    isinstance(self._used_traj_key, tuple)
                    and self._used_traj_key[0] == "_data"
                )
                vals = self._find_start_stop_traj(trajectory=trajectory[: len(storage)])
                if self.cache_values:
                    self._cache["stop-and-length"] = vals
                return vals
            except KeyError:
                if fallback:
                    self._fetch_traj = False
                    return self._get_stop_and_length(storage, fallback=False)
                raise

        else:
            try:
                # In some cases, the storage hides the data behind "_data".
                # In the future, this may be deprecated, and we don't want to mess
                # with the keys provided by the user so we fall back on a proxy to
                # the traj key.
                try:
                    done = storage._storage.get(self._used_end_key)
                except KeyError:
                    done = storage._storage.get(("_data", self.end_key))
                    # cache that value for future use
                    self._used_end_key = ("_data", self.end_key)
                self._uses_data_prefix = (
                    isinstance(self._used_end_key, tuple)
                    and self._used_end_key[0] == "_data"
                )
                vals = self._find_start_stop_traj(end=done.squeeze())[: len(storage)]
                if self.cache_values:
                    self._cache["stop-and-length"] = vals
                return vals
            except KeyError:
                if fallback:
                    self._fetch_traj = True
                    return self._get_stop_and_length(storage, fallback=False)
                raise

    def _adjusted_batch_size(self, batch_size):
        if self.num_slices is not None:
            if batch_size % self.num_slices != 0:
                raise RuntimeError(
                    f"The batch-size must be divisible by the number of slices, got batch_size={batch_size} and num_slices={self.num_slices}."
                )
            seq_length = batch_size // self.num_slices
            num_slices = self.num_slices
        else:
            if batch_size % self.slice_len != 0:
                raise RuntimeError(
                    f"The batch-size must be divisible by the slice length, got batch_size={batch_size} and slice_len={self.slice_len}."
                )
            seq_length = self.slice_len
            num_slices = batch_size // self.slice_len
        return seq_length, num_slices

    def sample(self, storage: Storage, batch_size: int) -> Tuple[torch.Tensor, dict]:
        if not isinstance(storage, TensorStorage):
            raise RuntimeError(
                f"{type(self)} can only sample from TensorStorage subclasses, got {type(storage)} instead."
            )

        # pick up as many trajs as we need
        start_idx, stop_idx, lengths = self._get_stop_and_length(storage)
        seq_length, num_slices = self._adjusted_batch_size(batch_size)
        return self._sample_slices(lengths, start_idx, seq_length, num_slices)

    def _sample_slices(
        self, lengths, start_idx, seq_length, num_slices, traj_idx=None
    ) -> Tuple[torch.Tensor, dict]:
        if (lengths < seq_length).any():
            if self.strict_length:
                raise RuntimeError(
                    "Some stored trajectories have a length shorter than the slice that was asked for. "
                    "Create the sampler with `strict_length=False` to allow shorter trajectories to appear "
                    "in you batch."
                )
            # make seq_length a tensor with values clamped by lengths
            seq_length = lengths.clamp_max(seq_length)

        if traj_idx is None:
            traj_idx = torch.randint(
                lengths.shape[0], (num_slices,), device=lengths.device
            )
        else:
            num_slices = traj_idx.shape[0]
        relative_starts = (
            (
                torch.rand(num_slices, device=lengths.device)
                * (lengths[traj_idx] - seq_length + 1)
            )
            .floor()
            .to(start_idx.dtype)
        )
        starts = start_idx[traj_idx] + relative_starts
        index = self._tensor_slices_from_startend(seq_length, starts)
        if self.truncated_key is not None:
            truncated_key = self.truncated_key

            truncated = torch.zeros(index.shape, dtype=torch.bool, device=index.device)
            if isinstance(seq_length, int):
                truncated.view(num_slices, -1)[:, -1] = 1
            else:
                truncated[seq_length.cumsum(0) - 1] = 1
            return index.to(torch.long), {truncated_key: truncated}
        return index.to(torch.long), {}

    @property
    def _used_traj_key(self):
        return self.__dict__.get("__used_traj_key", self.traj_key)

    @_used_traj_key.setter
    def _used_traj_key(self, value):
        self.__dict__["__used_traj_key"] = value

    @property
    def _used_end_key(self):
        return self.__dict__.get("__used_end_key", self.end_key)

    @_used_end_key.setter
    def _used_end_key(self, value):
        self.__dict__["__used_end_key"] = value

    def _empty(self):
        pass

    def dumps(self, path):
        # no op - cache does not need to be saved
        ...

    def loads(self, path):
        # no op
        ...

    def state_dict(self):
        ...

    def load_state_dict(self, state_dict):
        ...

    def __getstate__(self):
        state = copy(self.__dict__)
        state["_cache"] = {}
        return state
