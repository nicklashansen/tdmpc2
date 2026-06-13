import gymnasium

from .pick_and_place import PickAndPlace, PickAndPlaceOrient
from .reach import Reach, ReachOrient
from .wrapper import RotationWrapper

gymnasium.register("Reach-v0", entry_point=Reach, max_episode_steps=50)
gymnasium.register("ReachOrient-v0", entry_point=ReachOrient, max_episode_steps=50)
gymnasium.register("PickAndPlace-v0", entry_point=PickAndPlace, max_episode_steps=50)
gymnasium.register("PickAndPlaceOrient-v0", entry_point=PickAndPlaceOrient, max_episode_steps=50)

__all__ = ["Reach", "ReachOrient", "PickAndPlace", "PickAndPlaceOrient", "RotationWrapper"]
