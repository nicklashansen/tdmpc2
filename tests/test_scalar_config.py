"""The real Hydra configuration entry point must accept scalar modes."""

from pathlib import Path
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from common.parser import parse_cfg


@pytest.mark.parametrize("num_bins", [0, 1, 101])
def test_hydra_configuration_initializes_all_regression_modes(num_bins):
    # Resolve the config alongside the unchanged common package under test.
    import common

    config_dir = Path(common.__file__).resolve().parent.parent
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=[
                "hydra/launcher=basic",
                "task=walker-walk",
                "checkpoint=null",
                "data_dir=null",
                "model_size=null",
                "wandb_project=null",
                "wandb_entity=null",
                f"num_bins={num_bins}",
            ],
        )
        HydraConfig.instance().set_config(cfg)
        task_cfg = OmegaConf.masked_copy(cfg, [k for k in cfg if k != "hydra"])
        parsed = parse_cfg(task_cfg)
    assert parsed.num_bins == num_bins
    if num_bins > 1:
        assert parsed.bin_size == 0.2
