import sys
import os
import yaml
from dataclasses import dataclass, fields, asdict
from typing import Optional
from types import SimpleNamespace
import torch

# Add the 'hjepa' directory to the Python path. Change it to your path
sys.path.append("/scratch/wz1232/HIQL/hjepa")

from data.wall.wall import WallDataset, WallDatasetConfig


def dict_to_namespace(d):
    """
    # Function to convert dictionary to SimpleNamespace
    """
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_namespace(value)  # Recursively handle nested dictionaries
    return SimpleNamespace(**d)


def update_config_from_yaml(config_class, yaml_data):
    """
    Create an instance of `config_class` using default values, but override
    fields with those provided in `yaml_data`.
    """
    config_field_names = {f.name for f in fields(config_class)}
    relevant_yaml_data = {
        key: value for key, value in yaml_data.items() if key in config_field_names
    }
    return config_class(**relevant_yaml_data)


# yaml file for HJEPA experiment using random exploratory dataset
data_yaml_path = "/scratch/wz1232/HIQL/src/envs/wall/configs/no_fixed_wall.yaml"

# Load the YAML configuration from a file (or string)
with open(data_yaml_path, "r") as file:
    yaml_data = yaml.safe_load(file)

# Create offline dataset
data_yaml_data = yaml_data["wall_config"]
data_config = update_config_from_yaml(WallDatasetConfig, data_yaml_data)
ds = WallDataset(data_config)
states, locations, actions, bias_angle, wall_x, door_y = ds[
    0
]  # All the samples are normalized

# Create evaluation envs
eval_yaml_data = yaml_data["eval_cfg"]
eval_config = dict_to_namespace(eval_yaml_data)

from planning.wall.utils import construct_planning_envs

# NORMALIZED obs and target_obs by default (ds.normalizer.normalize_state(obs))
envs, obs, targets, target_obs, wall_locs, door_locs = construct_planning_envs(
    config=eval_config,
    wall_config=data_config,
    cross_wall=True,
    n_envs=eval_config.n_envs,
    normalizer=ds.normalizer,
)

# Then you can execute your policy / planner to get action and perform on the envs...
# NOTE: We assume that the policy outputs NORMALIZED actions. you want to UNNORMALIZE it before doing env.step(a)

# finally. figure out which trials are successful by looking at whether dot_position and target l2 distance < threshold
locations = torch.stack([e.dot_position for e in envs])
mse = (locations - targets).pow(2).mean(dim=1)
successes = (mse < eval_config.error_threshold).float()
