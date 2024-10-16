import numpy as np
import zarr

from jaxrl_m.dataset import Dataset


def get_mw_dataset(path: str, visual: bool = False):
    store = zarr.DirectoryStore(path)
    root = zarr.open(store, mode="r")  # Open in read-only mode

    obs = root["obs"][:] if visual else root["proprio"][:]
    action = root["action"][:]
    reward = root["reward"][:]
    episode = root["episode"][:]

    done = episode[1:] != episode[:-1]
    done = np.concatenate((done, [True]))

    nan_mask_reward = np.isnan(reward)
    nan_mask_action = np.isnan(action).any(axis=1)
    nan_mask = nan_mask_reward | nan_mask_action

    # drop values with nan
    obs = obs[~nan_mask]
    action = action[~nan_mask]
    reward = reward[~nan_mask]
    done = done[~nan_mask]
    episode = episode[~nan_mask]

    # get the episodes idxs
    episode_idxs = episode[done]

    ends = np.where(done)[0]
    lengths = np.concatenate(([ends[0] + 1], ends[1:] - ends[:-1]))

    # only keep the ones that are 103 long

    episode_idxs_mask = lengths != 100
    filtered_episode_idxs = episode_idxs[episode_idxs_mask]
    # filter all fields by the episode idxs
    filtered_idxs = (episode[:, None] != filtered_episode_idxs).all(axis=1)

    obs = obs[filtered_idxs]
    action = action[filtered_idxs]
    reward = reward[filtered_idxs]
    done = done[filtered_idxs]
    episode = episode[filtered_idxs]

    next_obs = np.concatenate((obs[1:], [obs[-1]]), axis=0)
    next_obs[done] = obs[done]

    dones_float = done.astype(np.float32)

    return Dataset.create(
        observations=obs,
        actions=action,
        rewards=reward,
        masks=np.logical_not(done),
        next_observations=next_obs,
        dones_float=dones_float,
        # env_infos=env_infos
    )
