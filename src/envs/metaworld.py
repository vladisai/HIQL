from dataclasses import dataclass
from collections import deque

import numpy as np
import gym
from tqdm import tqdm

from metaworld import policies
from metaworld import _make_tasks, _encode_task
from metaworld.envs import (
    ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
    ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
)


class PixelWrapper(gym.Wrapper):
    """
    Wrapper for pixel observations. Compatible with DMControl environments.
    """

    def __init__(self, env, cfg):
        super().__init__(env)
        self.cfg = cfg
        self.env = env
        self.proprio_env = env
        self.num_frames = cfg.num_frames
        render_size = cfg.img_size
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.num_frames * 3, render_size, render_size),
            dtype=np.uint8,
        )
        self._frames = deque([], maxlen=self.num_frames)
        self._render_size = render_size

    def _get_obs(self):
        # frame = self.env.render(
        #     mode="rgb_array", width=self._render_size, height=self._render_size
        # ).transpose(2, 0, 1)
        frame = self.env.render().transpose(2, 0, 1)
        self._frames.append(frame)
        return np.concatenate(self._frames)

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        for _ in range(self._frames.maxlen):
            obs = self._get_obs()
        return obs, info

    def step(self, action):
        _, reward, done, truncated, info = self.env.step(action)
        return self._get_obs(), reward, done, truncated, info


@dataclass
class MetaWorldConfig:
    task: str = "mw-pick-place-v2"
    visual: bool = True
    img_size: int = 128
    seed: int = 0
    freeze_rand_vec: bool = False
    num_frames: int = 3


class MetaWorldWrapper(gym.Wrapper):
    def __init__(self, env, cfg=None):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.camera_name = "corner2"
        self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        self.env.render_mode = "rgb_array"
        self.env.camera_name = self.camera_name
        self.env.width = cfg.img_size
        self.env.height = cfg.img_size
        self.env._freeze_rand_vec = self.cfg.freeze_rand_vec

        # import inspect

        # print(inspect.getsourcefile(MujocoRenderer))
        # print(inspect.getsource(MujocoRenderer))

        # fix for rendering
        # breakpoint()

        self.init_renderer()

    def init_renderer(self):
        pass
        # from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer  # noqa

        # # self.camera_name = "corner2"
        # # self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        # # self.env.render_mode = "rgb_array"
        # # self.env.camera_name = self.camera_name
        # # self.env.width = self.cfg.img_size
        # # self.env.height = self.cfg.img_size
        # # self.env._freeze_rand_vec = self.cfg.freeze_rand_vec

        # self.env.mujoco_renderer = MujocoRenderer(
        #     self.env.model,
        #     self.env.data,
        #     self.env.mujoco_renderer.default_cam_config,
        #     width=self.env.width,
        #     height=self.env.height,
        #     max_geom=self.env.mujoco_renderer.max_geom,
        #     camera_id=None,
        #     camera_name=self.env.camera_name,
        # )

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        for i in range(3):  # for some reason this is needed
            obs, _, trunc, termn, info = self.step(
                np.zeros(self.env.action_space.shape)
            )
        obs = obs.astype(np.float32)
        # self.env.step(np.zeros(self.env.action_space.shape))
        info["proprio"] = obs
        return obs, info

    def step(self, action):
        reward = 0
        obs, r, trunc, done, info = self.env.step(action.copy())
        reward += r
        obs = obs.astype(np.float32)
        info["proprio"] = obs
        return obs, reward, trunc, done, info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, *args, **kwargs):
        # return np.flip(
        #     self.env.sim.render(
        #         mode="offscreen",
        #         width=width,
        #         height=height,
        #         camera_name=self.camera_name,
        #     ),
        #     axis=0,
        # )
        # return self.env.render(
        #     offscreen=True, resolution=(384, 384), camera_name=self.camera_name
        # ).copy()
        result = self.env.render().copy()[::-1]
        if result.sum() == 0:
            self.init_renderer()
            result = self.env.render().copy()[::-1]
            if result.sum() == 0:
                raise ValueError("Rendering failed: 0 after reinit renderer.")
        return result  # flip vertically

    def _get_obs(self):
        return self.env._get_obs()


def build_metaworld_env(config: MetaWorldConfig):

    env_id = config.task.split("-", 1)[-1] + "-v2" + "-goal-observable"
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_id](seed=config.seed)
    env.seeded_rand_vec = False

    env = MetaWorldWrapper(env, config)
    if config.visual:
        env = PixelWrapper(env, config)

    return env


def task_name_to_policy_name(task_name: str):
    # task name is of the form mw-lever-pull
    # while the policy name is of the form SawyerLeverPullV2Policy
    # so we need to convert the task name to the policy name
    task_name = task_name.split("-")
    policy_name = "Sawyer"
    policy_name += "".join([word.capitalize() for word in task_name[1:]])
    policy_name += "V2Policy"
    return policy_name


def task_name_to_policy(task_name: str):
    policy_name = task_name_to_policy_name(task_name)
    return getattr(policies, policy_name)


def make_td(obs, info):
    return {
        "obs": obs,
        "proprio": info["proprio"] if "proprio" in info else obs,
    }


def unroll_agent(env, obs, info, actor):
    done = False
    pbar = tqdm(desc="Planning")
    success = False
    losses = []

    obs = make_td(obs, info)
    episode_imgs = [obs]

    pbar = tqdm(
        desc="executing agent",
        total=200,  # not sure if this is the right number
        initial=0,
        position=0,
        leave=True,
    )
    actions = []
    while not done:
        action = actor(obs)
        actions.append(action)
        # losses.append(agent._prev_losses)
        obs, reward, done, truncated, info = env.step(action)
        pbar.update(1)
        pbar.set_postfix(
            {
                "near_object": info["near_object"],
                "success": info["success"],
            }
        )
        obs = make_td(obs, info)
        episode_imgs.append(obs)

        if info["success"]:
            success = True
            done = True
        if truncated:
            success = False
            done = True

    pbar.close()

    return episode_imgs, actions, info["success"], losses


def get_goal_state(env, env_name):
    """a function to copy the environment and run the expert
    policy to get the goal state.
    """
    policy_cls = task_name_to_policy(env_name)

    # we need to unwrap the env and deepcopy it
    unwrapped = env.unwrapped
    env_cls = type(unwrapped)

    rand_vec = unwrapped._last_rand_vec

    task_data = _encode_task(
        "mw-reach",
        {
            "rand_vec": rand_vec,
            "env_cls": env_cls.__bases__[0],
            "partially_observable": False,
        },
    )

    env_expert = env_cls(seed=env.cfg.seed)
    # if env.cfg.visual:
    env_expert = MetaWorldWrapper(env_expert, env.cfg)
    env_expert = PixelWrapper(env_expert, env.cfg)
    env_expert.set_task(task_data)
    # env.set_task(task_data)
    obs, info = env_expert.reset()

    policy = policy_cls()

    def actor(obs):
        # need to convert to tensor because the env expects a tensor
        # return torch.tensor(env_expert.action_space.sample())
        return policy.get_action(obs["proprio"])

    episode_imgs, actions, info["success"], losses = unroll_agent(
        env_expert, obs, info, actor
    )
    last_state = episode_imgs[-1]

    goal_obs = last_state["obs"] if env.cfg.visual else last_state["proprio"]
    return goal_obs, last_state["obs"][:3]


def reset_warmup_env(env):
    obs, info = env.reset()
    for i in range(3):
        obs, _, trunc, termn, info = env.step(np.zeros(env.action_space.shape))
    return obs, info
