import gym
from gym.wrappers import RescaleAction


def make_env(
    env_name: str,
    seed: int,
) -> gym.Env:
    # Check if the env is in gym.
    env = gym.make(env_name)

    env.max_episode_steps = getattr(
        env, "max_episode_steps", env.spec.max_episode_steps
    )

    if isinstance(env.action_space, gym.spaces.Box):
        env = RescaleAction(env, -1.0, 1.0)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env
