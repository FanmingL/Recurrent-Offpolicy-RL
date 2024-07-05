import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from .wrappers import ConcatObs, FilterObsByIndex


LOW, HIGH = 1/5, 1


class PendulumVarLenFullEnv(gym.Env):

    """
    Full-fledged pendulum swing-up task.

    By full-fledged, I mean it tracks past action and pole length and output them through
    observations.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, g=10.0):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = None  # will be set in reset()
        self.viewer = None

        high = np.array([1., 1., self.max_speed, HIGH, 1], dtype=np.float32)
        low = np.array([-1., -1., -self.max_speed, LOW, -1], dtype=np.float32)

        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32
        )

        self.seed()

        # added
        self.last_u = np.zeros(self.action_space.shape)
        self.should_update_viewer = False  # use this flag to update view upon reset, since self.l might have changed

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = np.zeros(self.action_space.shape)  # modified; originally None
        self.l = self.np_random.choice([LOW, HIGH])
        self.should_update_viewer = True
        # print(f'Called reset with l={self.l}')
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot, self.l, float(self.last_u)])

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(self.l, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)
            self.should_update_viewer = False

        if self.should_update_viewer:
            from gym.envs.classic_control import rendering
            rod = rendering.make_capsule(self.l, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.geoms[0] = rod

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)


# indices and interpretations
# 0: cos(theta)
# 1: sin(theta)
# 2: theta_dot
# 3: pole length
# 4: last action


# notation
# p: position
# v: velocity
# l: pole length
# a: action


def pvl():
    return FilterObsByIndex(PendulumVarLenFullEnv(), indices_to_keep=[0, 1, 2, 3])


def pv():
    return FilterObsByIndex(PendulumVarLenFullEnv(), indices_to_keep=[0, 1, 2])


def pl():
    return FilterObsByIndex(PendulumVarLenFullEnv(), indices_to_keep=[0, 1, 3])


def pa():
    return FilterObsByIndex(PendulumVarLenFullEnv(), indices_to_keep=[0, 1, 4])


def pa_concat5():
    return ConcatObs(pa(), 5)
