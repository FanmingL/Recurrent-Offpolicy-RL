import gym
from gym import Wrapper
import numpy as np
import copy

# this file is adapted from https://github.com/dennisl88/rand_param_envs

class DynamicsMetaEnvWrapper(Wrapper):
    RAND_PARAMS = ['body_mass', 'dof_damping', 'body_inertia', 'geom_friction', 'gravity', 'density',
                   'wind', 'geom_friction_1_dim', 'dof_damping_1_dim']
    RAND_PARAMS_EXTENDED = RAND_PARAMS + ['geom_size']

    def __init__(self, env, n_tasks=60, rand_params=['gravity'], log_scale_limit=3.0):
        super().__init__(env)
        if len(rand_params) == 1 and rand_params[0] == 'None':
            rand_params = []
        self.n_tasks = n_tasks
        self.normalize_context = True
        self.log_scale_limit = log_scale_limit
        self.rand_params = rand_params
        self.save_parameters()
        self.min_param, self.max_param = self.get_minmax_parameter(log_scale_limit)
        self.cur_parameter_vector = self.env_parameter_vector_
        self.cur_step_ind = 0
        # for non-stationary changing
        self.setted_env_params = None
        self.setted_env_changing_period = None
        self.setted_env_changing_interval = None
        self.min_action = env.action_space.low
        self.max_action = env.action_space.high
        self.range_action = self.max_action - self.min_action
        self._debug_state = None
        self.tasks = self.sample_tasks(n_tasks)

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def get_minmax_parameter(self, log_scale_limit):
        min_param = {}
        max_param = {}
        bound = lambda x, y: np.array(1.5) ** (np.ones(shape=x) * ((-1 if y == 'low' else 1) * log_scale_limit))
        if 'body_mass' in self.rand_params:
            min_multiplyers = bound(self.model.body_mass.shape, 'low')
            max_multiplyers = bound(self.model.body_mass.shape, 'high')
            min_param['body_mass'] = self.init_params['body_mass'] * min_multiplyers
            max_param['body_mass'] = self.init_params['body_mass'] * max_multiplyers

        # body_inertia
        if 'body_inertia' in self.rand_params:
            min_multiplyers = bound(self.model.body_inertia.shape, 'low')
            max_multiplyers = bound(self.model.body_inertia.shape, 'high')
            min_param['body_inertia'] = self.init_params['body_inertia'] * min_multiplyers
            max_param['body_inertia'] = self.init_params['body_inertia'] * max_multiplyers

        # damping -> different multiplier for different dofs/joints
        if 'dof_damping' in self.rand_params:
            min_multiplyers = bound(self.model.dof_damping.shape, 'low')
            max_multiplyers = bound(self.model.dof_damping.shape, 'high')
            min_param['dof_damping'] = self.init_params['dof_damping'] * min_multiplyers
            max_param['dof_damping'] = self.init_params['dof_damping'] * max_multiplyers

        # friction at the body components
        if 'geom_friction' in self.rand_params:
            min_multiplyers = bound(self.model.geom_friction.shape, 'low')
            max_multiplyers = bound(self.model.geom_friction.shape, 'high')
            min_param['geom_friction'] = self.init_params['geom_friction'] * min_multiplyers
            max_param['geom_friction'] = self.init_params['geom_friction'] * max_multiplyers

        if 'geom_friction_1_dim' in self.rand_params:
            min_multiplyers = bound((1,), 'low')
            max_multiplyers = bound((1,), 'high')
            min_param['geom_friction_1_dim'] = np.array([min_multiplyers])
            max_param['geom_friction_1_dim'] = np.array([max_multiplyers])

        if 'dof_damping_1_dim' in self.rand_params:
            min_multiplyers = bound((1,), 'low')
            max_multiplyers = bound((1,), 'high')
            min_param['dof_damping_1_dim'] = np.array([min_multiplyers])
            max_param['dof_damping_1_dim'] = np.array([max_multiplyers])

        if 'gravity' in self.rand_params:
            min_multiplyers = bound(self.model.opt.gravity.shape, 'low')
            max_multiplyers = bound(self.model.opt.gravity.shape, 'high')
            min_param['gravity'] = self.init_params['gravity'] * min_multiplyers
            max_param['gravity'] = self.init_params['gravity'] * max_multiplyers

            if 'gravity_angle' in self.rand_params:
                min_param['gravity'][:2] = min_param['gravity'][2]
                max_param['gravity'][:2] = max_param['gravity'][2]

        if 'wind' in self.rand_params:
            min_param['wind'] = np.array([-log_scale_limit, -log_scale_limit])
            max_param['wind'] = np.array([log_scale_limit, log_scale_limit])

        if 'density' in self.rand_params:
            min_multiplyers = bound((1,), 'low')
            max_multiplyers = bound((1,), 'high')
            min_param['density'] = self.init_params['density'] * min_multiplyers
            max_param['density'] = self.init_params['density'] * max_multiplyers

        for key in min_param:
            min_it = min_param[key]
            max_it = max_param[key]
            min_real = np.min([min_it, max_it], 0)
            max_real = np.max([max_it, min_it], 0)
            min_param[key] = min_real
            max_param[key] = max_real
        return min_param, max_param

    def denormalization(self, action):
        return (action + 1) / 2 * self.range_action + self.min_action

    def normalization(self, action):
        return (action - self.min_action) / self.range_action * 2 - 1

    def step(self, action):
        self.cur_step_ind += 1
        if self.setted_env_params is not None and self.cur_step_ind % self.setted_env_changing_interval == 0:
            assert isinstance(self.setted_env_params, list)
            env_to_be = {}
            weight_origin = self.cur_step_ind / self.setted_env_changing_period
            # weight = min(weight_origin, 1)
            weight_in_duration = weight_origin - (weight_origin // 2 * 2)
            if weight_in_duration <= 1:
                weight = weight_in_duration
            else:
                weight = 2 - weight_in_duration
            ind = int(weight_origin)
            if isinstance(self.setted_env_params[0], dict):
                for key in self.setted_env_params[0]:
                    # env_to_be[key] = (1 - weight) * self.setted_env_params[0][key] + weight * self.setted_env_params[1][key]
                    # env_to_be[key] = self.setted_env_params[ind][key] if weight_in_duration <= 1 else self.setted_env_params[ind][key]
                    env_to_be[key] = self.setted_env_params[ind][key]
            elif isinstance(self.setted_env_params[0], int):
                env_to_be = self.setted_env_params[ind]
            elif isinstance(self.setted_env_params[0], list):
                env_to_be = copy.deepcopy(self.setted_env_params[ind])
            else:
                raise NotImplementedError(f'type of {type(self.setted_env_params[ind])} is not implemented.')
            self.set_task(env_to_be)
        try:
            res = super(DynamicsMetaEnvWrapper, self).step(action)
            self._debug_state = res[0]
            return res
        except Exception as e:
            print(e)
            print('Inf or NaN found in state or action!!')
            print('Action: ', action)
            print('Param: ', self.env_parameter_vector_)
            print('Current state: ', self._debug_state)
            param_variable = getattr(self.unwrapped.model, 'dof_damping')
            print('Current parameter: ', self.cur_params)
            print('Dof_damping: ', param_variable)
            return self._debug_state, 0, True, {}

            # raise e
        # return super(DynamicsMetaEnvWrapper, self).step(action)

    def set_nonstationary_para(self, setting_env_params, changine_period, changing_interval):
        self.setted_env_changing_period = changine_period
        self.setted_env_params = setting_env_params
        self.setted_env_changing_interval = changing_interval

    def reset_nonstationary(self):
        self.set_nonstationary_para(None, None, None)

    def reset(self, **kwargs):
        self.cur_step_ind = 0
        return super(DynamicsMetaEnvWrapper, self).reset(**kwargs)

    def sample_tasks(self, n_tasks, dig_range=None, linspace=False):
        """
        Generates randomized parameter sets for the mujoco env
        Args:
            n_tasks (int) : number of different meta-tasks needed
        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """

        current_task_count_ = 0
        param_sets = []
        if dig_range is None:
            if linspace:
                def uniform_function(low_, up_, size):
                    res = [0] * np.prod(size)
                    interval = (up_ - low_) / (n_tasks - 1)
                    for i in range(len(res)):
                        res[i] = interval * current_task_count_ + low_
                    res = np.array(res).reshape(size)
                    return res
                uniform = uniform_function
            else:
                uniform = lambda low_,up_,size: np.random.uniform(low_, up_, size=size)
        else:
            dig_range = np.abs(dig_range)
            def uniform_function(low_, up_, size):
                res = [0] * np.prod(size)
                for i in range(len(res)):
                    if linspace:
                        if current_task_count_ >= n_tasks // 2:
                            interval = (up_ - dig_range) / (n_tasks // 2)
                            # res[i] = interval * (current_task_count_ - n_tasks // 2 + 1) + dig_range
                            res[i] = interval * (current_task_count_ - n_tasks // 2 ) + dig_range
                        else:
                            interval = (-dig_range - low_) / (n_tasks // 2)
                            # res[i] = interval * (n_tasks // 2 - current_task_count_ - 1) + low_
                            res[i] = interval * (n_tasks // 2 - current_task_count_) + low_
                    else:
                        while True:
                            rand = np.random.uniform(low_, up_)
                            if rand > dig_range or rand < -dig_range:
                                res[i] = rand
                                break
                res = np.array(res).reshape(size)
                return res
            uniform = uniform_function
        bound = lambda x: np.array(1.5) ** uniform(-self.log_scale_limit, self.log_scale_limit,  x)
        bound_uniform = lambda x: uniform(-self.log_scale_limit, self.log_scale_limit,  x)
        for _ in range(n_tasks):
            # body mass -> one multiplier for all body parts
            new_params = {}

            if 'body_mass' in self.rand_params:
                body_mass_multiplyers = bound(self.model.body_mass.shape)
                new_params['body_mass'] = self.init_params['body_mass'] * body_mass_multiplyers

            # body_inertia
            if 'body_inertia' in self.rand_params:
                body_inertia_multiplyers = bound(self.model.body_inertia.shape)
                new_params['body_inertia'] = body_inertia_multiplyers * self.init_params['body_inertia']

            # damping -> different multiplier for different dofs/joints
            if 'dof_damping' in self.rand_params:
                dof_damping_multipliers = bound(self.model.dof_damping.shape)
                new_params['dof_damping'] = np.multiply(self.init_params['dof_damping'], dof_damping_multipliers)

            # friction at the body components
            if 'geom_friction' in self.rand_params:
                dof_damping_multipliers = bound(self.model.geom_friction.shape)
                new_params['geom_friction'] = np.multiply(self.init_params['geom_friction'], dof_damping_multipliers)

            if 'geom_friction_1_dim' in self.rand_params:
                geom_friction_1_dim_multipliers = bound((1,))
                new_params['geom_friction_1_dim'] = geom_friction_1_dim_multipliers

            if 'dof_damping_1_dim' in self.rand_params:
                dof_damping_1_dim_multipliers = bound((1,))
                new_params['dof_damping_1_dim'] = dof_damping_1_dim_multipliers

            if 'gravity' in self.rand_params:
                gravity_mutipliers = bound(self.model.opt.gravity.shape)
                new_params['gravity'] = np.multiply(self.init_params['gravity'], gravity_mutipliers)

                if 'gravity_angle' in self.rand_params:
                    min_angle = - self.log_scale_limit * np.array([1, 1]) / 8
                    max_angle = self.log_scale_limit * np.array([1, 1]) / 8
                    angle = np.random.uniform(min_angle, max_angle)
                    new_params['gravity'][0] = new_params['gravity'][2] * np.sin(angle[0]) * np.sin(angle[1])
                    new_params['gravity'][1] = new_params['gravity'][2] * np.sin(angle[0]) * np.cos(angle[1])
                    new_params['gravity'][2] *= np.cos(angle[0])

            if 'wind' in self.rand_params:
                new_params['wind'] = bound_uniform((2, ))

            if 'density' in self.rand_params:
                density_mutipliers = bound((1,))
                new_params['density'] = np.multiply(self.init_params['density'], density_mutipliers)
            param_sets.append(new_params)
            current_task_count_ += 1


        return param_sets

    def cross_params(self, param_a, param_b):
        param_res = []
        for item_a in param_a:
            for item_b in param_b:
                r = dict()
                for k, v in item_a.items():
                    r[k] = v
                for k, v in item_b.items():
                    r[k] = v
                param_res.append(r)
        return param_res

    def reset_task(self, task_idx):
        self.set_task(self.tasks[task_idx])
        self.reset()

    def set_task(self, task):
        for param, param_val in task.items():
            if param == 'gravity_angle':
                continue
            if param == 'gravity':
                param_variable = getattr(self.unwrapped.model.opt, param)
            elif param == 'density':
                self.unwrapped.model.opt.density = float(param_val[0])
                continue
            elif param == 'wind':
                param_variable = getattr(self.unwrapped.model.opt, param)
                param_variable[:2] = param_val
                continue
            elif param == 'geom_friction_1_dim':
                param_variable = getattr(self.unwrapped.model, 'geom_friction')
                param_variable[:] = self.init_params[param][:] * param_val
                continue
            elif param == 'dof_damping_1_dim':
                param_variable = getattr(self.unwrapped.model, 'dof_damping')
                param_variable[:] = self.init_params[param][:] * param_val
                continue
            else:
                param_variable = getattr(self.unwrapped.model, param)
            assert param_variable.shape == param_val.shape, 'shapes of new parameter value and old one must match'
            param_variable[:] = param_val
        self.cur_params = task
        self.cur_parameter_vector = self.env_parameter_vector_

    def get_task(self):
        return self.cur_params

    def save_parameters(self):
        self.init_params = {}
        if 'body_mass' in self.rand_params:
            self.init_params['body_mass'] = self.unwrapped.model.body_mass

        # body_inertia
        if 'body_inertia' in self.rand_params:
            self.init_params['body_inertia'] = self.unwrapped.model.body_inertia

        # damping -> different multiplier for different dofs/joints
        if 'dof_damping' in self.rand_params:
            self.init_params['dof_damping'] = np.array(self.unwrapped.model.dof_damping).copy()

        # friction at the body components
        if 'geom_friction' in self.rand_params:
            self.init_params['geom_friction'] = np.array(self.unwrapped.model.geom_friction).copy()

        if 'geom_friction_1_dim' in self.rand_params:
            self.init_params['geom_friction_1_dim'] = np.array(self.unwrapped.model.geom_friction).copy()

        if 'dof_damping_1_dim' in self.rand_params:
            self.init_params['dof_damping_1_dim'] = np.array(self.unwrapped.model.dof_damping).copy()

        if 'gravity' in self.rand_params:
            self.init_params['gravity'] = self.unwrapped.model.opt.gravity

        if 'wind' in self.rand_params:
            self.init_params['wind'] = self.unwrapped.model.opt.wind[:2]

        if 'density' in self.rand_params:
            self.init_params['density'] = np.array([self.unwrapped.model.opt.density])

        self.cur_params = copy.deepcopy(self.init_params)
        if 'dof_damping_1_dim' in self.cur_params:
            self.cur_params['dof_damping_1_dim'] = np.array([1.0])
        if 'geom_friction_1_dim' in self.cur_params:
            self.cur_params['geom_friction_1_dim'] = np.array([1.0])


    @property
    def env_parameter_vector(self):
        return self.cur_parameter_vector

    @property
    def env_parameter_vector_(self):
        keys = [key for key in self.rand_params]
        if len(keys) == 0:
            return []
        vec_ = [self.cur_params[key].reshape(-1,) for key in keys]
        cur_vec = np.hstack(vec_)
        if not self.normalize_context:
            return cur_vec
        vec_range = self.param_max - self.param_min
        vec_range[vec_range == 0] = 1.0
        cur_vec = (cur_vec - self.param_min) / vec_range
        return cur_vec

    @property
    def env_parameter_length(self):
        length = np.sum([np.shape(self.cur_params[key].reshape(-1, ))[-1] for key in self.cur_params])
        return length

    @property
    def param_max(self):
        keys = [key for key in self.rand_params]
        vec_ = [self.max_param[key].reshape(-1,) for key in keys]
        if len(vec_) == 0:
            return []
        return np.hstack(vec_)

    @property
    def param_min(self):
        keys = [key for key in self.rand_params]
        vec_ = [self.min_param[key].reshape(-1, ) for key in keys]
        if len(vec_) == 0:
            return []
        return np.hstack(vec_)

    @property
    def _elapsed_steps(self):
        return self.cur_step_ind

    @property
    def _max_episode_steps(self):
        if hasattr(self.env, '_max_episode_steps'):
            return self.env._max_episode_steps
        return 1000

