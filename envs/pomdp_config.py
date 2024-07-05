

env_config = {
    # meta tasks
    'AntDir-v0': {'max_rollouts_per_task': 2, 'env_type': 'meta', 'num_eval_tasks': 20},
    'CheetahDir-v0': {'max_rollouts_per_task': 2, 'env_type': 'meta', 'num_eval_tasks': 20},
    'HalfCheetahVel-v0': {'max_rollouts_per_task': 2, 'env_type': 'meta', 'num_tasks': 120, 'num_train_tasks': 100, 'num_eval_tasks': 20},

    'HumanoidDir-v0': {'max_rollouts_per_task': 2, 'env_type': 'meta', 'num_eval_tasks': 20},
    'PointRobotSparse-v0': {'max_rollouts_per_task': 2, 'env_type': 'meta', 'num_tasks': 100, 'num_train_tasks': 80, 'num_eval_tasks': 20},
    'Wind-v0': {'max_rollouts_per_task': 1, 'env_type': 'meta', 'num_tasks': 60, 'num_train_tasks': 40, 'num_eval_tasks': 20},

    "DM_Hopper_gravity-v2": {'max_rollouts_per_task': 1, 'env_type': 'meta', 'num_tasks': 120, 'num_train_tasks': 100,
                             'num_eval_tasks': 20},
    "DM_Hopper_dof_damping-v2": {'max_rollouts_per_task': 1, 'env_type': 'meta', 'num_tasks': 120,
                                 'num_train_tasks': 100, 'num_eval_tasks': 20},
    "DM_Hopper_gravity_dof_damping-v2": {'max_rollouts_per_task': 1, 'env_type': 'meta', 'num_tasks': 120,
                                         'num_train_tasks': 100, 'num_eval_tasks': 20},
    "DM_Hopper_body_mass-v2": {'max_rollouts_per_task': 1, 'env_type': 'meta', 'num_tasks': 120, 'num_train_tasks': 100,
                               'num_eval_tasks': 20},
    "DM_HalfCheetah_gravity-v2": {'max_rollouts_per_task': 1, 'env_type': 'meta', 'num_tasks': 120,
                                  'num_train_tasks': 100, 'num_eval_tasks': 20},
    "DM_HalfCheetah_dof_damping-v2": {'max_rollouts_per_task': 1, 'env_type': 'meta', 'num_tasks': 120,
                                      'num_train_tasks': 100, 'num_eval_tasks': 20},
    "DM_HalfCheetah_gravity_dof_damping-v2": {'max_rollouts_per_task': 1, 'env_type': 'meta', 'num_tasks': 120,
                                              'num_train_tasks': 100, 'num_eval_tasks': 20},
    "DM_HalfCheetah_body_mass-v2": {'max_rollouts_per_task': 1, 'env_type': 'meta', 'num_tasks': 120,
                                    'num_train_tasks': 100, 'num_eval_tasks': 20},
    "DM_Walker2d_gravity-v2": {'max_rollouts_per_task': 1, 'env_type': 'meta', 'num_tasks': 120, 'num_train_tasks': 100,
                               'num_eval_tasks': 20},
    "DM_Walker2d_dof_damping-v2": {'max_rollouts_per_task': 1, 'env_type': 'meta', 'num_tasks': 120,
                                   'num_train_tasks': 100, 'num_eval_tasks': 20},
    "DM_Walker2d_gravity_dof_damping-v2": {'max_rollouts_per_task': 1, 'env_type': 'meta', 'num_tasks': 120,
                                           'num_train_tasks': 100, 'num_eval_tasks': 20},
    "DM_Walker2d_body_mass-v2": {'max_rollouts_per_task': 1, 'env_type': 'meta', 'num_tasks': 120,
                                 'num_train_tasks': 100, 'num_eval_tasks': 20},
    "DM_Ant_gravity-v2": {'max_rollouts_per_task': 1, 'env_type': 'meta', 'num_tasks': 120, 'num_train_tasks': 100,
                          'num_eval_tasks': 20},
    "DM_Ant_dof_damping-v2": {'max_rollouts_per_task': 1, 'env_type': 'meta', 'num_tasks': 120, 'num_train_tasks': 100,
                              'num_eval_tasks': 20},
    "DM_Ant_gravity_dof_damping-v2": {'max_rollouts_per_task': 1, 'env_type': 'meta', 'num_tasks': 120,
                                      'num_train_tasks': 100, 'num_eval_tasks': 20},
    "DM_Ant_body_mass-v2": {'max_rollouts_per_task': 1, 'env_type': 'meta', 'num_tasks': 120, 'num_train_tasks': 100,
                            'num_eval_tasks': 20},
    "DM_Humanoid_gravity-v2": {'max_rollouts_per_task': 1, 'env_type': 'meta', 'num_tasks': 120, 'num_train_tasks': 100,
                               'num_eval_tasks': 20},
    "DM_Humanoid_dof_damping-v2": {'max_rollouts_per_task': 1, 'env_type': 'meta', 'num_tasks': 120,
                                   'num_train_tasks': 100, 'num_eval_tasks': 20},
    "DM_Humanoid_gravity_dof_damping-v2": {'max_rollouts_per_task': 1, 'env_type': 'meta', 'num_tasks': 120,
                                           'num_train_tasks': 100, 'num_eval_tasks': 20},
    "DM_Humanoid_body_mass-v2": {'max_rollouts_per_task': 1, 'env_type': 'meta', 'num_tasks': 120,
                                 'num_train_tasks': 100, 'num_eval_tasks': 20},

    # pomdp tasks
    'AntBLT-P-v0': {'max_rollouts_per_task': 1, 'env_type': 'pomdp', 'num_eval_tasks': 10},
    'AntBLT-V-v0': {'max_rollouts_per_task': 1, 'env_type': 'pomdp', 'num_eval_tasks': 10},

    'CartPole-V-v0': {'max_rollouts_per_task': 1, 'env_type': 'pomdp', 'num_eval_tasks': 20},
    'CartPole-F-v0': {'max_rollouts_per_task': 1, 'env_type': 'pomdp', 'num_eval_tasks': 20},

    'HalfCheetahBLT-P-v0': {'max_rollouts_per_task': 1, 'env_type': 'pomdp', 'num_eval_tasks': 10},
    'HalfCheetahBLT-V-v0': {'max_rollouts_per_task': 1, 'env_type': 'pomdp', 'num_eval_tasks': 10},

    'HopperBLT-P-v0': {'max_rollouts_per_task': 1, 'env_type': 'pomdp', 'num_eval_tasks': 10},
    'HopperBLT-V-v0': {'max_rollouts_per_task': 1, 'env_type': 'pomdp', 'num_eval_tasks': 10},

    'LunarLander-F-v0': {'max_rollouts_per_task': 1, 'env_type': 'pomdp', 'num_eval_tasks': 20},
    'LunarLander-V-v0': {'max_rollouts_per_task': 1, 'env_type': 'pomdp', 'num_eval_tasks': 20},

    'Pendulum-V-v0': {'max_rollouts_per_task': 1, 'env_type': 'pomdp', 'num_eval_tasks': 20},

    'WalkerBLT-P-v0': {'max_rollouts_per_task': 1, 'env_type': 'pomdp', 'num_eval_tasks': 10},
    'WalkerBLT-V-v0': {'max_rollouts_per_task': 1, 'env_type': 'pomdp', 'num_eval_tasks': 10},

    # rmdp
    'MRPOHalfCheetahRandomNormal-v0': {'max_rollouts_per_task': 1, 'env_type': 'rmdp', 'num_eval_tasks': 100, 'worst_percentile': 0.10},
    'MRPOHopperRandomNormal-v0': {'max_rollouts_per_task': 1, 'env_type': 'rmdp', 'num_eval_tasks': 100, 'worst_percentile': 0.10},
    'MRPOWalker2dRandomNormal-v0': {'max_rollouts_per_task': 1, 'env_type': 'rmdp', 'num_eval_tasks': 100, 'worst_percentile': 0.10},

    # credit
    'Catch-40-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'KeytoDoor-SR-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},

    'Mem-T-passive-50-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'Mem-T-passive-100-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'Mem-T-passive-250-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'Mem-T-passive-500-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'Mem-T-passive-750-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'Mem-T-passive-1000-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'Mem-T-passive-1250-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'Mem-T-passive-1500-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'Mem-T-active-20-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'Mem-T-active-50-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'Mem-T-active-100-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'Mem-T-active-250-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'Mem-T-active-500-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'Mem-SR-60-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'Mem-SR-120-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'Mem-SR-250-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'Mem-SR-500-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},

    'Mem-T-passive-50-cont-act-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'Mem-T-passive-100-cont-act-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'Mem-T-passive-250-cont-act-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'Mem-T-passive-500-cont-act-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'Mem-T-passive-750-cont-act-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'Mem-T-passive-1000-cont-act-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'Mem-T-passive-1250-cont-act-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'Mem-T-passive-1500-cont-act-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'Mem-T-active-20-cont-act-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'Mem-T-active-50-cont-act-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'Mem-T-active-100-cont-act-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'Mem-T-active-250-cont-act-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'Mem-T-active-500-cont-act-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'Mem-SR-60-cont-act-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'Mem-SR-120-cont-act-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'Mem-SR-250-cont-act-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},
    'Mem-SR-500-cont-act-v0': {'max_rollouts_per_task': 1, 'env_type': 'credit', 'num_eval_tasks': 20},

    # generalize
    'SunblazeHalfCheetah-v0': {'max_rollouts_per_task': 1, 'env_type': 'generalize', 'eval_envs': {'SunblazeHalfCheetah-v0': 20,
                                                                                                 'SunblazeHalfCheetahRandomNormal-v0': 125,
                                                                                                 'SunblazeHalfCheetahRandomExtreme-v0': 125}},
    'SunblazeHalfCheetahRandomNormal-v0': {'max_rollouts_per_task': 1, 'env_type': 'generalize',
                               'eval_envs': {'SunblazeHalfCheetah-v0': 20,
                                             'SunblazeHalfCheetahRandomNormal-v0': 125,
                                             'SunblazeHalfCheetahRandomExtreme-v0': 125}},
    'SunblazeHopper-v0': {'max_rollouts_per_task': 1, 'env_type': 'generalize', 'eval_envs': {'SunblazeHopper-v0': 20,
                                                                                             'SunblazeHopperRandomNormal-v0': 125,
                                                                                             'SunblazeHopperRandomExtreme-v0': 125}},
    'SunblazeHopperRandomNormal-v0': {'max_rollouts_per_task': 1, 'env_type': 'generalize',
                               'eval_envs': {'SunblazeHopper-v0': 20,
                                             'SunblazeHopperRandomNormal-v0': 125,
                                             'SunblazeHopperRandomExtreme-v0': 125}},

    # yang_domains
    'ur5-top-v0': {'max_rollouts_per_task': 1, 'env_type': 'yang_domains', 'num_eval_tasks': 10},
    'ur5-mdp-top-v0': {'max_rollouts_per_task': 1, 'env_type': 'yang_domains', 'num_eval_tasks': 10},
    'water-maze-mdp-v0': {'max_rollouts_per_task': 1, 'env_type': 'yang_domains', 'num_eval_tasks': 10},
    'water-maze-pomdp-v0': {'max_rollouts_per_task': 1, 'env_type': 'yang_domains', 'num_eval_tasks': 10},
    'water-maze-dense-mdp-v0': {'max_rollouts_per_task': 1, 'env_type': 'yang_domains', 'num_eval_tasks': 10},
    'water-maze-dense-pomdp-v0': {'max_rollouts_per_task': 1, 'env_type': 'yang_domains', 'num_eval_tasks': 10},
    'water-maze-simple-mdp-v0': {'max_rollouts_per_task': 1, 'env_type': 'yang_domains', 'num_eval_tasks': 10},
    'water-maze-simple-pomdp-v0': {'max_rollouts_per_task': 1, 'env_type': 'yang_domains', 'num_eval_tasks': 10},
    'water-maze-simple-mdp-concat10-v0': {'max_rollouts_per_task': 1, 'env_type': 'yang_domains', 'num_eval_tasks': 10},
}

env_categories = {}
for _env_name, _value in env_config.items():
    if _value['env_type'] not in env_categories:
        env_categories[_value['env_type']] = []
    env_categories[_value['env_type']].append(_env_name)