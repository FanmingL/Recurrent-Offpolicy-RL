from dm_control import suite
from gym.envs.registration import register
# DMC is not included in the POMDP tasks, just a simple task wrapper

for domain_name, task_name in suite.ALL_TASKS:
    register(
        f"dmc_{domain_name}_{task_name}-v0",
        entry_point="envs.dmc.dmc_env:DMCWrapper",
        kwargs={"domain_name": domain_name, "task_name": task_name},
    )

for domain_name, task_name in suite.ALL_TASKS:
    register(
        f"dmc_pixel_{domain_name}_{task_name}-v0",
        entry_point="envs.dmc.dmc_env:DMCWrapper",
        kwargs={"domain_name": domain_name, "task_name": task_name, 'from_pixels': True, 'visualize_reward': False},
    )

# dmc_acrobot_swingup-v0
# dmc_acrobot_swingup_sparse-v0
# dmc_ball_in_cup_catch-v0
# dmc_cartpole_balance-v0
# dmc_cartpole_balance_sparse-v0
# dmc_cartpole_swingup-v0
# dmc_cartpole_swingup_sparse-v0
# dmc_cheetah_run-v0
# dmc_finger_spin-v0
# dmc_finger_turn_easy-v0
# dmc_finger_turn_hard-v0
# dmc_fish_upright-v0
# dmc_fish_swim-v0
# dmc_hopper_stand-v0
# dmc_hopper_hop-v0
# dmc_humanoid_stand-v0
# dmc_humanoid_walk-v0
# dmc_humanoid_run-v0
# dmc_manipulator_bring_ball-v0
# dmc_pendulum_swingup-v0
# dmc_point_mass_easy-v0
# dmc_reacher_easy-v0
# dmc_reacher_hard-v0
# dmc_swimmer_swimmer6-v0
# dmc_swimmer_swimmer15-v0
# dmc_walker_stand-v0
# dmc_walker_walk-v0
# dmc_walker_run-v0

## FULL tasks
# dmc_acrobot_swingup-v0
# dmc_acrobot_swingup_sparse-v0
# dmc_ball_in_cup_catch-v0
# dmc_cartpole_balance-v0
# dmc_cartpole_balance_sparse-v0
# dmc_cartpole_swingup-v0
# dmc_cartpole_swingup_sparse-v0
# dmc_cartpole_two_poles-v0
# dmc_cartpole_three_poles-v0
# dmc_cheetah_run-v0
# dmc_dog_stand-v0
# dmc_dog_walk-v0
# dmc_dog_trot-v0
# dmc_dog_run-v0
# dmc_dog_fetch-v0
# dmc_finger_spin-v0
# dmc_finger_turn_easy-v0
# dmc_finger_turn_hard-v0
# dmc_fish_upright-v0
# dmc_fish_swim-v0
# dmc_hopper_stand-v0
# dmc_hopper_hop-v0
# dmc_humanoid_stand-v0
# dmc_humanoid_walk-v0
# dmc_humanoid_run-v0
# dmc_humanoid_run_pure_state-v0
# dmc_humanoid_CMU_stand-v0
# dmc_humanoid_CMU_walk-v0
# dmc_humanoid_CMU_run-v0
# dmc_lqr_lqr_2_1-v0
# dmc_lqr_lqr_6_2-v0
# dmc_manipulator_bring_ball-v0
# dmc_manipulator_bring_peg-v0
# dmc_manipulator_insert_ball-v0
# dmc_manipulator_insert_peg-v0
# dmc_pendulum_swingup-v0
# dmc_point_mass_easy-v0
# dmc_point_mass_hard-v0
# dmc_quadruped_walk-v0
# dmc_quadruped_run-v0
# dmc_quadruped_escape-v0
# dmc_quadruped_fetch-v0
# dmc_reacher_easy-v0
# dmc_reacher_hard-v0
# dmc_stacker_stack_2-v0
# dmc_stacker_stack_4-v0
# dmc_swimmer_swimmer6-v0
# dmc_swimmer_swimmer15-v0
# dmc_walker_stand-v0
# dmc_walker_walk-v0
# dmc_walker_run-v0

