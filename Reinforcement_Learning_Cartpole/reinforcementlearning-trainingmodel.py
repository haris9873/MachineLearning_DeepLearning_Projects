import os
import gymnasium as gym  # open ai gym
from stable_baselines3 import PPO  # Proximal Policy Optimization
from stable_baselines3.common.vec_env import DummyVecEnv  # Vectorized Environment
from stable_baselines3.common.evaluation import evaluate_policy  # Evaluating the model
# adding dependencies for callback
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

"""
# Observation Space Cartpole

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

# Action Space Cartpole
   | Num | Action                 |
    |-----|------------------------|
    | 0   | Push cart to the left  |
    | 1   | Push cart to the right |

    ### Rewards

    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
    including the termination step, is allotted. The threshold for rewards is 475 for v1.

    ### Starting State

    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ### Episode End

    The episode ends if any one of the following occurs:

    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)
"""

# Create the environment
env = gym.make("CartPole-v1", render_mode="human")

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
eval_callback = EvalCallback(env, callback_on_new_best=stop_callback,
                             eval_freq=10000, best_model_save_path='Reinforcement_Learning_Cartpole/Training/Saved_Models', verbose=1)

# Train RL model
# define log path
log_path = os.path.join('Reinforcement_Learning_Cartpole/Training/Logs')
print(log_path)

env = DummyVecEnv([lambda: env])  # Vectorized Environment
# using MlpPolicy = Multi Layer Perceptron
model = PPO("MlpPolicy", env, verbose=1,
            # PPO, A2C etc. are faster on CPU unless using CNNs, but my cpu is slow compared to gpu, so hehe
            tensorboard_log=log_path, device='cuda')
# timesteps are dependent on the environment
model.learn(total_timesteps=20000)

# Save and Reload Model

PPO_path = os.path.join(
    'Reinforcement_Learning_Cartpole/Training/Saved_Models', 'PPO_CartPole')
model.save(PPO_path)

# testing the model in an environment

# episodes = 5
# for episode in range(1, episodes+1):  # loop 1 to 5
#     obs, info = env.reset()  # resetting environment for initial set of observations
#     done = False
#     score = 0  # reward

#     while not done:
#         env.render()  # view graphical environment

#         # generate random action
#         action = env.action_space.sample()
#         obs, reward, terminated, truncated, info = env.step(
#             action)  # applying an action to env, and save info
#         score += reward  # accmulate rewards
#         done = terminated or truncated
#     print('Episode : {} Score: {}'.format(episode, score))
# env.close()

# # We pass these observations for the environment to out reinforcement learning agent to learn
