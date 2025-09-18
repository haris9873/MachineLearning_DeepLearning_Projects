import gymnasium_robotics
import gymnasium as gym
import os
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
import torch

# remove if torch not installed
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


gym.register_envs(gymnasium_robotics)
# change as per env, fetch reach is the easiest to solve in low epochs
env_id = 'FetchReach-v4'
# env = gym.make(env_id, render_mode="human")
env = make_vec_env(env_id, n_envs=1, seed=42)

# Testing
print('Action Space: ', env.action_space.sample())
print('\nObservation Space: ', env.observation_space.sample())

# Create directories
os.makedirs('RL_Robotic_Arm/Training/Logs', exist_ok=True)
os.makedirs('RL_Robotic_Arm/Training/Saved_Models', exist_ok=True)
log_path = os.path.join('RL_Robotic_Arm/Training', 'Logs')
save_path = os.path.join('RL_Robotic_Arm/Training', 'Saved_Models')

""" 
Training the robot

"""

print('Training the model ...')
model = SAC(
    "MultiInputPolicy",
    env,
    tensorboard_log=log_path,
    replay_buffer_class=HerReplayBuffer,  # for sparse rewards
    replay_buffer_kwargs=dict(
        n_sampled_goal=4, goal_selection_strategy='future',),
    verbose=1,
    device=device
)

# Increase steps for more training and higher success rate, 4m as per paper
model.learn(total_timesteps=100000)

print('Training Complete')
# Save model

SAC_path = os.path.join(
    save_path, 'SAC_{}'.format(env_id))
model.save(SAC_path)

print("Training finished and final model saved to ", SAC_path)


"""
Testing the environment

"""

# obs, info = env.reset(seed=42)
# score = 0
# for _ in range(1000):
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     score += reward
#     if terminated or truncated:
#         obs, info = env.reset()

# print(f"Score: {score}")
# env.close()
