import os
import gymnasium as gym  # open ai gym
from stable_baselines3 import A2C
# Transform a vectorized environment to a stack of frames
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from ale_py import ALEInterface
import ale_py
# --- Ensure CUDA is available ---

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

ale = ALEInterface()

gym.register_envs(ale_py)  # unnecessary but helpful for IDEs
# Vectorized Environment
vec_env = make_atari_env('ALE/Breakout-v5', n_envs=4)
stacked_vec_env = VecFrameStack(vec_env, n_stack=4)
# We use a single environment for evaluation to get more stable metrics.
eval_env = make_atari_env('ALE/Breakout-v5', n_envs=1,
                          seed=1)  # Different seed for eval
eval_env = VecFrameStack(eval_env, n_stack=4)

# Make directories
log_dir = 'RL_Atari/Training/Logs'
saved_model_dir = 'RL_Atari/Training/Saved_Models'

os.makedirs(log_dir, exist_ok=True)
os.makedirs(saved_model_dir, exist_ok=True)

log_path = os.path.join(
    'RL_Atari/Training', 'Logs')
print('Logging to : ', log_path)

# --- Callbacks ---
# Stop training when the reward threshold is met
reward_threshold = 500  # Example threshold for Breakout (adjust as needed)
stop_callback = StopTrainingOnRewardThreshold(
    reward_threshold=reward_threshold, verbose=1)

# Evaluate the agent periodically and save the best model
# Adjust eval_freq based on your total_timesteps and desired evaluation frequency
eval_freq = 50000  # Evaluate every 50000 steps
eval_callback = EvalCallback(eval_env, callback_on_new_best=stop_callback,
                             best_model_save_path=saved_model_dir,
                             log_path=log_path,
                             eval_freq=eval_freq,
                             deterministic=True, render=False)


# Training the model
print('Training the model ...')
model = A2C("CnnPolicy", stacked_vec_env, verbose=1,
            tensorboard_log=log_path, device=device)

model.learn(total_timesteps=100000, callback=eval_callback)
print('Training Complete')
# Save and Reload Model

# A2C_path = os.path.join(
#     saved_model_dir, 'A2C_Breakout')
# model.save(A2C_path)

# print("Training finished and final model saved to ", A2C_path)
