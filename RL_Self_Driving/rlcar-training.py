import os
import gymnasium as gym  # open ai gym
from stable_baselines3 import PPO
# Transform a vectorized environment to a stack of frames
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

env_id = 'CarRacing-v3'
env = gym.make(env_id)
# env = DummyVecEnv([lambda: env])  # Vectorized Environment

# Define Paths
os.makedirs('RL_Self_Driving/Training/Logs', exist_ok=True)
os.makedirs('RL_Self_Driving/Training/Saved_Models', exist_ok=True)

log_path = os.path.join('RL_Self_Driving/Training', 'Logs')
save_path = os.path.join('RL_Self_Driving/Training', 'Saved_Models')

# stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
# eval_callback = EvalCallback(env, callback_on_new_best=stop_callback,
#                              best_model_save_path=save_path,
#                              log_path=log_path,
#                              eval_freq=10000,
#                              deterministic=True, render=False)
# Train the model
print('Training the model ...')
model = PPO('CnnPolicy', env, verbose=1,
            tensorboard_log=log_path, device='cuda')

model.learn(total_timesteps=100000)

print('Training Complete')
# Save model

PPO_path = os.path.join(
    save_path, 'PPO_CarRacing')
model.save(PPO_path)

print("Training finished and final model saved to ", PPO_path)
