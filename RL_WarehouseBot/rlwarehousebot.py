import sys
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C
import gymnasium as gym  # open ai gym
import os
sys.path.insert(1, 'RL_WarehouseBot/Custom_Environment')  # nopep8
import WarehouseBot_v0_env  # nopep8


env_id = 'WarehouseBot-v0'
env = gym.make(env_id, render_mode="human")


# Define Paths
os.makedirs('RL_WarehouseBot/Training/Logs', exist_ok=True)
os.makedirs('RL_WarehouseBot/Training/Saved_Models', exist_ok=True)

log_path = os.path.join('RL_WarehouseBot/Training', 'Logs')
save_path = os.path.join('RL_WarehouseBot/Training', 'Saved_Models')
# Define model
model = A2C('MlpPolicy', env, verbose=1,
            tensorboard_log=log_path, device='cuda')

# Train the model
print('Training the model ...')
TIMESTEPS = 1000
iters = 0
while iters <= 1000:
    iters += 1

    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)  # train
    # Save a trained model every TIMESTEPS
    model.save(f"{save_path}/a2c_{TIMESTEPS*iters}")


A2C_path = os.path.join(
    save_path, 'A2C_1000')


del model

model = A2C.load(A2C_path, env=env)
mean_reward, std_reward = evaluate_policy(
    model, model.get_env(), n_eval_episodes=10, deterministic=True, render=True)
print("\nFinal Evaluation Results:")
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f} over 10 episodes")
