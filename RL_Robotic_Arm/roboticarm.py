import gymnasium_robotics
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy


env = gym.make('FetchPickAndPlace-v4',
               render_mode="human", max_episode_steps=50)
env.metadata["render_fps"] = 1
model = SAC.load(
    "RL_Robotic_Arm/Training/Saved_Models/SAC_FetchPickAndPlace", env=env
)
mean_reward, std_reward = evaluate_policy(
    model, model.get_env(), n_eval_episodes=50, deterministic=True)
print("\nFinal Evaluation Results:")
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f} over 20 episodes")

env.close()
