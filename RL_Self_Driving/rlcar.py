import os
import gymnasium as gym  # open ai gym
from stable_baselines3 import PPO
# Transform a vectorized environment to a stack of frames
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env_id = 'CarRacing-v3'
env = gym.make(env_id, render_mode="human")

model = PPO.load(
    "RL_Self_Driving/Training/Saved_Models/PPO_CarRacing", env=env)

mean_reward, std_reward = evaluate_policy(
    model, model.get_env(), n_eval_episodes=10, deterministic=True, render=True)
print("\nFinal Evaluation Results:")
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f} over 10 episodes")

# vec_env = model.get_env()

# episodes = 5
# for episode in range(1, episodes+1):  # loop 1 to 5
#     obs = vec_env.reset()  # resetting environment for initial set of observations
#     done = False
#     score = 0  # reward

#     while not done:
#         vec_env.render("human")  # view graphical environment
#         action, _states = model.predict(obs)
#         obs, rewards, done, info = vec_env.step(action)
#         score += rewards
#     print('Episode : {} Score: {}'.format(episode, score))
# vec_env.close()

# run tensorboard in terminal
# cd to the training log path and run
# tensorboard --logdir=.
