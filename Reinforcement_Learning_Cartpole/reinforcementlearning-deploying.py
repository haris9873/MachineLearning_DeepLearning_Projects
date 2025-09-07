import os
import gymnasium as gym  # open ai gym
from stable_baselines3 import PPO  # Proximal Policy Optimization
from stable_baselines3.common.vec_env import DummyVecEnv  # Vectorized Environment
from stable_baselines3.common.evaluation import evaluate_policy  # Evaluating the model

# Create Environment
env = gym.make("CartPole-v1", render_mode="human")
env = DummyVecEnv([lambda: env])  # Vectorized Environment
# Load Model that was saved previously
model = PPO.load(
    "Reinforcement_Learning_Cartpole/Training/Saved_Models/PPO_CartPole", env=env)
# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# Deploying the model, now we pass our observations after learning

episodes = 5
for episode in range(1, episodes+1):  # loop 1 to 5
    obs = env.reset()  # resetting environment for initial set of observations
    done = False
    score = 0  # reward

    while not done:
        env.render()  # view graphical environment

        # generate action
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(
            action)  # applying an action to env, and save info
        score += reward  # accmulate rewards

    print('Episode : {} Score: {}'.format(episode, score))
env.close()

# training log path

train_log_path = os.path.join(
    'Reinforcement_Learning_Cartpole/Training/Logs', 'PPO_1')

# run tensorboard in terminal
# cd to the training log path and run
# tensorboard --logdir=.
