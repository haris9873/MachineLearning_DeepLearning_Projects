import os
import gymnasium as gym  # open ai gym
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from ale_py import ALEInterface  # for the latest ALE versions of Atari envs
import ale_py

ale = ALEInterface()
gym.register_envs(ale_py)  # unnecessary but helpful for IDEs

env = make_atari_env('ALE/Breakout-v5', n_envs=1)
env = VecFrameStack(env, n_stack=4)

# Load Model that was saved previously
model = A2C.load(
    "RL_Atari/Training/Saved_Models/A2C_Breakout", env)

mean_reward, std_reward = evaluate_policy(
    model, model.get_env(), n_eval_episodes=10, deterministic=True)
print("\nFinal Evaluation Results:")
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f} over 10 episodes")

vec_env = model.get_env()
obs = vec_env.reset()

# Test environment
done = False
score = 0
obs = vec_env.reset()
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = vec_env.step(action)
    score += rewards
    vec_env.render("human")
print('Score: {}'.format(score))
env.close()


# run tensorboard in terminal
# cd to the training log path and run
# tensorboard --logdir=.
