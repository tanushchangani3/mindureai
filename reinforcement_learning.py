import numpy as np
import gym
from stable_baselines3 import PPO

class LearningEnv(gym.Env):
    def __init__(self):
        super(LearningEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

    def reset(self):
        return np.random.rand(10)

    def step(self, action):
        state = np.random.rand(10)
        reward = np.random.rand()
        done = np.random.rand() > 0.95
        return state, reward, done, {}

if __name__ == "__main__":
    env = LearningEnv()
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("learning_path_model")