import time
import numpy as np
import matplotlib.pyplot as plt
import torch

import helping_hands_rl_envs.env_factory as env_factory

workspace = np.array([[0.35, 0.65], [-0.15, 0.15], [0, 1]])
env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 128, 'action_sequence': 'pxy',
              'num_objects': 3, 'render': False, 'fast_mode': True}
planner_config = {'pos_noise': 0.015}
envs = env_factory.createEnvs(1, 'data', 'pybullet', 'block_stacking', env_config, planner_config=planner_config)

state, in_hand_img, obs = envs.reset()
done = False
while not done:
  plt.imshow(obs.squeeze(), cmap='gray', vmin=0.0, vmax=0.1); plt.show()
  plt.imshow(in_hand_img.squeeze(), cmap='gray', vmin=0.0, vmax=0.1); plt.show()
  action = envs.getNextAction()
  state_, in_hand_img_, obs_, reward, done = envs.step(action)

  obs = obs_
  in_hand_img = in_hand_img_

plt.imshow(obs.squeeze(), cmap='gray', vmin=0.0, vmax=0.1); plt.show()
plt.imshow(in_hand_img.squeeze(), cmap='gray', vmin=0.0, vmax=0.1); plt.show()