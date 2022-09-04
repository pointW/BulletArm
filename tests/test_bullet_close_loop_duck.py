import unittest
import time
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from bulletarm import env_factory

class TestBulletCloseLoopBlockPicking(unittest.TestCase):
  env_config = {'corrupt': ['two_specular'], 'black_workspace': False, 'robot': 'empty', 'obs_size': 142}

  planner_config = {'random_orientation': True, 'dpos': 0.05, 'drot': np.pi / 4}

  def testPlanner2(self):
    self.env_config['render'] = True
    self.env_config['seed'] = 1
    num_processes = 1
    env = env_factory.createEnvs(num_processes,  'close_loop_duck', self.env_config, self.planner_config)
    total = 0
    s = 0
    step_times = []
    while True:
      (states_, in_hands_, obs_) = env.reset()
      fig, axs = plt.subplots(1, 2, figsize=(10, 5))
      im = axs[0].imshow(np.moveaxis(obs_[0, :3], 0, 2))
      # plt.colorbar(im, ax=axs[0])
      im = axs[1].imshow(np.moveaxis(obs_[0, 3:], 0, 2))
      # plt.colorbar(im, ax=axs[1])
      plt.tight_layout()
      plt.show()