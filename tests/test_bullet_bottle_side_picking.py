import unittest
import time
import numpy as np

import matplotlib.pyplot as plt

from bulletarm import env_factory

class TestBottleSidePicking(unittest.TestCase):
  env_config = {}

  planner_config = {}

  def testPlanner(self):
    self.env_config['render'] = True
    env = env_factory.createEnvs(1, 'bottle_side_picking', self.env_config, self.planner_config)
    env.reset()
    for i in range(100):
      action = env.getNextAction()
      (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=True)
    env.close()

TestBottleSidePicking().testPlanner()