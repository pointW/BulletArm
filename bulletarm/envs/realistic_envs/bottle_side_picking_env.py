from copy import deepcopy
import numpy as np
from bulletarm.envs.base_env import BaseEnv
from bulletarm.pybullet.utils import constants

class BottleSidePickingEnv(BaseEnv):
  '''
  '''
  def __init__(self, config):
    # env specific parameters
    config['half_rotation'] = False
    config['pick_top_down_approach'] = True
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [0.8, 0.8]
    if 'num_objects' not in config:
      config['num_objects'] = 1
    if 'max_steps' not in config:
      config['max_steps'] = 1
    super(BottleSidePickingEnv, self).__init__(config)

  def _decodeAction(self, action):
    """
    decode input action base on self.action_sequence
    Args:
      action: action tensor

    Returns: motion_primative, x, y, z, rot

    """
    primative_idx, x_idx, y_idx, z_idx, rot_idx = map(lambda a: self.action_sequence.find(a), ['p', 'x', 'y', 'z', 'r'])
    motion_primative = action[primative_idx] if primative_idx != -1 else 0
    x = action[x_idx]
    y = action[y_idx]
    z = action[z_idx] if z_idx != -1 else self._getPrimativeHeight(motion_primative, x, y)
    rz, ry, rx = 0, 0, 0
    assert self.action_sequence.count('r') == 1
    rz = action[rot_idx] if rot_idx != -1 else 0
    ry = 0
    rx = np.pi/2

    rot = (rx, ry, rz)

    return motion_primative, x, y, z, rot

  def step(self, action):
    self.takeAction(action)
    self.wait(100)
    obs = self._getObservation(action)
    done = self._checkTermination()
    reward = 1.0 if done else 0.0

    if not done:
      done = self.current_episode_steps >= self.max_steps or not self.isSimValid()
    self.current_episode_steps += 1

    return obs, reward, done

  def reset(self):
    ''''''
    self.resetPybulletWorkspace()
    self._generateShapes(constants.BOTTLE, self.num_obj, random_orientation=self.random_orientation, padding=0.05, min_distance=self.min_object_distance, model_id=1)
    self.obj_grasped = 0
    return self._getObservation()
  
  def _getObservation(self, action=None):
    ''''''
    self.heightmap = self._getHeightmap()
    in_hand_img = self.getEmptyInHand()

    return self._isHolding(), in_hand_img, self.heightmap.reshape([1, self.heightmap_size, self.heightmap_size])


  def _checkTermination(self):
    ''''''
    if self._isObjectHeld(self.objects[0]):
      return True
    else:
      return False

def createBottleSidePickingEnv(config):
  return BottleSidePickingEnv(config)

if __name__ == '__main__':
  from bulletarm.planners.bottle_side_picking_planner import BottleSidePickingPlanner
  workspace = np.asarray([[0.3, 0.5],
                          [-0.2, 0.2],
                          [0, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 1, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 2, 'action_sequence': 'pxyr', 'num_objects': 1, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'object_init_space_check': 'point', 'physics_mode': 'fast', 'object_scale_range': (0.8, 0.8), 
                'pick_top_down_approach': True, 'half_rotation': False}
  planner_config = {'random_orientation': True, 'half_rotation': False}

  env = BottleSidePickingEnv(env_config)
  planner = BottleSidePickingPlanner(env, planner_config)
  s, in_hand, obs = env.reset()
  while True:
    action = planner.getNextAction()
    obs, reward, done = env.step(action)
    if done:
      env.reset()