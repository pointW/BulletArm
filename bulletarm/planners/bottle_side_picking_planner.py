import numpy as np
from bulletarm.planners.block_structure_base_planner import BlockStructureBasePlanner
from bulletarm.pybullet.utils import constants

class BottleSidePickingPlanner(BlockStructureBasePlanner):
  def __init__(self, env, config):
    config['half_rotation'] = False
    super(BottleSidePickingPlanner, self).__init__(env, config)

  def getPickingAction(self):
    # return self.pickSecondTallestObjOnTop(self.env.getObjsOutsideBox())
    objects, object_poses = self.getSizeSortedObjPoses(objects=self.env.objects)

    x, y, z, r = object_poses[0][0], object_poses[0][1], object_poses[0][2], object_poses[0][5]
    r = np.random.random_sample() * 2 * np.pi if self.random_orientation else 0

    return self.encodeAction(constants.PICK_PRIMATIVE, x, y, z, r)

  def getStepsLeft(self):
    return 100
