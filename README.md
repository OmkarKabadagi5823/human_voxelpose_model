# Human Voxelpose Model
## Description
`human_voxelpose_model` is a package for estimating the joint angles from human pose. The input format should be in a format that is similar to the output produced by [voxelpose](https://github.com/microsoft/voxelpose-pytorch) model.

## Voxelpose Output Format
Voxelpose is a Multi-Camera 3D Human Pose Estimation model which estimates 3D poses of multiple people from multiple camera views. The output of voxelpose is a position estimate in 3D-space of 15 human joints which are:
-  0: hip (hip centre)
-  1: r_hip (right hip)
-  2: r_knee (right knee)
-  3: r_foot (right foot)
-  4: l_hip (left hip)
-  5: l_knee (left knee)
-  6: l_foot (left foot)
-  7: nose
-  8: c_shoulder (shoulder centre)
-  9: r_shoulder (right shoulder)
- 10: r_elbow (right elbow)
- 11: r_wrist (right wrist)
- 12: l_shoulder (left shoulder)
- 13: l_elbow (left elbow)
- 14: l_wrist (left wrist)

This is output as `15 x 3` tensor which represents the `x`, `y` and `z` position coordinates of the `15` joints.

## Getting Started
### Installation
```bash
# human_voxelpose_model installation
cd ~/Downloads
git clone https://github.com/OmkarKabadagi5823/human_voxelpose_model.git
cd human_voxelpose_model/
python -m build
cd dist/
pip install human_voxelpose_model-<version>-py3-none-any.whl
```

### Using HumanPoseModel
The `HumanPoseModel` class provides the implementation of estimating the joint angles from a given pose. There is method exposed in public API which is the `update()` method which takes `np.array` of shape `15 x 3` which represents the position of the 15 joints and updates the joint angles accordingly. `<HumanPoseModel>.rot` dictionary can be used to access the the joint angles which are stored as `scipy.spatial.transform.Rotation`. This allows the user to then read the joint angles in their required format (rotation matrix, euler angles, quaternions or rotation vectors).

#### root frame
`root` frame acts as the base frame for the human model. This frame is centered at the hip, with its x-axis coming out of the hip towards the front with the frame always parallel to the ground plane.

#### Valid keys for HumanPoseModel.rot
- 'w|r'     (root frame in world frame)
- 'w|0'     (hip in world frame)
- 'w|8'     (neck in world frame)
- 'w|9'     (r_shoulder in world frame)
- 'w|12'    (l_shoulder in world frame)
- '9|10'    (r_elbow in r_shoulder frame) # Note that this simply gives the angle in radians as elbow is revolute joint
- '12|13'   (l_elbow in l_shoulder frame) # Note that this simply gives the angle in radians as elbow is revolute joint
- 'w|1'     (r_hip in world frame)
- 'w|4'     (l_hip in world frame)
- '1|2'     (r_knee in r_hip frame) # Note that this simply gives the angle in radians as knee is revolute joint
- '4|5'     (l_knee in l_hip frame) # Note that this simply gives the angle in radians as knee is revolute joint
- 'r|0'     (hip in root frame)
- '0|8'     (neck in hip frame)
- '0|9'     (r_shoulder in hip frame)
- '0|12'    (l_shoulder in hip frame)
- 'r|1'     (r_hip in root frame)
- 'r|4'     (l_hip in root frame)
  
#### Example
```python
from human_voxelpose_model import HumanPoseModel
from scipy.spatial.transform import Rotation

hpm = HumanPoseModel()

# assuming a function get_human_pose which returns human pose as [15 x 3]
joint_position_set = get_human_pose() 

hpm.update(joint_position_set)
rot = hpm.rot['0|9'] # rotation of r_shoulder (9) in frame of hip (0)

# access the rotation
rot.as_quat() # alternatively, rot.as_euler('XYZ') or rot.as_matrix() or rot.as_rotvec()
```
