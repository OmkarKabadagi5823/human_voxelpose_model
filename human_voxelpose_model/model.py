from scipy.spatial.transform import Rotation
import numpy as np
import math

import time
import logging
import pprint

logging.basicConfig(level=logging.INFO, format='[%(levelname)s %(asctime)-8s] %(message)s')

class HumanPoseModel():
    joint_names = {
    "hip": 0,
    "r_hip": 1,
    "r_knee": 2,
    "r_foot": 3,
    "l_hip": 4,
    "l_knee": 5,
    "l_foot": 6,
    "nose": 7,
    "c_shoulder": 8,
    "r_shoulder": 9,
    "r_elbow": 10,
    "r_wrist": 11,
    "l_shoulder": 12,
    "l_elbow": 13,
    "l_wrist": 14,
    }

    links = [
        [0,1],
        [1,2],
        [2,3],
        [0,4],
        [4,5],
        [5,6],
        [1,9],
        [4,12],
        [8,7],
        [8,9],
        [8,12],
        [9,10],
        [10,11],
        [12,13],
        [13,14]
    ]

    def __init__(self):
        self.joint_set = None
        self.rot = dict()

    def vec(self, start_idx, end_idx):
        return self.joint_set[end_idx] - self.joint_set[start_idx]

    def update(self, joint_set):
        self.joint_set = joint_set
        self.__update()

    def __get_root_basis(self):
        x_cap = np.cross(self.vec(0, 4), self.vec(0, 8))
        x_cap[2] = 0
        x_cap = x_cap / np.linalg.norm(x_cap)

        z_cap = np.array([0, 0, 1])
        
        y_cap = np.cross(z_cap, x_cap)

        logging.debug(f'root_basis: x_cap: {x_cap}, y_cap: {y_cap}, z_cap: {z_cap}')

        x_cap = np.expand_dims(x_cap, axis=1)
        y_cap = np.expand_dims(y_cap, axis=1)
        z_cap = np.expand_dims(z_cap, axis=1)

        return x_cap, y_cap, z_cap

    def __get_hip_basis(self):
        x_cap = np.cross(self.vec(0, 4), self.vec(0, 8))
        x_cap = x_cap / np.linalg.norm(x_cap)

        y_cap = np.cross(self.vec(0, 8), x_cap)
        y_cap = y_cap / np.linalg.norm(y_cap)

        z_cap = np.cross(x_cap, y_cap)

        logging.debug(f'hip_basis: x_cap: {x_cap}, y_cap: {y_cap}, z_cap: {z_cap}')

        x_cap = np.expand_dims(x_cap, axis=1)
        y_cap = np.expand_dims(y_cap, axis=1)
        z_cap = np.expand_dims(z_cap, axis=1)

        return x_cap, y_cap, z_cap

    def __get_neck_basis(self):
        y_cap = np.cross(self.vec(0, 8), self.vec(8, 7))
        y_cap = y_cap / np.linalg.norm(y_cap)
        
        offset = Rotation.from_rotvec((-np.pi/3) * y_cap)
        z_cap = offset.as_matrix().dot(self.vec(8, 7))
        z_cap = z_cap / np.linalg.norm(z_cap)

        x_cap = np.cross(y_cap, z_cap)

        x_cap = np.expand_dims(x_cap, axis=1)
        y_cap = np.expand_dims(y_cap, axis=1)
        z_cap = np.expand_dims(z_cap, axis=1)

        return x_cap, y_cap, z_cap

    def __get_shoulder_basis(
        self,
        shoulder_joint_idx,
        elbow_joint_idx,
        wrist_joint_idx
    ):
        z_cap = self.vec(elbow_joint_idx, shoulder_joint_idx)
        z_cap = z_cap / np.linalg.norm(z_cap)

        y_cap = np.cross(
            self.vec(wrist_joint_idx, elbow_joint_idx),
            self.vec(elbow_joint_idx, shoulder_joint_idx)
        )
        y_cap = y_cap / np.linalg.norm(y_cap)

        x_cap = np.cross(y_cap, z_cap)

        logging.debug(f'shoulder_{shoulder_joint_idx}_basis: x_cap: {x_cap}, y_cap: {y_cap}, z_cap: {z_cap}')

        x_cap = np.expand_dims(x_cap, axis=1)
        y_cap = np.expand_dims(y_cap, axis=1)
        z_cap = np.expand_dims(z_cap, axis=1)

        return x_cap, y_cap, z_cap

    def __get_elbow_angle(
        self,
        shoulder_joint_idx,
        elbow_joint_idx,
        wrist_joint_idx
    ):
        dot_product = self.vec(wrist_joint_idx, elbow_joint_idx).T.dot(
            self.vec(elbow_joint_idx, shoulder_joint_idx)
        ).squeeze()

        norm_w_e = np.linalg.norm(self.vec(wrist_joint_idx, elbow_joint_idx))
        norm_e_s = np.linalg.norm(self.vec(elbow_joint_idx, shoulder_joint_idx))

        phi = abs(math.acos(dot_product / (norm_w_e * norm_e_s)))

        logging.debug(f'elbow_{elbow_joint_idx}_angle: {phi}')

        return phi

    def __get_thigh_basis(
        self, 
        hip_joint_idx, 
        knee_joint_idx, 
        ankle_joint_idx
    ):
        z_cap = self.vec(knee_joint_idx, hip_joint_idx)
        z_cap = z_cap / np.linalg.norm(z_cap)

        y_cap = np.cross(
            self.vec(hip_joint_idx, knee_joint_idx),
            self.vec(knee_joint_idx, ankle_joint_idx)
        )    
        y_cap = y_cap / np.linalg.norm(y_cap)

        x_cap = np.cross(y_cap, z_cap)

        logging.debug(f'hip_{hip_joint_idx}_basis: x_cap: {x_cap}, y_cap: {y_cap}, z_cap: {z_cap}')

        x_cap = np.expand_dims(x_cap, axis=1)
        y_cap = np.expand_dims(y_cap, axis=1)
        z_cap = np.expand_dims(z_cap, axis=1)

        return x_cap, y_cap, z_cap

    def __get_knee_angle(
        self, 
        hip_joint_idx, 
        knee_joint_idx, 
        ankle_joint_idx
    ):
        dot_product = self.vec(hip_joint_idx, knee_joint_idx).T.dot(
            self.vec(knee_joint_idx, ankle_joint_idx)
        ).squeeze()

        norm_w_e = np.linalg.norm(self.vec(hip_joint_idx, knee_joint_idx))
        norm_e_s = np.linalg.norm(self.vec(knee_joint_idx, ankle_joint_idx))

        phi = -abs(math.acos(dot_product / (norm_w_e * norm_e_s)))

        logging.debug(f'knee_{knee_joint_idx}_angle: {phi}')

        return phi

    def __compute_all_rotation_matrices(self):
        _rot_mat =  np.hstack(self.__get_root_basis())
        self.rot['w|r'] = Rotation.from_matrix(_rot_mat)

        _rot_mat = np.hstack(self.__get_hip_basis())
        self.rot['w|0'] = Rotation.from_matrix(_rot_mat)

        _rot_mat = np.hstack(self.__get_neck_basis())
        self.rot['w|8'] = Rotation.from_matrix(_rot_mat)

        _rot_mat = np.hstack(self.__get_shoulder_basis(9, 10, 11))
        self.rot['w|9'] = Rotation.from_matrix(_rot_mat)

        _rot_mat = np.hstack(self.__get_shoulder_basis(12, 13, 14))
        self.rot['w|12'] = Rotation.from_matrix(_rot_mat)

        angle = self.__get_elbow_angle(9, 10, 11)
        self.rot['9|10'] = angle

        angle = self.__get_elbow_angle(12, 13, 14)
        self.rot['12|13'] = angle

        _rot_mat = np.hstack(self.__get_thigh_basis(1, 2, 3))
        self.rot['w|1'] = Rotation.from_matrix(_rot_mat)

        _rot_mat = np.hstack(self.__get_thigh_basis(4, 5, 6))
        self.rot['w|4'] = Rotation.from_matrix(_rot_mat)

        angle = self.__get_knee_angle(1, 2, 3)
        self.rot['1|2'] = angle

        angle = self.__get_knee_angle(4, 5, 6)
        self.rot['4|5'] = angle

    def __compute_change_of_reference_frame(self):
        self.rot['r|0'] = self.rot['w|r'].inv() * self.rot['w|0']
        self.rot['0|8'] = self.rot['w|0'].inv() * self.rot['w|8']
        self.rot['0|9'] = self.rot['w|0'].inv() * self.rot['w|9']
        self.rot['0|12'] = self.rot['w|0'].inv() * self.rot['w|12']
        self.rot['r|1'] = self.rot['w|r'].inv() * self.rot['w|1']
        self.rot['r|4'] = self.rot['w|r'].inv() * self.rot['w|4']

    def __update(self):
        self.__compute_all_rotation_matrices()
        self.__compute_change_of_reference_frame()
        