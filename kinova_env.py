import numpy as np
import copy
from typing import Optional
import os
import pinocchio as pin
import pybullet as p
import pybullet_data
from pinocchio.robot_wrapper import RobotWrapper
from scipy.spatial.transform import Rotation

class KinovaEnv:
    def __init__(self):
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        # Load the plane
        p.loadURDF("plane.urdf")
        # Load the Kinova Gen3 robot
        script_dir = os.path.dirname(os.path.abspath(__file__))
        robot_path = os.path.join(script_dir, "kinova_main.urdf")
        self.robot_id = p.loadURDF(robot_path)
        # Build pinocchio robot
        self.robot = RobotWrapper.BuildFromURDF(robot_path)
        
        # Get active joint indices
        self.active_joint_ids = [2, 4, 6, 8, 10, 12, 14, 21, 23, 25, 26, 27, 28]
        
        # Disable the velocity control for the joints
        p.setJointMotorControlArray(
            self.robot_id,
            self.active_joint_ids,
            p.VELOCITY_CONTROL,
            forces=np.zeros(len(self.active_joint_ids))
            )
        
        # Get number of joints
        self.n_joints = p.getNumJoints(self.robot_id)
        
        # frame ids
        self.BASE_FRAME_ID = self.robot.model.getFrameId("base_link")
        self.LINK2_FRAME_ID = self.robot.model.getFrameId("shoulder_link")
        self.LINK3_FRAME_ID = self.robot.model.getFrameId("half_arm_1_link")
        self.LINK4_FRAME_ID = self.robot.model.getFrameId("half_arm_2_link")
        self.LINK5_FRAME_ID = self.robot.model.getFrameId("forearm_link")
        self.LINK6_FRAME_ID = self.robot.model.getFrameId("spherical_wrist_1_link")
        self.LINK7_FRAME_ID = self.robot.model.getFrameId("spherical_wrist_2_link")
        self.EE_FRAME_ID = self.robot.model.getFrameId("end_effector_link")
        self.HAND_FRAME_ID = self.robot.model.getFrameId("robotiq_85_base_link")
        
        # frame ids for bounding box
        self.LINK2_BOUNDING_BOX_FRAME_ID = self.robot.model.getFrameId("shoulder_link_bounding_box")
        self.LINK3_BOUNDING_BOX_FRAME_ID = self.robot.model.getFrameId("half_arm_1_bounding_box")
        self.LINK4_BOUNDING_BOX_FRAME_ID = self.robot.model.getFrameId("half_arm_2_bounding_box")
        self.LINK5_BOUNDING_BOX_FRAME_ID = self.robot.model.getFrameId("forearm_bounding_box")
        self.LINK6_BOUNDING_BOX_FRAME_ID = self.robot.model.getFrameId("spherical_wrist_1_bounding_box")
        self.LINK7_BOUNDING_BOX_FRAME_ID = self.robot.model.getFrameId("spherical_wrist_2_bounding_box")
        self.EE_BOUNDING_BOX_FRAME_ID = self.robot.model.getFrameId("EE_bounding_box")
        # self.HAND_FRAME_ID = self.robot.model.getFrameId("robotiq_85_base_link")
        # Get frame ID for grasp target
        self.jacobian_frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
          
    
    def get_info(self, q, dq):
        """
        info contains:
        -------------------------------------
        q: joint position
        dq: joint velocity
        f(x): drift
        g(x): control influence matrix
        G: gravitational vector
        J_EE: end-effector Jacobian
        dJ_EE: time derivative of end-effector Jacobian
        pJ_EE: pseudo-inverse of end-effector Jacobian
        R_LINK5_1: rotation matrix of the first part of the 5th link
        P_LINK5_1: postion vector of the first part of the 5th link
        q_LINK5_1: quaternion of the first part of the 5th link [x, y, z, w]
        R_LINK5_2: rotation matrix of the second part of the 5th link
        P_LINK5_2: postion vector of the second part of the 5th link
        q_LINK5_2: quaternion of the second part of the 5th link [x, y, z, w]
        R_LINK6: rotation matrix of the 6th link
        P_LINK6: postion vector of the 6th link
        q_LINK6: quaternion of the 6th link [x, y, z, w]
        R_LINK7: rotation matrix of the 7th link
        P_LINK7: postion vector of the 7th link
        q_LINK7: quaternion of the 7th link [x, y, z, w]
        R_HAND: rotation matrix of the hand
        P_HAND: position vector of the hand
        q_HAND: quaternion of the hand [x, y, z, w]
        R_EE: end-effector rotation matrix
        P_EE: end-effector position vector
        """

        # Get Jacobian from grasp target frame
        # preprocessing is done in get_state_update_pinocchio()
        jacobian = self.robot.getFrameJacobian(self.EE_FRAME_ID, self.jacobian_frame)
        jacobian_link2 = self.robot.getFrameJacobian(
            self.LINK2_BOUNDING_BOX_FRAME_ID, self.jacobian_frame
        )
        jacobian_link3 = self.robot.getFrameJacobian(
            self.LINK3_BOUNDING_BOX_FRAME_ID, self.jacobian_frame
        )
        jacobian_link4 = self.robot.getFrameJacobian(
            self.LINK4_BOUNDING_BOX_FRAME_ID, self.jacobian_frame
        )
        jacobian_link5 = self.robot.getFrameJacobian(
            self.LINK5_BOUNDING_BOX_FRAME_ID, self.jacobian_frame
        )
        jacobian_link6 = self.robot.getFrameJacobian(
            self.LINK6_BOUNDING_BOX_FRAME_ID, self.jacobian_frame
        )
        jacobian_link7 = self.robot.getFrameJacobian(
            self.LINK7_BOUNDING_BOX_FRAME_ID, self.jacobian_frame
        )

        # Get pseudo-inverse of frame Jacobian
        pinv_jac = np.linalg.pinv(jacobian)

        dJ = pin.getFrameJacobianTimeVariation(
            self.robot.model, self.robot.data, self.EE_FRAME_ID, self.jacobian_frame
        )

        f, g, M, Minv, nle = self.get_dynamics(q, dq)

        # compute the position and rotation of the crude models
        p_link2, R_link2, q_LINK2 = self.compute_crude_location(
            np.eye(3), np.zeros((3, 1)), self.LINK2_BOUNDING_BOX_FRAME_ID
        )

        p_link3, R_link3, q_LINK3 = self.compute_crude_location(
            np.eye(3), np.zeros((3, 1)), self.LINK3_BOUNDING_BOX_FRAME_ID
        )

        p_link4, R_link4, q_LINK4 = self.compute_crude_location(
            np.eye(3), np.zeros((3, 1)), self.LINK4_BOUNDING_BOX_FRAME_ID
        )

        p_link5, R_link5, q_LINK5 = self.compute_crude_location(
            np.eye(3), np.zeros((3, 1)), self.LINK5_BOUNDING_BOX_FRAME_ID
        )

        p_link6, R_link6, q_LINK6 = self.compute_crude_location(
            np.eye(3), np.zeros((3, 1)), self.LINK6_BOUNDING_BOX_FRAME_ID
        )

        p_link7, R_link7, q_LINK7 = self.compute_crude_location(
            np.eye(3), np.zeros((3, 1)), self.LINK7_BOUNDING_BOX_FRAME_ID
        )

        p_end, R_end, q_end = self.compute_crude_location(
            np.eye(3), np.zeros((3, 1)), self.EE_BOUNDING_BOX_FRAME_ID
        )

        info = {
            "q": q,
            "dq": dq,
            "f(x)": f,
            "g(x)": g,
            "M(q)": M,
            "M(q)^{-1}": Minv,
            "nle": nle,
            "G": self.robot.gravity(q),
            "pJ_EE": pinv_jac,
            "R_LINK2": copy.deepcopy(R_link2),
            "P_LINK2": copy.deepcopy(p_link2),
            "q_LINK2": copy.deepcopy(q_LINK2),
            "J_LINK2": jacobian_link2,
            "R_LINK3": copy.deepcopy(R_link3),
            "P_LINK3": copy.deepcopy(p_link3),
            "q_LINK3": copy.deepcopy(q_LINK3),
            "J_LINK3": jacobian_link3,
            "R_LINK4": copy.deepcopy(R_link4),
            "P_LINK4": copy.deepcopy(p_link4),
            "q_LINK4": copy.deepcopy(q_LINK4),
            "J_LINK4": jacobian_link4,
            "R_LINK5": copy.deepcopy(R_link5),
            "P_LINK5": copy.deepcopy(p_link5),
            "q_LINK5": copy.deepcopy(q_LINK5),
            "J_LINK5": jacobian_link5,
            "R_LINK6": copy.deepcopy(R_link6),
            "P_LINK6": copy.deepcopy(p_link6),
            "q_LINK6": copy.deepcopy(q_LINK6),
            "J_LINK6": jacobian_link6,
            "R_LINK7": copy.deepcopy(R_link7),
            "P_LINK7": copy.deepcopy(p_link7),
            "q_LINK7": copy.deepcopy(q_LINK7),
            "J_LINK7": jacobian_link7,
            "R_HAND": copy.deepcopy(R_end),
            "P_HAND": copy.deepcopy(p_end),
            "q_HAND": copy.deepcopy(q_end),
            "J_EE": jacobian,
            "dJ_EE": dJ,
            "R_EE": copy.deepcopy(self.robot.data.oMf[self.EE_FRAME_ID].rotation),
            "P_EE": copy.deepcopy(self.robot.data.oMf[self.EE_FRAME_ID].translation),
        }

        return info
    
    def reset(
        self,
        *,
        cameraDistance=1.4,
        cameraYaw=66.4,
        cameraPitch=-16.2,
        lookat=[0.0, 0.0, 0.0],
    ):
        # super().reset(seed=seed)

        target_joint_angles = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ]

        self.q_nominal = np.array(target_joint_angles)

        for i, joint_ang in enumerate(target_joint_angles):
            p.resetJointState(self.robot_id, self.active_joint_ids[i], joint_ang, 0.0)

        q, dq = self.get_state_update_pinocchio()
        info = self.get_info(q, dq)

        p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, lookat)

        return info
    
    def step(self, action):
        # Apply action in pybullet
        self.send_joint_command(action)
        p.stepSimulation()
        
        # Get the current state in pinocchio
        q, dq = self.get_state_update_pinocchio()
        
        info = self.get_info(q, dq)

        return info
    
    def get_state(self):
        q = np.zeros(13)
        dq = np.zeros(13)
        # k = 0
        for i, id in enumerate(self.active_joint_ids):
            _joint_state = p.getJointState(self.robot_id, id)
            # if i not in [2, 4, 6]:
            q[i], dq[i] = _joint_state[0], _joint_state[1]
            # else :
            #     q[k] = np.cos(_joint_state[0])
            #     k = k + 1
            #     q[k] = np.sin(_joint_state[0])
            #     dq[i] = _joint_state[1]
            # k = k + 1

        return q, dq
    
    def get_dynamics(self, q, dq):
        """
        f.shape = (18, 1), g.shape = (18, 9)
        """
        Minv = pin.computeMinverse(self.robot.model, self.robot.data, q)
        M = self.robot.mass(q)
        nle = self.robot.nle(q, dq)

        f = np.vstack((dq[:, np.newaxis], -Minv @ nle[:, np.newaxis]))
        g = np.vstack((np.zeros((26, 13)), Minv))
        
        return f, g, M, Minv, nle
    
    def update_pinocchio(self, q, dq):
        self.robot.computeJointJacobians(q)
        self.robot.framesForwardKinematics(q)
        self.robot.centroidalMomentum(q, dq)
        
    def get_state_update_pinocchio(self):
        q, dq = self.get_state()
        self.update_pinocchio(q, dq)

        return q, dq
    
    def send_joint_command(self, tau):
        zeroGains = tau.shape[0] * (0.0,)
        p.setJointMotorControlArray(
            self.robot_id,
            self.active_joint_ids,
            p.TORQUE_CONTROL,
            forces=tau,
            positionGains=zeroGains,
            velocityGains=zeroGains,
        )

        
    def compute_crude_location(self, R_offset, p_offset, frame_id):
        # get link orientation and position

        _p = self.robot.data.oMf[frame_id].translation
        _Rot = self.robot.data.oMf[frame_id].rotation

        # compute link transformation matrix
        _T = np.hstack((_Rot, _p[:, np.newaxis]))
        T = np.vstack((_T, np.array([[0.0, 0.0, 0.0, 1.0]])))

        # compute link offset transformation matrix
        _TB = np.hstack((R_offset, p_offset))
        TB = np.vstack((_TB, np.array([[0.0, 0.0, 0.0, 1.0]])))

        # get transformation matrix
        T_mat = T @ TB

        # compute crude model location
        p = (T_mat @ np.array([[0.0], [0.0], [0.0], [1.0]]))[:3, 0]

        # compute crude model orientation
        Rot = T_mat[:3, :3]

        # quaternion
        q = Rotation.from_matrix(Rot).as_quat()

        return p, Rot, q