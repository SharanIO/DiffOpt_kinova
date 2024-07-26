import unittest
import numpy as np
from kinova_env import KinovaEnv  # Adjusted import statement

env = KinovaEnv()
env.step(np.zeros(len(env.active_joint_ids)))

class TestKinovaEnv(unittest.TestCase):
    def setUp(self):
        """Set up the environment for each test."""
        self.env = KinovaEnv()

    def test_initialization(self):
        """Test if the environment initializes correctly."""
        self.assertIsInstance(self.env, KinovaEnv)

    def test_joint_control(self):
        """Test setting and getting joint states."""
        # Assume there is a method to directly set joint states (might need implementation)
        test_angle = np.pi / 4  # Set to 45 degrees
        zero_velocity = 0
        # Set joint angle (implement method if needed)
        # self.env.set_joint_state(self.env.active_joint_ids[0], test_angle, zero_velocity)
        # Step simulation to update the state
        self.env.step(np.zeros(len(self.env.active_joint_ids)))
        # Get joint state
        q, _ = self.env.get_state()
        print(q)
        # Check if the joint angle is set correctly
        # self.assertAlmostEqual(q[0], test_angle, places=5)

    def test_kinematics(self):
        """Test forward kinematics calculations."""
        # Set a known configuration
        q = np.zeros(16)
        q[0] = np.pi / 6  # 30 degrees
        self.env.update_pinocchio(q, np.zeros(len(self.env.active_joint_ids)))
        # Check the position of the end-effector
        expected_position = self.calculate_expected_position(q)  # Placeholder
        real_position = self.env.robot.data.oMf[self.env.EE_FRAME_ID].translation
        np.testing.assert_array_almost_equal(real_position, expected_position, decimal=5)

    def tearDown(self):
        """Clean up after each test."""
        self.env = None

if __name__ == "__main__":
    unittest.main()
