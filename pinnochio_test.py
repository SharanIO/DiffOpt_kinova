import pinocchio as pin
import os
import numpy as np
def load_robot_model(urdf_path):
    """
    Load the robot model from a URDF file and print information about the joints.
    
    Parameters:
        urdf_path (str): The path to the URDF file.
    """

    # Load the robot model
    model = pin.buildModelFromUrdf(urdf_path)
    print("Robot model loaded successfully.")
    print(f"Total number of joints (including fixed): {model.njoints}")
    # q = pin.randomConfiguration(model)
    for frame_index in range(model.nframes):
        frame = model.frames[frame_index]
        # Safely access the frame type name with a default value in case it's not found
        frame_type_name = pin.FrameType.names.get(frame.type, 'Unknown Type')
        print(f"Frame {frame_index}: {frame.name}, Type: {frame_type_name}")
    q = np.array([0., 0., 0.5, 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    print("q: %s" % q)
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    print(data.oMi[2])
    for name, oMi in zip(model.names, data.oMi):
        print(("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat)))
    # Print information about each joint
    for i in range(model.njoints):
            joint = model.joints[i]
            joint_type = joint.shortname()
            joint_idx = joint.idx_q  # Access as property, not a method
            joint_dof = joint.nq
            joint_name = model.names[i]
            print(f"Joint {i}: Name={joint_name}, Type={joint_type}, Index={joint_idx}, DoF={joint_dof}")


# Define the path to your URDF file
script_dir = os.path.dirname(os.path.abspath(__file__))
robot_path = os.path.join(script_dir, "kinova_main.urdf") # Modify this path to where your URDF file is located

# Load the robot model and print joint information
load_robot_model(robot_path)
