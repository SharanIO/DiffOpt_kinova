import pybullet as p
import pybullet_data
import time
import math

# Start the physics client in GUI mode
physicsClient = p.connect(p.GUI)

# Optional: Set the path to look for additional resources
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load a simple plane and a URDF model (e.g., a robot)
planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("kinova_main.urdf", basePosition=[0, 0, 0])

# Set the gravity vector
p.setGravity(0, 0, -9.81)

# Get the number of joints and their names
numJoints = p.getNumJoints(robotId)
jointNames = [p.getJointInfo(robotId, i)[1].decode('utf-8') for i in range(numJoints)]
print(jointNames)
revoluteJoints = [i for i in range(numJoints) if p.getJointInfo(robotId, i)[2] == p.JOINT_REVOLUTE]
print(p.getJointInfo(robotId, 9))
# Print joint information
for j in revoluteJoints:
    print("Revolute joint index:", j, "Name:", jointNames[j])

# Angle for the revolute joints
angle = 0

# Simulation loop that runs until the GUI window is closed
while True:
    # Update the angle
    angle += 0.05  # Increment to move the joints
    if angle > math.pi:
        angle = -math.pi  # Reset the angle for continuous motion
    
    # Apply the control to revolute joints
    for j in revoluteJoints:
        p.setJointMotorControl2(bodyUniqueId=robotId, 
                                jointIndex=j, 
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=angle)
    
    # Step the simulation
    p.stepSimulation()
    
    # Delay to match the real-time simulation speed
    time.sleep(1./240.)

# Disconnect the physics client (not reached if the window is closed manually)
p.disconnect()
