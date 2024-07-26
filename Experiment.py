import pickle
import time
from sys import platform
import matplotlib.pyplot as plt
import numpy as np


def main():
    # create environment
    env = ThreeBlocksEnv()

    # define solver
    try:
        controller = ThreeBlocksController()
    except:
        controller = ThreeBlocksController()

    # reset environment
    info = env.reset(
        cameraDistance=2.0, cameraYaw=-1e-3, cameraPitch=-1e-3, lookat=[0.45, 0.0, 0.55]
    )
    print("Environment reset")
    # initialize clock
    t = 0.0

    # create list to store data
    history = []
    torques = []

    # how many iterations per update velocity control command
    control_interval = 50

    computation_times = []

    for i in range(30000):
        t += 1/1000
        
        # get data from info
        q = info["q"]
        dq = info["dq"]
        G = info["G"][:, np.newaxis]
        
        tic = time.time()
        if i % control_interval == 0:
            dq_target, _info = controller.controller(t, q, dq)
            print(f"Step {i}")
            # store data for plotting
            _info["dq_target"] = dq_target
            history.append(_info)
        
        # compute torque command
        τ = (
            6.0 * (dq_target[:, np.newaxis] - dq[:, np.newaxis])
            + G
            - 0.1 * dq[:, np.newaxis]
        )
        torques.append((t, τ))

        if i >= 1:
            computation_times.append(time.time() - tic)
        print(τ)
        # send joint commands to motor
        info = env.step(τ)
    
    # Extract times and torques from the stored list
    times = [time for time, _ in torques]
    torque1 = [τ[0, 0] for _, τ in torques]  # Torque for 1st joint
    torque2 = [τ[1, 0] for _, τ in torques]  # Torque for 2nd joint
    torque3 = [τ[2, 0] for _, τ in torques]
    torque4 = [τ[3, 0] for _, τ in torques]
    torque5 = [τ[4, 0] for _, τ in torques]
    torque6 = [τ[5, 0] for _, τ in torques]
    torque7 = [τ[6, 0] for _, τ in torques]
    plt.figure(1)
    plt.plot(times, torque1, label='Joint 1')
    plt.plot(times, torque2, label='Joint 2')
    plt.plot(times, torque3, label='Joint 3')
    plt.plot(times, torque4, label='Joint 4')
    plt.plot(times, torque5, label='Joint 5')
    plt.plot(times, torque6, label='Joint 6')
    plt.plot(times, torque7, label='Joint 7')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (Nm)')
    plt.title('Torque Over Time')
    plt.legend()
    plt.show()

    # Extract dq_target for each joint over time
    dq_targets = [info['dq_target'] for info in history if 'dq_target' in info]
    times = np.linspace(0, len(dq_targets)/10000, len(dq_targets))

    # Assuming dq_target has 6 elements for 6 joints
    plt.figure(2)
    for i in range(7):
        plt.plot(times, [dq[i] for dq in dq_targets], label=f'Joint {i+1} Target Velocity')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Target Velocity (rad/s)')
    plt.title('Target Joint Velocities Over Time')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    if platform == "darwin":
        from julia.api import Julia

        jl = Julia(compiled_modules=False)

    from three_blocks_controller import (
        ThreeBlocksController,
    )
    from kinova_three_block_env import ThreeBlocksEnv

    main()