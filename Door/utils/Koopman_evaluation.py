"""
Define the functions that are used to test the learned koopman matrix
"""
from glob import escape
from attr import asdict
import numpy as np
import time
from tqdm import tqdm
from utils.gym_env import GymEnv
from utils.quatmath import euler2quat
from utils.coord_trans import ori_transform, ori_transform_inverse
import torch

def koopman_evaluation(koopman_object, koopman_matrix, Test_data, Velocity, num_hand, num_obj):
    '''
    Input: Koopman object (Drafted, MLP, GNN) for observable lifting
           Learned koopman matrix
           Testing data
           Velocity flag
    '''
    ErrorInLifted = np.zeros(koopman_object.compute_observable(num_hand, num_obj))
    ErrorInOriginal = np.zeros(num_hand + num_obj) 
    for k in tqdm(range(len(Test_data))):
        if Velocity:
            hand_OriState = np.append(Test_data[k][0]['handpos'], Test_data[k][0]['handvel'])
        else:
            hand_OriState = Test_data[k][0]['handpos']
        obj_OriState = np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objvel'], Test_data[k][0]['handle_init']))
        z_t = koopman_object.z(hand_OriState, obj_OriState)  # initial states in lifted space
        for t in range(len(Test_data[k]) - 1):
            if Velocity:
                hand_OriState = np.append(Test_data[k][t + 1]['handpos'], Test_data[k][t + 1]['handvel'])
            else:
                hand_OriState = Test_data[k][t + 1]['handpos']
            obj_OriState = np.append(Test_data[k][t + 1]['objpos'], np.append(Test_data[k][t + 1]['objvel'], Test_data[k][t + 1]['handle_init']))
            z_t_1 = koopman_object.z(hand_OriState, obj_OriState) # states in lifted space at next time step (extracted from data)
            z_t_1_computed = np.dot(koopman_matrix, z_t)
            x_t_1 = np.append(z_t_1[:num_hand], z_t_1[2 * num_hand: 2 * num_hand + num_obj])  # observation functions: hand_state, hand_state^2, object_state, object_state^2
            x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])  
            ErrorInLifted += np.abs(z_t_1 - z_t_1_computed)  # if using np.square, we will get weird results.
            ErrorInOriginal += np.abs(x_t_1 - x_t_1_computed)
            z_t = z_t_1
    M = len(Test_data) * (len(Test_data[0]) - 1)
    ErrorInLifted /= M
    ErrorInOriginal /= M
    return ErrorInLifted, ErrorInOriginal

def koopman_policy_control(env_name, controller, koopman_object, koopman_matrix, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize):
    print("Begin to compute the simulation errors!")
    hand_pos_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])  # palm_pos - handle_pos
    demo_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error_koopman = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    Computed_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    Demo_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    # for this task, we are only able to report the success rate for the simulation results
    # also, for this task, no failure cases.
    success_list_sim = []
    success_rate = str()
    for k in tqdm(range(len(Test_data))):
        num_handpos = len(Test_data[k][0]['handpos'])
        if Velocity:
            hand_OriState = np.append(Test_data[k][0]['handpos'], Test_data[k][0]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k][0]['handpos']
        obj_OriState = np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objvel'], Test_data[k][0]['handle_init']))
        init_state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        init_state_dict['door_body_pos'] = Test_data[k][0]['init']['door_body_pos']  # fixed for each piece of demo data
        e.set_env_state(init_state_dict)
        if koopmanoption == 'Drafted': 
            z_t = koopman_object.z(hand_OriState, obj_OriState)  # initial states in lifted space
        else:
            input_state = np.append(hand_OriState, obj_OriState)
            input_state = torch.from_numpy(input_state)
            z_t = koopman_object.z(input_state).numpy()  # lift the initial states and then roll it out using Koopman operator
        for t in range(len(Test_data[k]) - 1):  # this loop is for system evolution, open loop control, no feedback
            z_t_1_computed = np.dot(koopman_matrix, z_t)
            if koopmanoption == 'Drafted':  
                x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])
                if not Velocity:
                    obj_pos_world = x_t_1_computed[28:31]  # handle pos
                else:
                    obj_pos_world = x_t_1_computed[56:59]
            else:
                x_t_1_computed = koopman_object.z_inverse(torch.from_numpy(z_t_1_computed)).numpy()
                if not Velocity:
                    obj_pos_world = x_t_1_computed[28:31]
                else:
                    obj_pos_world = x_t_1_computed[56:59]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint state
            # hand_pos_desired = hand_pos + Test_data[k][0]['final_handpos']
            hand_pos_desired = hand_pos  # control frequency
            z_t = z_t_1_computed
            current = e.get_env_state()['qpos'][:28] # current state
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()
            Computed_torque_values[:, t, k] = NN_output
            Demo_torque_values[:, t, k] = Test_data[k][t]['action']
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation)
            if Visualize:
                e.env.mj_render()
            # ground-truth state values (obtained from RL in simulator)
            # gt_hand_pos = Test_data[k][t + 1]['handpos'] + Test_data[k][0]['final_handpos']
            gt_hand_pos = Test_data[k][t + 1]['handpos']
            # calculated state values using Koopman rollouts (in simulator)
            hand_pos = e.env.get_full_obs_visualization()[:num_handpos]     
            obj_obs = e.get_obs()
            # compute the errors
            hand_pos_error_handJoint[t, k] = np.mean(np.abs(gt_hand_pos[4:] - hand_pos[4:]))  # probably not return the mean value?
            hand_pos_PID_error_handJoint[t, k] = np.mean(np.abs(hand_pos_desired[4:] - hand_pos[4:]))  
            hand_pos_error_base[t, k] = np.mean(np.abs(gt_hand_pos[:4] - hand_pos[:4]))  # probably not return the mean value?
            hand_pos_PID_error_base[t, k] = np.mean(np.abs(hand_pos_desired[:4] - hand_pos[:4]))  
            # instead of defining the position error of the nail, we define it as the difference between the hammer tool and the nail
            obj_pos_error[t, k] = np.mean(np.abs(obj_obs[35:38]))  # palm_pos - handle_pos
            obj_pos_error_koopman[t, k] = np.mean(np.abs(obj_obs[29:32] - obj_pos_world))  # obj_obs[29:32] -> palm position
            demo_pos_error[t, k] = np.mean(np.abs(Test_data[k][t + 1]['observation'])) # palm-handle
            current_hinge_pos = obj_obs[28:29] # door opening angle
        if current_hinge_pos > 1.35:
            success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))
    print("Success rate (RL) = %f" % (1))
    if error_type == 'demo':
        return [hand_pos_error_handJoint, hand_pos_PID_error_handJoint, hand_pos_error_base, hand_pos_PID_error_base, Computed_torque_values, Demo_torque_values, success_rate]
    else:
        return [obj_pos_error, obj_pos_error_koopman, demo_pos_error, success_rate]     

def koopman_policy_control_unseenTest(env_name, controller, koopman_object, koopman_matrix, Test_data, Velocity, num_hand, num_obj, koopmanoption, Visualize):
    print("Testing the learned koopman dynamcis!")
    print("Begin to compute the simulation errors!")
    e = GymEnv(env_name)
    horizon = 70 # selected time steps that is enough to finish the task with good performance
    obj_pos_error = np.zeros([horizon - 1, len(Test_data)])
    obj_pos_error_koopman = np.zeros([horizon - 1, len(Test_data)])
    e.reset()
    init_state_dict = dict()
    # e.set_env_state(path['init_state_dict'])
    success_list_sim = []
    success_rate = str()
    for k in tqdm(range(len(Test_data))):
        num_handpos = len(Test_data[k]['handpos'])
        if Velocity:
            hand_OriState = np.append(Test_data[k]['handpos'], Test_data[k]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k]['handpos']
        obj_OriState = np.append(Test_data[k]['objpos'], np.append(Test_data[k]['objvel'], Test_data[k]['handle_init']))
        init_state_dict['qpos'] = Test_data[k]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k]['init']['qvel']
        init_state_dict['door_body_pos'] = Test_data[k]['init']['door_body_pos']  # fixed for each piece of demo data
        e.set_env_state(init_state_dict)
        if koopmanoption == 'Drafted': 
            z_t = koopman_object.z(hand_OriState, obj_OriState)  # initial states in lifted space
        else:
            input_state = np.append(hand_OriState, obj_OriState)
            input_state = torch.from_numpy(input_state)
            z_t = koopman_object.z(input_state).numpy()  # lift the initial states and then roll it out using Koopman operator
        for t in range(horizon - 1):  # this loop is for system evolution, open loop control, no feedback
            z_t_1_computed = np.dot(koopman_matrix, z_t)
            if koopmanoption == 'Drafted':  
                x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])
                if not Velocity:
                    obj_pos_world = x_t_1_computed[28:31]  # handle pos
                else:
                    obj_pos_world = x_t_1_computed[56:59]
            else:
                x_t_1_computed = koopman_object.z_inverse(torch.from_numpy(z_t_1_computed)).numpy()
                if not Velocity:
                    obj_pos_world = x_t_1_computed[28:31]
                else:
                    obj_pos_world = x_t_1_computed[56:59]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint states
            hand_pos_desired = hand_pos
            z_t = z_t_1_computed
            current = e.get_env_state()['qpos'][:28] # current state
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation)
            if Visualize:
                e.env.mj_render()
            # calculated state values using Koopman rollouts (in simulator)      
            obj_obs = e.get_obs()
            # compute the errors
            obj_pos_error[t, k] = np.mean(np.abs(obj_obs[35:38]))  # palm_pos - handle_pos
            obj_pos_error_koopman[t, k] = np.mean(np.abs(obj_obs[29:32] - obj_pos_world))  # obj_obs[29:32] -> palm position
            current_hinge_pos = obj_obs[28:29] # door opening angle
        if current_hinge_pos > 1.35:
            success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))
    return obj_pos_error, obj_pos_error_koopman, success_rate
    
def koopman_error_visualization(env_name, koopman_object, koopman_matrix, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize):
    print("Begin to compute the koopman rollout errors!")
    hand_pos_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    Computed_joint_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    Demo_joint_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    e = GymEnv(env_name)
    e.reset()
    state_dict = {}
    for k in tqdm(range(len(Test_data))):
        num_handpos = len(Test_data[k][0]['handpos'])
        num_handvel = len(Test_data[k][0]['handvel'])
        if Velocity:
            hand_OriState = np.append(Test_data[k][0]['handpos'], Test_data[k][0]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k][0]['handpos']
        Demo_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        Computed_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        obj_OriState = np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objvel'], Test_data[k][0]['handle_init']))
        state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        state_dict['door_body_pos'] = Test_data[k][0]['init']['door_body_pos']  # fixed for each piece of demo data
        if koopmanoption == 'Drafted': 
            z_t = koopman_object.z(hand_OriState, obj_OriState)  # initial states in lifted space
        else:
            input_state = np.append(hand_OriState, obj_OriState)
            input_state = torch.from_numpy(input_state)
            z_t = koopman_object.z(input_state).numpy()  # lift the initial states and then roll it out using Koopman operator
        for t in range(len(Test_data[k]) - 1):
            e.KoopmanVisualize(seed = None, state_dict = state_dict) # visualize the demo data without any actions (more like a visualization)
            z_t_1_computed = np.dot(koopman_matrix, z_t)
            if koopmanoption == 'Drafted':  
                x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])
            else:
                x_t_1_computed = koopman_object.z_inverse(torch.from_numpy(z_t_1_computed)).numpy()
            # ground-truth state values
            gt_hand_pos = Test_data[k][t + 1]['handpos']
            # calculated state values using Koopman rollouts vb
            hand_pos = x_t_1_computed[:num_handpos]
            z_t = z_t_1_computed
            # print(x_t_1_computed[-3:]) # almost keep constant as the Test_data[k][0]['handle_init']
            if Velocity:
                hand_vel = x_t_1_computed[num_handpos:num_handpos + num_handvel]
            else:
                hand_vel = np.zeros(num_handvel)
            # compute the errors
            hand_pos_error_handJoint[t, k] = np.mean(np.abs(gt_hand_pos[4:] - hand_pos[4:]))  # probably not return the mean value?
            hand_pos_error_base[t, k] = np.mean(np.abs(gt_hand_pos[:4] - hand_pos[:4]))  # probably not return the mean value?
            Computed_joint_values[:, t + 1, k] = hand_pos
            Demo_joint_values[:, t + 1, k] = gt_hand_pos
            state_dict['qpos'] = np.append(hand_pos, np.zeros([2]))
            state_dict['qvel'] = np.zeros([30])
            if Visualize:  # Since the PID controller works well, we do not need to visualize the trajectories any more
                e.env.mj_render()
    if error_type == 'demo':
        return [hand_pos_error_handJoint, hand_pos_error_base, Computed_joint_values, Demo_joint_values]

def BC_error_visualization(env_name, BC_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize):
    print("Begin to compute the koopman rollout errors!")
    hand_pos_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    Computed_joint_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    Demo_joint_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    e = GymEnv(env_name)
    e.reset()
    state_dict = {}
    for k in tqdm(range(len(Test_data))):
        num_handpos = len(Test_data[k][0]['handpos'])
        num_handvel = len(Test_data[k][0]['handvel'])
        if Velocity:
            hand_OriState = np.append(Test_data[k][0]['handpos'], Test_data[k][0]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k][0]['handpos']
        Demo_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        Computed_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        obj_OriState = np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objvel'], Test_data[k][0]['handle_init']))
        state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        state_dict['door_body_pos'] = Test_data[k][0]['init']['door_body_pos']  # fixed for each piece of demo data
        if koopmanoption == 'Drafted': 
            tmp_input = np.append(hand_OriState, obj_OriState)
            z_t = torch.from_numpy(tmp_input)
        for t in range(len(Test_data[k]) - 1):
            # e.KoopmanVisualize(seed = None, state_dict = state_dict) # visualize the demo data without any actions (more like a visualization)
            z_t_1_computed = BC_agent(z_t)
            if koopmanoption == 'Drafted':  
                x_t_1_computed = z_t_1_computed.detach().numpy()
            # ground-truth state values
            gt_hand_pos = Test_data[k][t + 1]['handpos']
            # calculated state values using Koopman rollouts vb
            hand_pos = x_t_1_computed[:num_handpos]
            z_t = z_t_1_computed
            # print(x_t_1_computed[-3:]) # almost keep constant as the Test_data[k][0]['handle_init']
            if Velocity:
                hand_vel = x_t_1_computed[num_handpos:num_handpos + num_handvel]
            else:
                hand_vel = np.zeros(num_handvel)
            # compute the errors
            hand_pos_error_handJoint[t, k] = np.mean(np.abs(gt_hand_pos[4:] - hand_pos[4:]))  # probably not return the mean value?
            hand_pos_error_base[t, k] = np.mean(np.abs(gt_hand_pos[:4] - hand_pos[:4]))  # probably not return the mean value?
            Computed_joint_values[:, t + 1, k] = hand_pos
            Demo_joint_values[:, t + 1, k] = gt_hand_pos
            state_dict['qpos'] = np.append(hand_pos, np.zeros([2]))
            state_dict['qvel'] = np.zeros([30])
            if Visualize:  # Since the PID controller works well, we do not need to visualize the trajectories any more
                e.env.mj_render()
    if error_type == 'demo':
        return [hand_pos_error_handJoint, hand_pos_error_base, Computed_joint_values, Demo_joint_values]

def BC_policy_control(env_name, controller, BC_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize):
    print("Begin to compute the simulation errors!")
    hand_pos_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])  # palm_pos - handle_pos
    demo_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error_koopman = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    Computed_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    Demo_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    # for this task, we are only able to report the success rate for the simulation results
    # also, for this task, no failure cases.
    success_list_sim = []
    success_rate = str()
    for k in tqdm(range(len(Test_data))):
        num_handpos = len(Test_data[k][0]['handpos'])
        if Velocity:
            hand_OriState = np.append(Test_data[k][0]['handpos'], Test_data[k][0]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k][0]['handpos']
        obj_OriState = np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objvel'], Test_data[k][0]['handle_init']))
        init_state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        init_state_dict['door_body_pos'] = Test_data[k][0]['init']['door_body_pos']  # fixed for each piece of demo data
        e.set_env_state(init_state_dict)
        if koopmanoption == 'Drafted': 
            tmp_input = np.append(hand_OriState, obj_OriState)
            z_t = torch.from_numpy(tmp_input)
        for t in range(len(Test_data[k]) - 1):  # this loop is for system evolution, open loop control, no feedback
            z_t_1_computed = BC_agent(z_t)
            if koopmanoption == 'Drafted':  
                x_t_1_computed = z_t_1_computed.detach().numpy()
                if not Velocity:
                    obj_pos_world = x_t_1_computed[28:31]  # handle pos
                else:
                    obj_pos_world = x_t_1_computed[56:59]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint state
            # hand_pos_desired = hand_pos + Test_data[k][0]['final_handpos']
            hand_pos_desired = hand_pos  # control frequency
            z_t = z_t_1_computed
            current = e.get_env_state()['qpos'][:28] # current state
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()
            Computed_torque_values[:, t, k] = NN_output
            Demo_torque_values[:, t, k] = Test_data[k][t]['action']
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation)
            if Visualize:
                e.env.mj_render()
            # ground-truth state values (obtained from RL in simulator)
            # gt_hand_pos = Test_data[k][t + 1]['handpos'] + Test_data[k][0]['final_handpos']
            gt_hand_pos = Test_data[k][t + 1]['handpos']
            # calculated state values using Koopman rollouts (in simulator)
            hand_pos = e.env.get_full_obs_visualization()[:num_handpos]     
            obj_obs = e.get_obs()
            # compute the errors
            hand_pos_error_handJoint[t, k] = np.mean(np.abs(gt_hand_pos[4:] - hand_pos[4:]))  # probably not return the mean value?
            hand_pos_PID_error_handJoint[t, k] = np.mean(np.abs(hand_pos_desired[4:] - hand_pos[4:]))  
            hand_pos_error_base[t, k] = np.mean(np.abs(gt_hand_pos[:4] - hand_pos[:4]))  # probably not return the mean value?
            hand_pos_PID_error_base[t, k] = np.mean(np.abs(hand_pos_desired[:4] - hand_pos[:4]))  
            # instead of defining the position error of the nail, we define it as the difference between the hammer tool and the nail
            obj_pos_error[t, k] = np.mean(np.abs(obj_obs[35:38]))  # palm_pos - handle_pos
            obj_pos_error_koopman[t, k] = np.mean(np.abs(obj_obs[29:32] - obj_pos_world))  # obj_obs[29:32] -> palm position
            demo_pos_error[t, k] = np.mean(np.abs(Test_data[k][t + 1]['observation'])) # palm-handle
            current_hinge_pos = obj_obs[28:29] # door opening angle
        if current_hinge_pos > 1.35:
            success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))
    print("Success rate (RL) = %f" % (1))
    if error_type == 'demo':
        return [hand_pos_error_handJoint, hand_pos_PID_error_handJoint, hand_pos_error_base, hand_pos_PID_error_base, Computed_torque_values, Demo_torque_values, success_rate]
    else:
        return [obj_pos_error, obj_pos_error_koopman, demo_pos_error, success_rate]  

def BC_policy_control_unseenTest(env_name, controller, BC_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, Visualize):
    print("Testing the learned koopman dynamcis!")
    print("Begin to compute the simulation errors!")
    e = GymEnv(env_name)
    horizon = 70 # selected time steps that is enough to finish the task with good performance
    obj_pos_error = np.zeros([horizon - 1, len(Test_data)])
    obj_pos_error_koopman = np.zeros([horizon - 1, len(Test_data)])
    e.reset()
    init_state_dict = dict()
    # e.set_env_state(path['init_state_dict'])
    success_list_sim = []
    success_rate = str()
    for k in tqdm(range(len(Test_data))):
        num_handpos = len(Test_data[k]['handpos'])
        if Velocity:
            hand_OriState = np.append(Test_data[k]['handpos'], Test_data[k]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k]['handpos']
        obj_OriState = np.append(Test_data[k]['objpos'], np.append(Test_data[k]['objvel'], Test_data[k]['handle_init']))
        init_state_dict['qpos'] = Test_data[k]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k]['init']['qvel']
        init_state_dict['door_body_pos'] = Test_data[k]['init']['door_body_pos']  # fixed for each piece of demo data
        e.set_env_state(init_state_dict)
        if koopmanoption == 'Drafted': 
            tmp_input = np.append(hand_OriState, obj_OriState)
            z_t = torch.from_numpy(tmp_input)
        for t in range(horizon - 1):  # this loop is for system evolution, open loop control, no feedback
            z_t_1_computed = BC_agent(z_t)
            if koopmanoption == 'Drafted':  
                x_t_1_computed = z_t_1_computed.detach().numpy()
                if not Velocity:
                    obj_pos_world = x_t_1_computed[28:31]  # handle pos
                else:
                    obj_pos_world = x_t_1_computed[56:59]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint states
            hand_pos_desired = hand_pos
            z_t = z_t_1_computed
            current = e.get_env_state()['qpos'][:28] # current state
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation)
            if Visualize:
                e.env.mj_render()
            # calculated state values using Koopman rollouts (in simulator)      
            obj_obs = e.get_obs()
            # compute the errors
            obj_pos_error[t, k] = np.mean(np.abs(obj_obs[35:38]))  # palm_pos - handle_pos
            obj_pos_error_koopman[t, k] = np.mean(np.abs(obj_obs[29:32] - obj_pos_world))  # obj_obs[29:32] -> palm position
            current_hinge_pos = obj_obs[28:29] # door opening angle
        if current_hinge_pos > 1.35:
            success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))
    return obj_pos_error, obj_pos_error_koopman, success_rate

def LSTM_error_visualization(env_name, LSTM_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize):
    print("Begin to compute the koopman rollout errors!")
    hand_pos_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    Computed_joint_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    Demo_joint_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    e = GymEnv(env_name)
    e.reset()
    state_dict = {}
    for k in tqdm(range(len(Test_data))):
        num_handpos = len(Test_data[k][0]['handpos'])
        num_handvel = len(Test_data[k][0]['handvel'])
        if Velocity:
            hand_OriState = np.append(Test_data[k][0]['handpos'], Test_data[k][0]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k][0]['handpos']
        Demo_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        Computed_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        obj_OriState = np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objvel'], Test_data[k][0]['handle_init']))
        state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        state_dict['door_body_pos'] = Test_data[k][0]['init']['door_body_pos']  # fixed for each piece of demo data
        if koopmanoption == 'Drafted': 
            tmp_input = np.append(hand_OriState, obj_OriState)
            z_t = torch.from_numpy(tmp_input)
        hidden_state = (np.zeros((1, 1, LSTM_agent.model.rnn_cell.hidden_size)), np.zeros((1, 1, LSTM_agent.model.rnn_cell.hidden_size)))
        for t in range(len(Test_data[k]) - 1):
            e.KoopmanVisualize(seed = None, state_dict = state_dict) # visualize the demo data without any actions (more like a visualization)
            z_t_1_computed, hidden_state = LSTM_agent.get_action(z_t, hidden_state)
            if koopmanoption == 'Drafted':  
                x_t_1_computed = z_t_1_computed
            # ground-truth state values
            gt_hand_pos = Test_data[k][t + 1]['handpos']
            # calculated state values using Koopman rollouts vb
            hand_pos = x_t_1_computed[:num_handpos]
            z_t = z_t_1_computed
            # print(x_t_1_computed[-3:]) # almost keep constant as the Test_data[k][0]['handle_init']
            if Velocity:
                hand_vel = x_t_1_computed[num_handpos:num_handpos + num_handvel]
            else:
                hand_vel = np.zeros(num_handvel)
            # compute the errors
            hand_pos_error_handJoint[t, k] = np.mean(np.abs(gt_hand_pos[4:] - hand_pos[4:]))  # probably not return the mean value?
            hand_pos_error_base[t, k] = np.mean(np.abs(gt_hand_pos[:4] - hand_pos[:4]))  # probably not return the mean value?
            Computed_joint_values[:, t + 1, k] = hand_pos
            Demo_joint_values[:, t + 1, k] = gt_hand_pos
            state_dict['qpos'] = np.append(hand_pos, np.zeros([2]))
            state_dict['qvel'] = np.zeros([30])
            if Visualize:  # Since the PID controller works well, we do not need to visualize the trajectories any more
                e.env.mj_render()
    if error_type == 'demo':
        return [hand_pos_error_handJoint, hand_pos_error_base, Computed_joint_values, Demo_joint_values]

def LSTM_policy_control(env_name, controller, LSTM_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize):
    print("Begin to compute the simulation errors!")
    hand_pos_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])  # palm_pos - handle_pos
    demo_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error_koopman = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    Computed_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    Demo_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    # for this task, we are only able to report the success rate for the simulation results
    # also, for this task, no failure cases.
    success_list_sim = []
    success_rate = str()
    for k in tqdm(range(len(Test_data))):
        num_handpos = len(Test_data[k][0]['handpos'])
        if Velocity:
            hand_OriState = np.append(Test_data[k][0]['handpos'], Test_data[k][0]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k][0]['handpos']
        obj_OriState = np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objvel'], Test_data[k][0]['handle_init']))
        init_state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        init_state_dict['door_body_pos'] = Test_data[k][0]['init']['door_body_pos']  # fixed for each piece of demo data
        e.set_env_state(init_state_dict)
        if koopmanoption == 'Drafted': 
            tmp_input = np.append(hand_OriState, obj_OriState)
            z_t = torch.from_numpy(tmp_input)
        hidden_state = (np.zeros((1, 1, LSTM_agent.model.rnn_cell.hidden_size)), np.zeros((1, 1, LSTM_agent.model.rnn_cell.hidden_size)))
        for t in range(len(Test_data[k]) - 1):  # this loop is for system evolution, open loop control, no feedback
            z_t_1_computed, hidden_state = LSTM_agent.get_action(z_t, hidden_state)
            if koopmanoption == 'Drafted':  
                x_t_1_computed = z_t_1_computed
                if not Velocity:
                    obj_pos_world = x_t_1_computed[28:31]  # handle pos
                else:
                    obj_pos_world = x_t_1_computed[56:59]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint state
            # hand_pos_desired = hand_pos + Test_data[k][0]['final_handpos']
            hand_pos_desired = hand_pos  # control frequency
            z_t = z_t_1_computed
            current = e.get_env_state()['qpos'][:28] # current state
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()
            Computed_torque_values[:, t, k] = NN_output
            Demo_torque_values[:, t, k] = Test_data[k][t]['action']
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation)
            if Visualize:
                e.env.mj_render()
            # ground-truth state values (obtained from RL in simulator)
            # gt_hand_pos = Test_data[k][t + 1]['handpos'] + Test_data[k][0]['final_handpos']
            gt_hand_pos = Test_data[k][t + 1]['handpos']
            # calculated state values using Koopman rollouts (in simulator)
            hand_pos = e.env.get_full_obs_visualization()[:num_handpos]     
            obj_obs = e.get_obs()
            # compute the errors
            hand_pos_error_handJoint[t, k] = np.mean(np.abs(gt_hand_pos[4:] - hand_pos[4:]))  # probably not return the mean value?
            hand_pos_PID_error_handJoint[t, k] = np.mean(np.abs(hand_pos_desired[4:] - hand_pos[4:]))  
            hand_pos_error_base[t, k] = np.mean(np.abs(gt_hand_pos[:4] - hand_pos[:4]))  # probably not return the mean value?
            hand_pos_PID_error_base[t, k] = np.mean(np.abs(hand_pos_desired[:4] - hand_pos[:4]))  
            # instead of defining the position error of the nail, we define it as the difference between the hammer tool and the nail
            obj_pos_error[t, k] = np.mean(np.abs(obj_obs[35:38]))  # palm_pos - handle_pos
            obj_pos_error_koopman[t, k] = np.mean(np.abs(obj_obs[29:32] - obj_pos_world))  # obj_obs[29:32] -> palm position
            demo_pos_error[t, k] = np.mean(np.abs(Test_data[k][t + 1]['observation'])) # palm-handle
            current_hinge_pos = obj_obs[28:29] # door opening angle
        if current_hinge_pos > 1.35:
            success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))
    print("Success rate (RL) = %f" % (1))
    if error_type == 'demo':
        return [hand_pos_error_handJoint, hand_pos_PID_error_handJoint, hand_pos_error_base, hand_pos_PID_error_base, Computed_torque_values, Demo_torque_values, success_rate]
    else:
        return [obj_pos_error, obj_pos_error_koopman, demo_pos_error, success_rate]  

def LSTM_policy_control_unseenTest(env_name, controller, LSTM_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, Visualize):
    print("Testing the learned koopman dynamcis!")
    print("Begin to compute the simulation errors!")
    e = GymEnv(env_name)
    horizon = 70 # selected time steps that is enough to finish the task with good performance
    obj_pos_error = np.zeros([horizon - 1, len(Test_data)])
    obj_pos_error_koopman = np.zeros([horizon - 1, len(Test_data)])
    e.reset()
    init_state_dict = dict()
    # e.set_env_state(path['init_state_dict'])
    success_list_sim = []
    success_rate = str()
    for k in tqdm(range(len(Test_data))):
        num_handpos = len(Test_data[k]['handpos'])
        if Velocity:
            hand_OriState = np.append(Test_data[k]['handpos'], Test_data[k]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k]['handpos']
        obj_OriState = np.append(Test_data[k]['objpos'], np.append(Test_data[k]['objvel'], Test_data[k]['handle_init']))
        init_state_dict['qpos'] = Test_data[k]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k]['init']['qvel']
        init_state_dict['door_body_pos'] = Test_data[k]['init']['door_body_pos']  # fixed for each piece of demo data
        e.set_env_state(init_state_dict)
        if koopmanoption == 'Drafted': 
            tmp_input = np.append(hand_OriState, obj_OriState)
            z_t = torch.from_numpy(tmp_input)
        hidden_state = (np.zeros((1, 1, LSTM_agent.model.rnn_cell.hidden_size)), np.zeros((1, 1, LSTM_agent.model.rnn_cell.hidden_size)))
        for t in range(horizon - 1):  # this loop is for system evolution, open loop control, no feedback
            z_t_1_computed, hidden_state = LSTM_agent.get_action(z_t, hidden_state)
            if koopmanoption == 'Drafted':  
                x_t_1_computed = z_t_1_computed
                if not Velocity:
                    obj_pos_world = x_t_1_computed[28:31]  # handle pos
                else:
                    obj_pos_world = x_t_1_computed[56:59]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint states
            hand_pos_desired = hand_pos
            z_t = z_t_1_computed
            current = e.get_env_state()['qpos'][:28] # current state
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation)
            if Visualize:
                e.env.mj_render()
            # calculated state values using Koopman rollouts (in simulator)      
            obj_obs = e.get_obs()
            # compute the errors
            obj_pos_error[t, k] = np.mean(np.abs(obj_obs[35:38]))  # palm_pos - handle_pos
            obj_pos_error_koopman[t, k] = np.mean(np.abs(obj_obs[29:32] - obj_pos_world))  # obj_obs[29:32] -> palm position
            current_hinge_pos = obj_obs[28:29] # door opening angle
        if current_hinge_pos > 1.35:
            success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))
    return obj_pos_error, obj_pos_error_koopman, success_rate

def NDP_error_visualization(env_name, NDP_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize):
    print("Begin to compute the koopman rollout errors!")
    hand_pos_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    Computed_joint_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    Demo_joint_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    e = GymEnv(env_name)
    e.reset()
    state_dict = {}
    for k in tqdm(range(len(Test_data))):
        num_handpos = len(Test_data[k][0]['handpos'])
        num_handvel = len(Test_data[k][0]['handvel'])
        if Velocity:
            hand_OriState = np.append(Test_data[k][0]['handpos'], Test_data[k][0]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k][0]['handpos']
        Demo_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        Computed_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        obj_OriState = np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objvel'], Test_data[k][0]['handle_init']))
        state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        state_dict['door_body_pos'] = Test_data[k][0]['init']['door_body_pos']  # fixed for each piece of demo data
        if koopmanoption == 'Drafted': 
            tmp_input = np.append(hand_OriState, obj_OriState)
            z_t = torch.from_numpy(tmp_input)
        traj = NDP_agent.execute(z_t, z_t) # traj -> [T, dim]
        for t in range(len(Test_data[k]) - 1):
            e.KoopmanVisualize(seed = None, state_dict = state_dict) # visualize the demo data without any actions (more like a visualization)
            if koopmanoption == 'Drafted':  
                x_t_1_computed = traj[t + 1]
            # ground-truth state values
            gt_hand_pos = Test_data[k][t + 1]['handpos']
            # calculated state values using Koopman rollouts vb
            hand_pos = x_t_1_computed[:num_handpos]
            # print(x_t_1_computed[-3:]) # almost keep constant as the Test_data[k][0]['handle_init']
            if Velocity:
                hand_vel = x_t_1_computed[num_handpos:num_handpos + num_handvel]
            else:
                hand_vel = np.zeros(num_handvel)
            # compute the errors
            hand_pos_error_handJoint[t, k] = np.mean(np.abs(gt_hand_pos[4:] - hand_pos[4:]))  # probably not return the mean value?
            hand_pos_error_base[t, k] = np.mean(np.abs(gt_hand_pos[:4] - hand_pos[:4]))  # probably not return the mean value?
            Computed_joint_values[:, t + 1, k] = hand_pos
            Demo_joint_values[:, t + 1, k] = gt_hand_pos
            state_dict['qpos'] = np.append(hand_pos, np.zeros([2]))
            state_dict['qvel'] = np.zeros([30])
            if Visualize:  # Since the PID controller works well, we do not need to visualize the trajectories any more
                e.env.mj_render()
    if error_type == 'demo':
        return [hand_pos_error_handJoint, hand_pos_error_base, Computed_joint_values, Demo_joint_values]

def NDP_policy_control(env_name, controller, NDP_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize):
    print("Begin to compute the simulation errors!")
    hand_pos_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])  # palm_pos - handle_pos
    demo_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error_koopman = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    Computed_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    Demo_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    # for this task, we are only able to report the success rate for the simulation results
    # also, for this task, no failure cases.
    success_list_sim = []
    success_rate = str()
    for k in tqdm(range(len(Test_data))):
        num_handpos = len(Test_data[k][0]['handpos'])
        if Velocity:
            hand_OriState = np.append(Test_data[k][0]['handpos'], Test_data[k][0]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k][0]['handpos']
        obj_OriState = np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objvel'], Test_data[k][0]['handle_init']))
        init_state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        init_state_dict['door_body_pos'] = Test_data[k][0]['init']['door_body_pos']  # fixed for each piece of demo data
        e.set_env_state(init_state_dict)
        if koopmanoption == 'Drafted': 
            tmp_input = np.append(hand_OriState, obj_OriState)
            z_t = torch.from_numpy(tmp_input)
        traj = NDP_agent.execute(z_t, z_t) # traj -> [T, dim]
        for t in range(len(Test_data[k]) - 1):  # this loop is for system evolution, open loop control, no feedback
            if koopmanoption == 'Drafted':  
                x_t_1_computed = traj[t + 1]
                if not Velocity:
                    obj_pos_world = x_t_1_computed[28:31]  # handle pos
                else:
                    obj_pos_world = x_t_1_computed[56:59]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint state
            # hand_pos_desired = hand_pos + Test_data[k][0]['final_handpos']
            hand_pos_desired = hand_pos  # control frequency
            current = e.get_env_state()['qpos'][:28] # current state
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()
            Computed_torque_values[:, t, k] = NN_output
            Demo_torque_values[:, t, k] = Test_data[k][t]['action']
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation)
            if Visualize:
                e.env.mj_render()
            # ground-truth state values (obtained from RL in simulator)
            # gt_hand_pos = Test_data[k][t + 1]['handpos'] + Test_data[k][0]['final_handpos']
            gt_hand_pos = Test_data[k][t + 1]['handpos']
            # calculated state values using Koopman rollouts (in simulator)
            hand_pos = e.env.get_full_obs_visualization()[:num_handpos]     
            obj_obs = e.get_obs()
            # compute the errors
            hand_pos_error_handJoint[t, k] = np.mean(np.abs(gt_hand_pos[4:] - hand_pos[4:]))  # probably not return the mean value?
            hand_pos_PID_error_handJoint[t, k] = np.mean(np.abs(hand_pos_desired[4:] - hand_pos[4:]))  
            hand_pos_error_base[t, k] = np.mean(np.abs(gt_hand_pos[:4] - hand_pos[:4]))  # probably not return the mean value?
            hand_pos_PID_error_base[t, k] = np.mean(np.abs(hand_pos_desired[:4] - hand_pos[:4]))  
            # instead of defining the position error of the nail, we define it as the difference between the hammer tool and the nail
            obj_pos_error[t, k] = np.mean(np.abs(obj_obs[35:38]))  # palm_pos - handle_pos
            obj_pos_error_koopman[t, k] = np.mean(np.abs(obj_obs[29:32] - obj_pos_world))  # obj_obs[29:32] -> palm position
            demo_pos_error[t, k] = np.mean(np.abs(Test_data[k][t + 1]['observation'])) # palm-handle
            current_hinge_pos = obj_obs[28:29] # door opening angle
        if current_hinge_pos > 1.35:
            success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))
    print("Success rate (RL) = %f" % (1))
    if error_type == 'demo':
        return [hand_pos_error_handJoint, hand_pos_PID_error_handJoint, hand_pos_error_base, hand_pos_PID_error_base, Computed_torque_values, Demo_torque_values, success_rate]
    else:
        return [obj_pos_error, obj_pos_error_koopman, demo_pos_error, success_rate]  

def NDP_policy_control_unseenTest(env_name, controller, NDP_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, Visualize):
    print("Testing the learned koopman dynamcis!")
    print("Begin to compute the simulation errors!")
    e = GymEnv(env_name)
    horizon = 70 # selected time steps that is enough to finish the task with good performance
    obj_pos_error = np.zeros([horizon - 1, len(Test_data)])
    obj_pos_error_koopman = np.zeros([horizon - 1, len(Test_data)])
    e.reset()
    init_state_dict = dict()
    # e.set_env_state(path['init_state_dict'])
    success_list_sim = []
    success_rate = str()
    for k in tqdm(range(len(Test_data))):
        num_handpos = len(Test_data[k]['handpos'])
        if Velocity:
            hand_OriState = np.append(Test_data[k]['handpos'], Test_data[k]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k]['handpos']
        obj_OriState = np.append(Test_data[k]['objpos'], np.append(Test_data[k]['objvel'], Test_data[k]['handle_init']))
        init_state_dict['qpos'] = Test_data[k]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k]['init']['qvel']
        init_state_dict['door_body_pos'] = Test_data[k]['init']['door_body_pos']  # fixed for each piece of demo data
        e.set_env_state(init_state_dict)
        if koopmanoption == 'Drafted': 
            tmp_input = np.append(hand_OriState, obj_OriState)
            z_t = torch.from_numpy(tmp_input)
        traj = NDP_agent.execute(z_t, z_t) # traj -> [T, dim]
        for t in range(horizon - 1):  # this loop is for system evolution, open loop control, no feedback
            if koopmanoption == 'Drafted':  
                x_t_1_computed = traj[t + 1]
                if not Velocity:
                    obj_pos_world = x_t_1_computed[28:31]  # handle pos
                else:
                    obj_pos_world = x_t_1_computed[56:59]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint states
            hand_pos_desired = hand_pos
            current = e.get_env_state()['qpos'][:28] # current state
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation)
            if Visualize:
                e.env.mj_render()
            # calculated state values using Koopman rollouts (in simulator)      
            obj_obs = e.get_obs()
            # compute the errors
            obj_pos_error[t, k] = np.mean(np.abs(obj_obs[35:38]))  # palm_pos - handle_pos
            obj_pos_error_koopman[t, k] = np.mean(np.abs(obj_obs[29:32] - obj_pos_world))  # obj_obs[29:32] -> palm position
            current_hinge_pos = obj_obs[28:29] # door opening angle
        if current_hinge_pos > 1.35:
            success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))
    return obj_pos_error, obj_pos_error_koopman, success_rate

def NGF_error_visualization(env_name, NGF_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize):
    print("Begin to compute the koopman rollout errors!")
    hand_pos_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    Computed_joint_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    Demo_joint_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    e = GymEnv(env_name)
    e.reset()
    state_dict = {}
    for k in tqdm(range(len(Test_data))):
        num_handpos = len(Test_data[k][0]['handpos'])
        if Velocity:
            hand_OriState = np.append(Test_data[k][0]['handpos'], Test_data[k][0]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k][0]['handpos']
        Demo_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        Computed_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        obj_OriState = np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objvel'], Test_data[k][0]['handle_init']))
        state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        state_dict['door_body_pos'] = Test_data[k][0]['init']['door_body_pos']  # fixed for each piece of demo data
        Initial_input = dict()
        Initial_input['hand_joint'] = torch.zeros(1, num_hand)
        Initial_input['hand_joint'][0] = torch.from_numpy(hand_OriState)
        Initial_input['object_states'] = torch.zeros(1, num_obj)
        Initial_input['object_states'][0] = torch.from_numpy(obj_OriState)
        traj = NGF_agent(Initial_input)[0].detach().numpy() # traj -> [T, dim]
        for t in range(len(Test_data[k]) - 1):
            e.KoopmanVisualize(seed = None, state_dict = state_dict) # visualize the demo data without any actions (more like a visualization)
            if koopmanoption == 'Drafted':  
                x_t_1_computed = traj[t + 1]
            # ground-truth state values
            gt_hand_pos = Test_data[k][t + 1]['handpos']
            # calculated state values using Koopman rollouts vb
            hand_pos = x_t_1_computed[:num_handpos]
            # compute the errors
            hand_pos_error_handJoint[t, k] = np.mean(np.abs(gt_hand_pos[4:] - hand_pos[4:]))  # probably not return the mean value?
            hand_pos_error_base[t, k] = np.mean(np.abs(gt_hand_pos[:4] - hand_pos[:4]))  # probably not return the mean value?
            Computed_joint_values[:, t + 1, k] = hand_pos
            Demo_joint_values[:, t + 1, k] = gt_hand_pos
            state_dict['qpos'] = np.append(hand_pos, np.zeros([2]))
            state_dict['qvel'] = np.zeros([30])
            if Visualize:  # Since the PID controller works well, we do not need to visualize the trajectories any more
                e.env.mj_render()
    if error_type == 'demo':
        return [hand_pos_error_handJoint, hand_pos_error_base, Computed_joint_values, Demo_joint_values]

def NGF_policy_control(env_name, controller, NGF_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize):
    print("Begin to compute the simulation errors!")
    hand_pos_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])  # palm_pos - handle_pos
    demo_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error_koopman = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    Computed_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    Demo_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    # for this task, we are only able to report the success rate for the simulation results
    # also, for this task, no failure cases.
    success_list_sim = []
    success_rate = str()
    for k in tqdm(range(len(Test_data))):
        num_handpos = len(Test_data[k][0]['handpos'])
        if Velocity:
            hand_OriState = np.append(Test_data[k][0]['handpos'], Test_data[k][0]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k][0]['handpos']
        obj_OriState = np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objvel'], Test_data[k][0]['handle_init']))
        init_state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        init_state_dict['door_body_pos'] = Test_data[k][0]['init']['door_body_pos']  # fixed for each piece of demo data
        e.set_env_state(init_state_dict)
        Initial_input = dict()
        Initial_input['hand_joint'] = torch.zeros(1, num_hand)
        Initial_input['hand_joint'][0] = torch.from_numpy(hand_OriState)
        Initial_input['object_states'] = torch.zeros(1, num_obj)
        Initial_input['object_states'][0] = torch.from_numpy(obj_OriState)
        traj = NGF_agent(Initial_input)[0].detach().numpy() # traj -> [T, dim]
        for t in range(len(Test_data[k]) - 1):  # this loop is for system evolution, open loop control, no feedback
            if koopmanoption == 'Drafted':  
                x_t_1_computed = traj[t + 1]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint state
            # hand_pos_desired = hand_pos + Test_data[k][0]['final_handpos']
            hand_pos_desired = hand_pos  # control frequency
            current = e.get_env_state()['qpos'][:28] # current state
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()
            Computed_torque_values[:, t, k] = NN_output
            Demo_torque_values[:, t, k] = Test_data[k][t]['action']
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation)
            if Visualize:
                e.env.mj_render()
            # ground-truth state values (obtained from RL in simulator)
            # gt_hand_pos = Test_data[k][t + 1]['handpos'] + Test_data[k][0]['final_handpos']
            gt_hand_pos = Test_data[k][t + 1]['handpos']
            # calculated state values using Koopman rollouts (in simulator)
            hand_pos = e.env.get_full_obs_visualization()[:num_handpos]     
            obj_obs = e.get_obs()
            # compute the errors
            hand_pos_error_handJoint[t, k] = np.mean(np.abs(gt_hand_pos[4:] - hand_pos[4:]))  # probably not return the mean value?
            hand_pos_PID_error_handJoint[t, k] = np.mean(np.abs(hand_pos_desired[4:] - hand_pos[4:]))  
            hand_pos_error_base[t, k] = np.mean(np.abs(gt_hand_pos[:4] - hand_pos[:4]))  # probably not return the mean value?
            hand_pos_PID_error_base[t, k] = np.mean(np.abs(hand_pos_desired[:4] - hand_pos[:4]))  
            # instead of defining the position error of the nail, we define it as the difference between the hammer tool and the nail
            obj_pos_error[t, k] = np.mean(np.abs(obj_obs[35:38]))  # palm_pos - handle_pos
            demo_pos_error[t, k] = np.mean(np.abs(Test_data[k][t + 1]['observation'])) # palm-handle
            current_hinge_pos = obj_obs[28:29] # door opening angle
        if current_hinge_pos > 1.35:
            success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))
    print("Success rate (RL) = %f" % (1))
    if error_type == 'demo':
        return [hand_pos_error_handJoint, hand_pos_PID_error_handJoint, hand_pos_error_base, hand_pos_PID_error_base, Computed_torque_values, Demo_torque_values, success_rate]
    else:
        return [obj_pos_error, obj_pos_error_koopman, demo_pos_error, success_rate]  

def NGF_policy_control_unseenTest(env_name, controller, NGF_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, Visualize):
    print("Testing the learned koopman dynamcis!")
    print("Begin to compute the simulation errors!")
    e = GymEnv(env_name)
    horizon = 70 # selected time steps that is enough to finish the task with good performance
    obj_pos_error = np.zeros([horizon - 1, len(Test_data)])
    obj_pos_error_koopman = np.zeros([horizon - 1, len(Test_data)])
    e.reset()
    init_state_dict = dict()
    # e.set_env_state(path['init_state_dict'])
    success_list_sim = []
    success_rate = str()
    for k in tqdm(range(len(Test_data))):
        num_handpos = len(Test_data[k]['handpos'])
        if Velocity:
            hand_OriState = np.append(Test_data[k]['handpos'], Test_data[k]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k]['handpos']
        obj_OriState = np.append(Test_data[k]['objpos'], np.append(Test_data[k]['objvel'], Test_data[k]['handle_init']))
        init_state_dict['qpos'] = Test_data[k]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k]['init']['qvel']
        init_state_dict['door_body_pos'] = Test_data[k]['init']['door_body_pos']  # fixed for each piece of demo data
        e.set_env_state(init_state_dict)
        Initial_input = dict()
        Initial_input['hand_joint'] = torch.zeros(1, num_hand)
        Initial_input['hand_joint'][0] = torch.from_numpy(hand_OriState)
        Initial_input['object_states'] = torch.zeros(1, num_obj)
        Initial_input['object_states'][0] = torch.from_numpy(obj_OriState)
        traj = NGF_agent(Initial_input)[0].detach().numpy() # traj -> [T, dim]
        for t in range(horizon - 1):  # this loop is for system evolution, open loop control, no feedback
            if koopmanoption == 'Drafted':  
                x_t_1_computed = traj[t + 1]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint states
            hand_pos_desired = hand_pos
            current = e.get_env_state()['qpos'][:28] # current state
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation)
            if Visualize:
                e.env.mj_render()
            # calculated state values using Koopman rollouts (in simulator)      
            obj_obs = e.get_obs()
            # compute the errors
            obj_pos_error[t, k] = np.mean(np.abs(obj_obs[35:38]))  # palm_pos - handle_pos
            current_hinge_pos = obj_obs[28:29] # door opening angle
        if current_hinge_pos > 1.35:
            success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))
    return obj_pos_error, obj_pos_error_koopman, success_rate

