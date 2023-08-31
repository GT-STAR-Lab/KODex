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
        obj_OriState = np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objorient'], Test_data[k][0]['objvel']))
        z_t = koopman_object.z(hand_OriState, obj_OriState)  # initial states in lifted space
        for t in range(len(Test_data[k]) - 1):
            if Velocity:
                hand_OriState = np.append(Test_data[k][t + 1]['handpos'], Test_data[k][t + 1]['handvel'])
            else:
                hand_OriState = Test_data[k][t + 1]['handpos']
            obj_OriState = np.append(Test_data[k][t + 1]['objpos'], np.append(Test_data[k][t + 1]['objorient'], Test_data[k][t + 1]['objvel']))
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
    obj_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    demo_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error_koopman = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    Computed_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    Demo_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    success_threshold = 10
    success_list_sim = []
    success_list_koopman = []
    success_list_demo = []
    success_rate = str()
    for k in tqdm(range(len(Test_data))):
        success_count_sim = np.zeros(len(Test_data[k]) - 1)
        success_count_koopman = np.zeros(len(Test_data[k]) - 1)
        success_count_RL = np.zeros(len(Test_data[k]) - 1)
        num_handpos = len(Test_data[k][0]['handpos'])
        if Velocity:
            hand_OriState = np.append(Test_data[k][0]['handpos'], Test_data[k][0]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k][0]['handpos']
        obj_OriState = np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objorient'], Test_data[k][0]['objvel']))
        init_state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        init_state_dict['obj_pos'] = Test_data[k][0]['init']['obj_pos']
        init_state_dict['target_pos'] = Test_data[k][0]['init']['target_pos']
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
                    obj_pos_world = x_t_1_computed[30:33]
                    # this is the obj_pos in the new frame (converged object trajecotry), as modified in training:
                    # tmp['objpos'] = objpos - obs[45:48] # converged object position (obs[45:48] -> target position)
                    # this is also the error of the object relocation (current position - goal position)
                    # to accutally obtain the position in the world frame:
                    # obj_pos_world += init_state_dict['target_pos']
                else:
                    obj_pos_world = x_t_1_computed[60:63]
            else:
                x_t_1_computed = koopman_object.z_inverse(torch.from_numpy(z_t_1_computed)).numpy()
                if not Velocity:
                    obj_pos_world = x_t_1_computed[30:33]
                else:
                    obj_pos_world = x_t_1_computed[60:63]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint state
            # hand_pos_desired = hand_pos + Test_data[k][0]['final_handpos']
            hand_pos_desired = hand_pos
            z_t = z_t_1_computed
            current = e.get_env_state()['hand_qpos'] # current state
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
            hand_pos = e.get_obs()[:num_handpos]     
            obj_obs = e.get_obs()
            # compute the errors
            hand_pos_error_handJoint[t, k] = np.mean(np.abs(gt_hand_pos[6:] - hand_pos[6:]))  # probably not return the mean value?
            hand_pos_PID_error_handJoint[t, k] = np.mean(np.abs(hand_pos_desired[6:] - hand_pos[6:]))  
            hand_pos_error_base[t, k] = np.mean(np.abs(gt_hand_pos[:6] - hand_pos[:6]))  # probably not return the mean value?
            hand_pos_PID_error_base[t, k] = np.mean(np.abs(hand_pos_desired[:6] - hand_pos[:6]))  
            obj_pos_error[t, k] = np.mean(np.abs(obj_obs[36:39]))  # obj_pos - tar_pos
            obj_pos_error_koopman[t, k] = np.mean(np.abs(obj_pos_world))  # goal error in koopman
            demo_pos_error[t, k] = np.mean(np.abs(Test_data[k][t]['observation']))
            if np.linalg.norm(obj_obs[36:39]) < 0.1:
                success_count_sim[t] = 1
            if np.linalg.norm(obj_pos_world) < 0.1:
                success_count_koopman[t] = 1
            if np.linalg.norm(Test_data[k][t]['observation']) < 0.1:
                success_count_RL[t] = 1
        if sum(success_count_sim) > success_threshold:
            success_list_sim.append(1)
        if sum(success_count_koopman) > success_threshold:
            success_list_koopman.append(1)
        if sum(success_count_RL) > success_threshold:
            success_list_demo.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f\n" % (len(success_list_sim) / len(Test_data))
    print("Success rate (koopman) = %f" % (len(success_list_koopman) / len(Test_data)))
    success_rate += "Success rate (koopman) = %f\n" % (len(success_list_koopman) / len(Test_data))
    print("Success rate (RL) = %f" % (len(success_list_demo) / len(Test_data)))
    if error_type == 'demo':
        return [hand_pos_error_handJoint, hand_pos_PID_error_handJoint, hand_pos_error_base, hand_pos_PID_error_base, Computed_torque_values, Demo_torque_values, success_rate]
    else:
        return [obj_pos_error, obj_pos_error_koopman, demo_pos_error, success_rate]     

def koopman_policy_control_unseenTest(env_name, controller, koopman_object, koopman_matrix, Test_data, Velocity, num_hand, num_obj, koopmanoption, Visualize):
    print("Testing the learned koopman dynamcis!")
    print("Begin to compute the simulation errors!")
    e = GymEnv(env_name)
    horizon = 100
    obj_ori_error = np.zeros([horizon - 1, len(Test_data)])
    obj_ori_error_koopman = np.zeros([horizon - 1, len(Test_data)])
    Computed_joint_values = np.zeros([num_hand, horizon, len(Test_data)])
    e.reset()
    init_state_dict = dict()
    success_threshold = 10
    success_list_sim = []
    success_list_koopman = []
    success_rate = str()
    for k in tqdm(range(len(Test_data))):
        success_count_sim = np.zeros(horizon - 1)
        success_count_koopman = np.zeros(horizon - 1)
        num_handpos = len(Test_data[k]['handpos'])
        if Velocity:
            hand_OriState = np.append(Test_data[k]['handpos'], Test_data[k]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k]['handpos']
        obj_OriState = np.append(Test_data[k]['objpos'], np.append(Test_data[k]['objorient'], Test_data[k]['objvel']))
        init_state_dict['qpos'] = Test_data[k]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k]['init']['qvel']
        init_state_dict['obj_pos'] = Test_data[k]['init']['obj_pos']
        init_state_dict['target_pos'] = Test_data[k]['init']['target_pos']
        Computed_joint_values[:, 0, k] = Test_data[k]['init']['qpos'][:num_hand]
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
                    obj_pos_world = x_t_1_computed[30:33]  # this is the obj_pos in the new frame (converged object trajecotry), as in training:
                    # tmp['objpos'] = objpos - obs[45:48] # converged object position (obs[45:48] -> target position)
                    # to accutally obtain the position in the world frame:
                    # obj_pos_world += init_state_dict['target_pos']
                else:
                    obj_pos_world = x_t_1_computed[60:63]
            else:
                x_t_1_computed = koopman_object.z_inverse(torch.from_numpy(z_t_1_computed)).numpy()
                if not Velocity:
                    obj_pos_world = x_t_1_computed[30:33]
                else:
                    obj_pos_world = x_t_1_computed[60:63]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint states
            Computed_joint_values[:, t + 1, k] = hand_pos
            z_t = z_t_1_computed
            hand_pos_desired = hand_pos
            current = e.get_env_state()['hand_qpos'] # current state
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()   
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation)
            if Visualize:
                e.env.mj_render()
            # calculated state values using Koopman rollouts (in simulator)      
            obj_obs = e.get_obs()
            # compute the errors
            obj_ori_error[t, k] = np.mean(np.abs(obj_obs[36:39]))
            obj_ori_error_koopman[t, k] = np.mean(np.abs(obj_pos_world))  # goal error in koopman
            if np.linalg.norm(obj_obs[36:39]) < 0.1:
                success_count_sim[t] = 1
            if np.linalg.norm(obj_pos_world) < 0.1:
                success_count_koopman[t] = 1
        if sum(success_count_sim) > success_threshold:
            success_list_sim.append(1)
        if sum(success_count_koopman) > success_threshold:
            success_list_koopman.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))
    print("Success rate (koopman) = %f" % (len(success_list_koopman) / len(Test_data)))
    return obj_ori_error, obj_ori_error_koopman, Computed_joint_values, success_rate
    
def koopman_error_visualization(env_name, koopman_object, koopman_matrix, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize):
    print("Begin to compute the koopman rollout errors!")
    hand_pos_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_vel_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_ori_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    Computed_joint_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    Demo_joint_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    e = GymEnv(env_name)
    e.reset()
    state_dict = {}
    for k in tqdm(range(len(Test_data))):
        num_handpos = len(Test_data[k][0]['handpos'])
        num_handvel = len(Test_data[k][0]['handvel'])
        num_objpos = len(Test_data[k][0]['objpos'])
        num_objorient = len(Test_data[k][0]['objorient'])
        num_objvel = len(Test_data[k][0]['objvel'])
        if Velocity:
            hand_OriState = np.append(Test_data[k][0]['handpos'], Test_data[k][0]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k][0]['handpos']
        Demo_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        Computed_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        obj_OriState = np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objorient'], Test_data[k][0]['objvel']))
        state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        state_dict['obj_pos'] = Test_data[k][0]['init']['obj_pos']
        state_dict['target_pos'] = Test_data[k][0]['init']['target_pos']  # fixed for each piece of demo data
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
            gt_obj_pos = Test_data[k][t + 1]['objpos']
            gt_obj_orient = Test_data[k][t + 1]['objorient']
            gt_obj_vel = Test_data[k][t + 1]['objvel']
            # calculated state values using Koopman rollouts vb
            hand_pos = x_t_1_computed[:num_handpos]
            obj_pos = x_t_1_computed[num_hand: num_hand + num_objpos]
            obj_orient = x_t_1_computed[num_hand + num_objpos: num_hand + num_objpos + num_objorient]
            obj_vel = x_t_1_computed[num_hand + num_objpos + num_objorient: num_hand + num_objpos + num_objorient + num_objvel]
            z_t = z_t_1_computed
            if Velocity:
                hand_vel = x_t_1_computed[num_handpos:num_handpos + num_handvel]
            else:
                hand_vel = np.zeros(num_handvel)
            # compute the errors
            hand_pos_error_handJoint[t, k] = np.mean(np.abs(gt_hand_pos[6:] - hand_pos[6:]))  # probably not return the mean value?
            hand_pos_error_base[t, k] = np.mean(np.abs(gt_hand_pos[:6] - hand_pos[:6]))  # probably not return the mean value?
            obj_pos_error[t, k] = np.mean(np.abs(gt_obj_pos - obj_pos) + np.abs(gt_obj_orient - obj_orient))
            obj_vel_error[t, k] = np.mean(np.abs(gt_obj_vel - obj_vel))
            Computed_joint_values[:, t + 1, k] = hand_pos
            Demo_joint_values[:, t + 1, k] = gt_hand_pos
            state_dict['qpos'] = x_t_1_computed[:36]
            state_dict['obj_pos'] = np.array([0.,0.,0.035])
            state_dict['qpos'][-6:-3] = np.array([0.,0.,0.])
            state_dict['qvel'] = np.append(hand_vel, x_t_1_computed[36:])
            # print("state_dict['obj_pos']:", state_dict['obj_pos'])
            # print("state_dict['qpos'][-6:-3]:", state_dict['qpos'][-6:-3])
            if Visualize:
                e.env.mj_render()
    if error_type == 'demo':
        return [hand_pos_error_handJoint, hand_pos_error_base, Computed_joint_values, Demo_joint_values]
    else:
        return [obj_ori_error] # this is incorrect and should not be used, because state_dict['qpos'] is not correct(it requires the orien in robot frame, but it actually is defined in world frame)
        # no goal error for this visualization function

def koopman_policy_visualization_unseenTest(env_name, koopman_object, koopman_matrix, Test_data, Velocity, num_hand, num_obj, koopmanoption):
    print("Testing the learned koopman dynamcis!")
    print("Begin to visualize the hand trajectories!")
    e = GymEnv(env_name)
    horizon = 100
    e.reset()
    state_dict = dict()
    for k in tqdm(range(len(Test_data))):
        num_handpos = len(Test_data[k]['handpos'])
        num_handvel = len(Test_data[k]['handvel'])
        if Velocity:
            hand_OriState = np.append(Test_data[k]['handpos'], Test_data[k]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k]['handpos']
        obj_OriState = np.append(Test_data[k]['objpos'], np.append(Test_data[k]['objorient'], Test_data[k]['objvel']))
        state_dict['qpos'] = Test_data[k]['init']['qpos']
        state_dict['qvel'] = Test_data[k]['init']['qvel']
        state_dict['obj_pos'] = Test_data[k]['init']['obj_pos']
        state_dict['target_pos'] = Test_data[k]['init']['target_pos']
        if koopmanoption == 'Drafted': 
            z_t = koopman_object.z(hand_OriState, obj_OriState)  # initial states in lifted space
        else:
            input_state = np.append(hand_OriState, obj_OriState)
            input_state = torch.from_numpy(input_state)
            z_t = koopman_object.z(input_state).numpy()  # lift the initial states and then roll it out using Koopman operator
        for t in range(horizon - 1):  # this loop is for system evolution, open loop control, no feedback
            e.KoopmanVisualize(seed = None, state_dict = state_dict) # visualize the demo data without any actions (more like a visualization)
            z_t_1_computed = np.dot(koopman_matrix, z_t)
            if koopmanoption == 'Drafted':  
                x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])
            else:
                x_t_1_computed = koopman_object.z_inverse(torch.from_numpy(z_t_1_computed)).numpy()
            z_t = z_t_1_computed
            if Velocity:
                hand_vel = x_t_1_computed[num_handpos:num_handpos + num_handvel]
            else:
                hand_vel = np.zeros(num_handvel)
            state_dict['qpos'] = x_t_1_computed[:36]
            state_dict['obj_pos'] = np.array([0.,0.,0.035])
            state_dict['qpos'][-6:-3] = np.array([0.,0.,0.])  # obj_Txyz
            state_dict['qvel'] = np.append(hand_vel, x_t_1_computed[36:])
            e.env.mj_render()

def BC_error_visualization(env_name, BC_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize):
    print("Begin to compute the koopman rollout errors!")
    hand_pos_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_vel_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_ori_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    Computed_joint_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    Demo_joint_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    e = GymEnv(env_name)
    e.reset()
    state_dict = {}
    for k in tqdm(range(len(Test_data))):
        num_handpos = len(Test_data[k][0]['handpos'])
        num_handvel = len(Test_data[k][0]['handvel'])
        num_objpos = len(Test_data[k][0]['objpos'])
        num_objorient = len(Test_data[k][0]['objorient'])
        num_objvel = len(Test_data[k][0]['objvel'])
        if Velocity:
            hand_OriState = np.append(Test_data[k][0]['handpos'], Test_data[k][0]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k][0]['handpos']
        Demo_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        Computed_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        obj_OriState = np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objorient'], Test_data[k][0]['objvel']))
        state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        state_dict['obj_pos'] = Test_data[k][0]['init']['obj_pos']
        state_dict['target_pos'] = Test_data[k][0]['init']['target_pos']  # fixed for each piece of demo data
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
            gt_obj_pos = Test_data[k][t + 1]['objpos']
            gt_obj_orient = Test_data[k][t + 1]['objorient']
            gt_obj_vel = Test_data[k][t + 1]['objvel']
            # calculated state values using Koopman rollouts vb
            hand_pos = x_t_1_computed[:num_handpos]
            obj_pos = x_t_1_computed[num_hand: num_hand + num_objpos]
            obj_orient = x_t_1_computed[num_hand + num_objpos: num_hand + num_objpos + num_objorient]
            obj_vel = x_t_1_computed[num_hand + num_objpos + num_objorient: num_hand + num_objpos + num_objorient + num_objvel]
            z_t = z_t_1_computed
            if Velocity:
                hand_vel = x_t_1_computed[num_handpos:num_handpos + num_handvel]
            else:
                hand_vel = np.zeros(num_handvel)
            # compute the errors
            hand_pos_error_handJoint[t, k] = np.mean(np.abs(gt_hand_pos[6:] - hand_pos[6:]))  # probably not return the mean value?
            hand_pos_error_base[t, k] = np.mean(np.abs(gt_hand_pos[:6] - hand_pos[:6]))  # probably not return the mean value?
            obj_pos_error[t, k] = np.mean(np.abs(gt_obj_pos - obj_pos) + np.abs(gt_obj_orient - obj_orient))
            obj_vel_error[t, k] = np.mean(np.abs(gt_obj_vel - obj_vel))
            Computed_joint_values[:, t + 1, k] = hand_pos
            Demo_joint_values[:, t + 1, k] = gt_hand_pos
            state_dict['qpos'] = x_t_1_computed[:36]
            state_dict['obj_pos'] = np.array([0.,0.,0.035])
            state_dict['qpos'][-6:-3] = np.array([0.,0.,0.])
            state_dict['qvel'] = np.append(hand_vel, x_t_1_computed[36:])
            # print("state_dict['obj_pos']:", state_dict['obj_pos'])
            # print("state_dict['qpos'][-6:-3]:", state_dict['qpos'][-6:-3])
            if Visualize:
                e.env.mj_render()
    if error_type == 'demo':
        return [hand_pos_error_handJoint, hand_pos_error_base, Computed_joint_values, Demo_joint_values]
    else:
        return [obj_ori_error] # this is incorrect and should not be used, because state_dict['qpos'] is not correct(it requires the orien in robot frame, but it actually is defined in world frame)
        # no goal error for this visualization function

def BC_policy_control(env_name, controller, BC_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize):
    print("Begin to compute the simulation errors!")
    hand_pos_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    demo_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error_koopman = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    Computed_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    Demo_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    success_threshold = 10
    success_list_sim = []
    success_list_koopman = []
    success_list_demo = []
    success_rate = str()
    for k in tqdm(range(len(Test_data))):
        success_count_sim = np.zeros(len(Test_data[k]) - 1)
        success_count_koopman = np.zeros(len(Test_data[k]) - 1)
        success_count_RL = np.zeros(len(Test_data[k]) - 1)
        num_handpos = len(Test_data[k][0]['handpos'])
        if Velocity:
            hand_OriState = np.append(Test_data[k][0]['handpos'], Test_data[k][0]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k][0]['handpos']
        obj_OriState = np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objorient'], Test_data[k][0]['objvel']))
        init_state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        init_state_dict['obj_pos'] = Test_data[k][0]['init']['obj_pos']
        init_state_dict['target_pos'] = Test_data[k][0]['init']['target_pos']
        e.set_env_state(init_state_dict)
        if koopmanoption == 'Drafted': 
            tmp_input = np.append(hand_OriState, obj_OriState)
            z_t = torch.from_numpy(tmp_input)
        for t in range(len(Test_data[k]) - 1):  # this loop is for system evolution, open loop control, no feedback
            z_t_1_computed = BC_agent(z_t)
            if koopmanoption == 'Drafted':  
                x_t_1_computed = z_t_1_computed.detach().numpy()
                if not Velocity:
                    obj_pos_world = x_t_1_computed[30:33]
                    # this is the obj_pos in the new frame (converged object trajecotry), as modified in training:
                    # tmp['objpos'] = objpos - obs[45:48] # converged object position (obs[45:48] -> target position)
                    # this is also the error of the object relocation (current position - goal position)
                    # to accutally obtain the position in the world frame:
                    # obj_pos_world += init_state_dict['target_pos']
                else:
                    obj_pos_world = x_t_1_computed[60:63]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint state
            # hand_pos_desired = hand_pos + Test_data[k][0]['final_handpos']
            hand_pos_desired = hand_pos
            z_t = z_t_1_computed
            current = e.get_env_state()['hand_qpos'] # current state
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
            hand_pos = e.get_obs()[:num_handpos]     
            obj_obs = e.get_obs()
            # compute the errors
            hand_pos_error_handJoint[t, k] = np.mean(np.abs(gt_hand_pos[6:] - hand_pos[6:]))  # probably not return the mean value?
            hand_pos_PID_error_handJoint[t, k] = np.mean(np.abs(hand_pos_desired[6:] - hand_pos[6:]))  
            hand_pos_error_base[t, k] = np.mean(np.abs(gt_hand_pos[:6] - hand_pos[:6]))  # probably not return the mean value?
            hand_pos_PID_error_base[t, k] = np.mean(np.abs(hand_pos_desired[:6] - hand_pos[:6]))  
            obj_pos_error[t, k] = np.mean(np.abs(obj_obs[36:39]))  # obj_pos - tar_pos
            obj_pos_error_koopman[t, k] = np.mean(np.abs(obj_pos_world))  # goal error in koopman
            demo_pos_error[t, k] = np.mean(np.abs(Test_data[k][t]['observation']))
            if np.linalg.norm(obj_obs[36:39]) < 0.1:
                success_count_sim[t] = 1
            if np.linalg.norm(obj_pos_world) < 0.1:
                success_count_koopman[t] = 1
            if np.linalg.norm(Test_data[k][t]['observation']) < 0.1:
                success_count_RL[t] = 1
        if sum(success_count_sim) > success_threshold:
            success_list_sim.append(1)
        if sum(success_count_koopman) > success_threshold:
            success_list_koopman.append(1)
        if sum(success_count_RL) > success_threshold:
            success_list_demo.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f\n" % (len(success_list_sim) / len(Test_data))
    print("Success rate (koopman) = %f" % (len(success_list_koopman) / len(Test_data)))
    success_rate += "Success rate (koopman) = %f\n" % (len(success_list_koopman) / len(Test_data))
    print("Success rate (RL) = %f" % (len(success_list_demo) / len(Test_data)))
    if error_type == 'demo':
        return [hand_pos_error_handJoint, hand_pos_PID_error_handJoint, hand_pos_error_base, hand_pos_PID_error_base, Computed_torque_values, Demo_torque_values, success_rate]
    else:
        return [obj_pos_error, obj_pos_error_koopman, demo_pos_error, success_rate]     

def BC_policy_control_unseenTest(env_name, controller, BC_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, Visualize):
    print("Testing the learned koopman dynamcis!")
    print("Begin to compute the simulation errors!")
    e = GymEnv(env_name)
    horizon = 100
    obj_ori_error = np.zeros([horizon - 1, len(Test_data)])
    obj_ori_error_koopman = np.zeros([horizon - 1, len(Test_data)])
    Computed_joint_values = np.zeros([num_hand, horizon, len(Test_data)])
    e.reset()
    init_state_dict = dict()
    success_threshold = 10
    success_list_sim = []
    success_list_koopman = []
    success_rate = str()
    for k in tqdm(range(len(Test_data))):
        success_count_sim = np.zeros(horizon - 1)
        success_count_koopman = np.zeros(horizon - 1)
        num_handpos = len(Test_data[k]['handpos'])
        if Velocity:
            hand_OriState = np.append(Test_data[k]['handpos'], Test_data[k]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k]['handpos']
        obj_OriState = np.append(Test_data[k]['objpos'], np.append(Test_data[k]['objorient'], Test_data[k]['objvel']))
        init_state_dict['qpos'] = Test_data[k]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k]['init']['qvel']
        init_state_dict['obj_pos'] = Test_data[k]['init']['obj_pos']
        init_state_dict['target_pos'] = Test_data[k]['init']['target_pos']
        Computed_joint_values[:, 0, k] = Test_data[k]['init']['qpos'][:num_hand]
        e.set_env_state(init_state_dict)
        if koopmanoption == 'Drafted': 
            tmp_input = np.append(hand_OriState, obj_OriState)
            z_t = torch.from_numpy(tmp_input)
        for t in range(horizon - 1):  # this loop is for system evolution, open loop control, no feedback
            z_t_1_computed = BC_agent(z_t)
            if koopmanoption == 'Drafted':  
                x_t_1_computed = z_t_1_computed.detach().numpy()
                if not Velocity:
                    obj_pos_world = x_t_1_computed[30:33]  # this is the obj_pos in the new frame (converged object trajecotry), as in training:
                    # tmp['objpos'] = objpos - obs[45:48] # converged object position (obs[45:48] -> target position)
                    # to accutally obtain the position in the world frame:
                    # obj_pos_world += init_state_dict['target_pos']
                else:
                    obj_pos_world = x_t_1_computed[60:63]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint states
            Computed_joint_values[:, t + 1, k] = hand_pos
            z_t = z_t_1_computed
            hand_pos_desired = hand_pos
            current = e.get_env_state()['hand_qpos'] # current state
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()   
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation)
            if Visualize:
                e.env.mj_render()
            # calculated state values using Koopman rollouts (in simulator)      
            obj_obs = e.get_obs()
            # compute the errors
            obj_ori_error[t, k] = np.mean(np.abs(obj_obs[36:39]))
            obj_ori_error_koopman[t, k] = np.mean(np.abs(obj_pos_world))  # goal error in koopman
            if np.linalg.norm(obj_obs[36:39]) < 0.1:
                success_count_sim[t] = 1
            if np.linalg.norm(obj_pos_world) < 0.1:
                success_count_koopman[t] = 1
        if sum(success_count_sim) > success_threshold:
            success_list_sim.append(1)
        if sum(success_count_koopman) > success_threshold:
            success_list_koopman.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))
    print("Success rate (koopman) = %f" % (len(success_list_koopman) / len(Test_data)))
    return obj_ori_error, obj_ori_error_koopman, Computed_joint_values, success_rate

def LSTM_error_visualization(env_name, LSTM_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize):
    print("Begin to compute the koopman rollout errors!")
    hand_pos_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_vel_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_ori_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    Computed_joint_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    Demo_joint_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    e = GymEnv(env_name)
    e.reset()
    state_dict = {}
    for k in tqdm(range(len(Test_data))):
        num_handpos = len(Test_data[k][0]['handpos'])
        num_handvel = len(Test_data[k][0]['handvel'])
        num_objpos = len(Test_data[k][0]['objpos'])
        num_objorient = len(Test_data[k][0]['objorient'])
        num_objvel = len(Test_data[k][0]['objvel'])
        if Velocity:
            hand_OriState = np.append(Test_data[k][0]['handpos'], Test_data[k][0]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k][0]['handpos']
        Demo_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        Computed_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        obj_OriState = np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objorient'], Test_data[k][0]['objvel']))
        state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        state_dict['obj_pos'] = Test_data[k][0]['init']['obj_pos']
        state_dict['target_pos'] = Test_data[k][0]['init']['target_pos']  # fixed for each piece of demo data
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
            gt_obj_pos = Test_data[k][t + 1]['objpos']
            gt_obj_orient = Test_data[k][t + 1]['objorient']
            gt_obj_vel = Test_data[k][t + 1]['objvel']
            # calculated state values using Koopman rollouts vb
            hand_pos = x_t_1_computed[:num_handpos]
            obj_pos = x_t_1_computed[num_hand: num_hand + num_objpos]
            obj_orient = x_t_1_computed[num_hand + num_objpos: num_hand + num_objpos + num_objorient]
            obj_vel = x_t_1_computed[num_hand + num_objpos + num_objorient: num_hand + num_objpos + num_objorient + num_objvel]
            z_t = z_t_1_computed
            if Velocity:
                hand_vel = x_t_1_computed[num_handpos:num_handpos + num_handvel]
            else:
                hand_vel = np.zeros(num_handvel)
            # compute the errors
            hand_pos_error_handJoint[t, k] = np.mean(np.abs(gt_hand_pos[6:] - hand_pos[6:]))  # probably not return the mean value?
            hand_pos_error_base[t, k] = np.mean(np.abs(gt_hand_pos[:6] - hand_pos[:6]))  # probably not return the mean value?
            obj_pos_error[t, k] = np.mean(np.abs(gt_obj_pos - obj_pos) + np.abs(gt_obj_orient - obj_orient))
            obj_vel_error[t, k] = np.mean(np.abs(gt_obj_vel - obj_vel))
            Computed_joint_values[:, t + 1, k] = hand_pos
            Demo_joint_values[:, t + 1, k] = gt_hand_pos
            state_dict['qpos'] = x_t_1_computed[:36]
            state_dict['obj_pos'] = np.array([0.,0.,0.035])
            state_dict['qpos'][-6:-3] = np.array([0.,0.,0.])
            state_dict['qvel'] = np.append(hand_vel, x_t_1_computed[36:])
            # print("state_dict['obj_pos']:", state_dict['obj_pos'])
            # print("state_dict['qpos'][-6:-3]:", state_dict['qpos'][-6:-3])
            if Visualize:
                e.env.mj_render()
    if error_type == 'demo':
        return [hand_pos_error_handJoint, hand_pos_error_base, Computed_joint_values, Demo_joint_values]
    else:
        return [obj_ori_error] # this is incorrect and should not be used, because state_dict['qpos'] is not correct(it requires the orien in robot frame, but it actually is defined in world frame)
        # no goal error for this visualization function

def LSTM_policy_control(env_name, controller, LSTM_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize):
    print("Begin to compute the simulation errors!")
    hand_pos_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    demo_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error_koopman = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    Computed_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    Demo_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    success_threshold = 10
    success_list_sim = []
    success_list_koopman = []
    success_list_demo = []
    success_rate = str()
    for k in tqdm(range(len(Test_data))):
        success_count_sim = np.zeros(len(Test_data[k]) - 1)
        success_count_koopman = np.zeros(len(Test_data[k]) - 1)
        success_count_RL = np.zeros(len(Test_data[k]) - 1)
        num_handpos = len(Test_data[k][0]['handpos'])
        if Velocity:
            hand_OriState = np.append(Test_data[k][0]['handpos'], Test_data[k][0]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k][0]['handpos']
        obj_OriState = np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objorient'], Test_data[k][0]['objvel']))
        init_state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        init_state_dict['obj_pos'] = Test_data[k][0]['init']['obj_pos']
        init_state_dict['target_pos'] = Test_data[k][0]['init']['target_pos']
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
                    obj_pos_world = x_t_1_computed[30:33]
                    # this is the obj_pos in the new frame (converged object trajecotry), as modified in training:
                    # tmp['objpos'] = objpos - obs[45:48] # converged object position (obs[45:48] -> target position)
                    # this is also the error of the object relocation (current position - goal position)
                    # to accutally obtain the position in the world frame:
                    # obj_pos_world += init_state_dict['target_pos']
                else:
                    obj_pos_world = x_t_1_computed[60:63]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint state
            # hand_pos_desired = hand_pos + Test_data[k][0]['final_handpos']
            hand_pos_desired = hand_pos
            z_t = z_t_1_computed
            current = e.get_env_state()['hand_qpos'] # current state
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
            hand_pos = e.get_obs()[:num_handpos]     
            obj_obs = e.get_obs()
            # compute the errors
            hand_pos_error_handJoint[t, k] = np.mean(np.abs(gt_hand_pos[6:] - hand_pos[6:]))  # probably not return the mean value?
            hand_pos_PID_error_handJoint[t, k] = np.mean(np.abs(hand_pos_desired[6:] - hand_pos[6:]))  
            hand_pos_error_base[t, k] = np.mean(np.abs(gt_hand_pos[:6] - hand_pos[:6]))  # probably not return the mean value?
            hand_pos_PID_error_base[t, k] = np.mean(np.abs(hand_pos_desired[:6] - hand_pos[:6]))  
            obj_pos_error[t, k] = np.mean(np.abs(obj_obs[36:39]))  # obj_pos - tar_pos
            obj_pos_error_koopman[t, k] = np.mean(np.abs(obj_pos_world))  # goal error in koopman
            demo_pos_error[t, k] = np.mean(np.abs(Test_data[k][t]['observation']))
            if np.linalg.norm(obj_obs[36:39]) < 0.1:
                success_count_sim[t] = 1
            if np.linalg.norm(obj_pos_world) < 0.1:
                success_count_koopman[t] = 1
            if np.linalg.norm(Test_data[k][t]['observation']) < 0.1:
                success_count_RL[t] = 1
        if sum(success_count_sim) > success_threshold:
            success_list_sim.append(1)
        if sum(success_count_koopman) > success_threshold:
            success_list_koopman.append(1)
        if sum(success_count_RL) > success_threshold:
            success_list_demo.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f\n" % (len(success_list_sim) / len(Test_data))
    print("Success rate (koopman) = %f" % (len(success_list_koopman) / len(Test_data)))
    success_rate += "Success rate (koopman) = %f\n" % (len(success_list_koopman) / len(Test_data))
    print("Success rate (RL) = %f" % (len(success_list_demo) / len(Test_data)))
    if error_type == 'demo':
        return [hand_pos_error_handJoint, hand_pos_PID_error_handJoint, hand_pos_error_base, hand_pos_PID_error_base, Computed_torque_values, Demo_torque_values, success_rate]
    else:
        return [obj_pos_error, obj_pos_error_koopman, demo_pos_error, success_rate]     

def LSTM_policy_control_unseenTest(env_name, controller, LSTM_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, Visualize):
    print("Testing the learned koopman dynamcis!")
    print("Begin to compute the simulation errors!")
    e = GymEnv(env_name)
    horizon = 100
    obj_ori_error = np.zeros([horizon - 1, len(Test_data)])
    obj_ori_error_koopman = np.zeros([horizon - 1, len(Test_data)])
    Computed_joint_values = np.zeros([num_hand, horizon, len(Test_data)])
    e.reset()
    init_state_dict = dict()
    success_threshold = 10
    success_list_sim = []
    success_list_koopman = []
    success_rate = str()
    for k in tqdm(range(len(Test_data))):
        success_count_sim = np.zeros(horizon - 1)
        success_count_koopman = np.zeros(horizon - 1)
        num_handpos = len(Test_data[k]['handpos'])
        if Velocity:
            hand_OriState = np.append(Test_data[k]['handpos'], Test_data[k]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k]['handpos']
        obj_OriState = np.append(Test_data[k]['objpos'], np.append(Test_data[k]['objorient'], Test_data[k]['objvel']))
        init_state_dict['qpos'] = Test_data[k]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k]['init']['qvel']
        init_state_dict['obj_pos'] = Test_data[k]['init']['obj_pos']
        init_state_dict['target_pos'] = Test_data[k]['init']['target_pos']
        Computed_joint_values[:, 0, k] = Test_data[k]['init']['qpos'][:num_hand]
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
                    obj_pos_world = x_t_1_computed[30:33]  # this is the obj_pos in the new frame (converged object trajecotry), as in training:
                    # tmp['objpos'] = objpos - obs[45:48] # converged object position (obs[45:48] -> target position)
                    # to accutally obtain the position in the world frame:
                    # obj_pos_world += init_state_dict['target_pos']
                else:
                    obj_pos_world = x_t_1_computed[60:63]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint states
            Computed_joint_values[:, t + 1, k] = hand_pos
            z_t = z_t_1_computed
            hand_pos_desired = hand_pos
            current = e.get_env_state()['hand_qpos'] # current state
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()   
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation)
            if Visualize:
                e.env.mj_render()
            # calculated state values using Koopman rollouts (in simulator)      
            obj_obs = e.get_obs()
            # compute the errors
            obj_ori_error[t, k] = np.mean(np.abs(obj_obs[36:39]))
            obj_ori_error_koopman[t, k] = np.mean(np.abs(obj_pos_world))  # goal error in koopman
            if np.linalg.norm(obj_obs[36:39]) < 0.1:
                success_count_sim[t] = 1
            if np.linalg.norm(obj_pos_world) < 0.1:
                success_count_koopman[t] = 1
        if sum(success_count_sim) > success_threshold:
            success_list_sim.append(1)
        if sum(success_count_koopman) > success_threshold:
            success_list_koopman.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))
    print("Success rate (koopman) = %f" % (len(success_list_koopman) / len(Test_data)))
    return obj_ori_error, obj_ori_error_koopman, Computed_joint_values, success_rate

def NDP_error_visualization(env_name, NDP_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize):
    print("Begin to compute the NDP rollout errors!")
    hand_pos_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_vel_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_ori_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    Computed_joint_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    Demo_joint_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    e = GymEnv(env_name)
    e.reset()
    state_dict = {}
    for k in tqdm(range(len(Test_data))):
        num_handpos = len(Test_data[k][0]['handpos'])
        num_handvel = len(Test_data[k][0]['handvel'])
        num_objpos = len(Test_data[k][0]['objpos'])
        num_objorient = len(Test_data[k][0]['objorient'])
        num_objvel = len(Test_data[k][0]['objvel'])
        if Velocity:
            hand_OriState = np.append(Test_data[k][0]['handpos'], Test_data[k][0]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k][0]['handpos']
        Demo_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        Computed_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        obj_OriState = np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objorient'], Test_data[k][0]['objvel']))
        state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        state_dict['obj_pos'] = Test_data[k][0]['init']['obj_pos']
        state_dict['target_pos'] = Test_data[k][0]['init']['target_pos']  # fixed for each piece of demo data
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
            gt_obj_pos = Test_data[k][t + 1]['objpos']
            gt_obj_orient = Test_data[k][t + 1]['objorient']
            gt_obj_vel = Test_data[k][t + 1]['objvel']
            # calculated state values using Koopman rollouts vb
            hand_pos = x_t_1_computed[:num_handpos]
            obj_pos = x_t_1_computed[num_hand: num_hand + num_objpos]
            obj_orient = x_t_1_computed[num_hand + num_objpos: num_hand + num_objpos + num_objorient]
            obj_vel = x_t_1_computed[num_hand + num_objpos + num_objorient: num_hand + num_objpos + num_objorient + num_objvel]
            if Velocity:
                hand_vel = x_t_1_computed[num_handpos:num_handpos + num_handvel]
            else:
                hand_vel = np.zeros(num_handvel)
            # compute the errors
            hand_pos_error_handJoint[t, k] = np.mean(np.abs(gt_hand_pos[6:] - hand_pos[6:]))  # probably not return the mean value?
            hand_pos_error_base[t, k] = np.mean(np.abs(gt_hand_pos[:6] - hand_pos[:6]))  # probably not return the mean value?
            obj_pos_error[t, k] = np.mean(np.abs(gt_obj_pos - obj_pos) + np.abs(gt_obj_orient - obj_orient))
            obj_vel_error[t, k] = np.mean(np.abs(gt_obj_vel - obj_vel))
            Computed_joint_values[:, t + 1, k] = hand_pos
            Demo_joint_values[:, t + 1, k] = gt_hand_pos
            state_dict['qpos'] = x_t_1_computed[:36]
            state_dict['obj_pos'] = np.array([0.,0.,0.035])
            state_dict['qpos'][-6:-3] = np.array([0.,0.,0.])
            state_dict['qvel'] = np.append(hand_vel, x_t_1_computed[36:])
            # print("state_dict['obj_pos']:", state_dict['obj_pos'])
            # print("state_dict['qpos'][-6:-3]:", state_dict['qpos'][-6:-3])
            if Visualize:
                e.env.mj_render()
    if error_type == 'demo':
        return [hand_pos_error_handJoint, hand_pos_error_base, Computed_joint_values, Demo_joint_values]
    else:
        return [obj_ori_error] # this is incorrect and should not be used, because state_dict['qpos'] is not correct(it requires the orien in robot frame, but it actually is defined in world frame)
        # no goal error for this visualization function

def NDP_policy_control(env_name, controller, NDP_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize):
    print("Begin to compute the simulation errors!")
    hand_pos_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    demo_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error_koopman = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    Computed_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    Demo_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    success_threshold = 10
    success_list_sim = []
    success_list_koopman = []
    success_list_demo = []
    success_rate = str()
    for k in tqdm(range(len(Test_data))):
        success_count_sim = np.zeros(len(Test_data[k]) - 1)
        success_count_koopman = np.zeros(len(Test_data[k]) - 1)
        success_count_RL = np.zeros(len(Test_data[k]) - 1)
        num_handpos = len(Test_data[k][0]['handpos'])
        if Velocity:
            hand_OriState = np.append(Test_data[k][0]['handpos'], Test_data[k][0]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k][0]['handpos']
        obj_OriState = np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objorient'], Test_data[k][0]['objvel']))
        init_state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        init_state_dict['obj_pos'] = Test_data[k][0]['init']['obj_pos']
        init_state_dict['target_pos'] = Test_data[k][0]['init']['target_pos']
        e.set_env_state(init_state_dict)
        if koopmanoption == 'Drafted': 
            tmp_input = np.append(hand_OriState, obj_OriState)
            z_t = torch.from_numpy(tmp_input)
        traj = NDP_agent.execute(z_t, z_t) # traj -> [T, dim]
        for t in range(len(Test_data[k]) - 1):  # this loop is for system evolution, open loop control, no feedback
            if koopmanoption == 'Drafted':  
                x_t_1_computed = traj[t + 1]
                if not Velocity:
                    obj_pos_world = x_t_1_computed[30:33]
                    # this is the obj_pos in the new frame (converged object trajecotry), as modified in training:
                    # tmp['objpos'] = objpos - obs[45:48] # converged object position (obs[45:48] -> target position)
                    # this is also the error of the object relocation (current position - goal position)
                    # to accutally obtain the position in the world frame:
                    # obj_pos_world += init_state_dict['target_pos']
                else:
                    obj_pos_world = x_t_1_computed[60:63]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint state
            # hand_pos_desired = hand_pos + Test_data[k][0]['final_handpos']
            hand_pos_desired = hand_pos
            current = e.get_env_state()['hand_qpos'] # current state
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
            hand_pos = e.get_obs()[:num_handpos]     
            obj_obs = e.get_obs()
            # compute the errors
            hand_pos_error_handJoint[t, k] = np.mean(np.abs(gt_hand_pos[6:] - hand_pos[6:]))  # probably not return the mean value?
            hand_pos_PID_error_handJoint[t, k] = np.mean(np.abs(hand_pos_desired[6:] - hand_pos[6:]))  
            hand_pos_error_base[t, k] = np.mean(np.abs(gt_hand_pos[:6] - hand_pos[:6]))  # probably not return the mean value?
            hand_pos_PID_error_base[t, k] = np.mean(np.abs(hand_pos_desired[:6] - hand_pos[:6]))  
            obj_pos_error[t, k] = np.mean(np.abs(obj_obs[36:39]))  # obj_pos - tar_pos
            obj_pos_error_koopman[t, k] = np.mean(np.abs(obj_pos_world))  # goal error in koopman
            demo_pos_error[t, k] = np.mean(np.abs(Test_data[k][t]['observation']))
            if np.linalg.norm(obj_obs[36:39]) < 0.1:
                success_count_sim[t] = 1
            if np.linalg.norm(obj_pos_world) < 0.1:
                success_count_koopman[t] = 1
            if np.linalg.norm(Test_data[k][t]['observation']) < 0.1:
                success_count_RL[t] = 1
        if sum(success_count_sim) > success_threshold:
            success_list_sim.append(1)
        if sum(success_count_koopman) > success_threshold:
            success_list_koopman.append(1)
        if sum(success_count_RL) > success_threshold:
            success_list_demo.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f\n" % (len(success_list_sim) / len(Test_data))
    print("Success rate (koopman) = %f" % (len(success_list_koopman) / len(Test_data)))
    success_rate += "Success rate (koopman) = %f\n" % (len(success_list_koopman) / len(Test_data))
    print("Success rate (RL) = %f" % (len(success_list_demo) / len(Test_data)))
    if error_type == 'demo':
        return [hand_pos_error_handJoint, hand_pos_PID_error_handJoint, hand_pos_error_base, hand_pos_PID_error_base, Computed_torque_values, Demo_torque_values, success_rate]
    else:
        return [obj_pos_error, obj_pos_error_koopman, demo_pos_error, success_rate]     

def NDP_policy_control_unseenTest(env_name, controller, NDP_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, Visualize):
    print("Testing the learned koopman dynamcis!")
    print("Begin to compute the simulation errors!")
    e = GymEnv(env_name)
    horizon = 100
    obj_ori_error = np.zeros([horizon - 1, len(Test_data)])
    obj_ori_error_koopman = np.zeros([horizon - 1, len(Test_data)])
    Computed_joint_values = np.zeros([num_hand, horizon, len(Test_data)])
    e.reset()
    init_state_dict = dict()
    success_threshold = 10
    success_list_sim = []
    success_list_koopman = []
    success_rate = str()
    for k in tqdm(range(len(Test_data))):
        success_count_sim = np.zeros(horizon - 1)
        success_count_koopman = np.zeros(horizon - 1)
        num_handpos = len(Test_data[k]['handpos'])
        if Velocity:
            hand_OriState = np.append(Test_data[k]['handpos'], Test_data[k]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k]['handpos']
        obj_OriState = np.append(Test_data[k]['objpos'], np.append(Test_data[k]['objorient'], Test_data[k]['objvel']))
        init_state_dict['qpos'] = Test_data[k]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k]['init']['qvel']
        init_state_dict['obj_pos'] = Test_data[k]['init']['obj_pos']
        init_state_dict['target_pos'] = Test_data[k]['init']['target_pos']
        Computed_joint_values[:, 0, k] = Test_data[k]['init']['qpos'][:num_hand]
        e.set_env_state(init_state_dict)
        if koopmanoption == 'Drafted': 
            tmp_input = np.append(hand_OriState, obj_OriState)
            z_t = torch.from_numpy(tmp_input)
        traj = NDP_agent.execute(z_t, z_t) # traj -> [T, dim]
        for t in range(horizon - 1):  # this loop is for system evolution, open loop control, no feedback
            if koopmanoption == 'Drafted':  
                x_t_1_computed = traj[t + 1]
                if not Velocity:
                    obj_pos_world = x_t_1_computed[30:33]  # this is the obj_pos in the new frame (converged object trajecotry), as in training:
                    # tmp['objpos'] = objpos - obs[45:48] # converged object position (obs[45:48] -> target position)
                    # to accutally obtain the position in the world frame:
                    # obj_pos_world += init_state_dict['target_pos']
                else:
                    obj_pos_world = x_t_1_computed[60:63]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint states
            Computed_joint_values[:, t + 1, k] = hand_pos
            hand_pos_desired = hand_pos
            current = e.get_env_state()['hand_qpos'] # current state
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()   
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation)
            if Visualize:
                e.env.mj_render()
            # calculated state values using Koopman rollouts (in simulator)      
            obj_obs = e.get_obs()
            # compute the errors
            obj_ori_error[t, k] = np.mean(np.abs(obj_obs[36:39]))
            obj_ori_error_koopman[t, k] = np.mean(np.abs(obj_pos_world))  # goal error in koopman
            if np.linalg.norm(obj_obs[36:39]) < 0.1:
                success_count_sim[t] = 1
            if np.linalg.norm(obj_pos_world) < 0.1:
                success_count_koopman[t] = 1
        if sum(success_count_sim) > success_threshold:
            success_list_sim.append(1)
        if sum(success_count_koopman) > success_threshold:
            success_list_koopman.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))
    print("Success rate (koopman) = %f" % (len(success_list_koopman) / len(Test_data)))
    return obj_ori_error, obj_ori_error_koopman, Computed_joint_values, success_rate

def NGF_error_visualization(env_name, NGF_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize):
    print("Begin to compute the NGF rollout errors!")
    hand_pos_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_ori_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
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
        obj_OriState = np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objorient'], Test_data[k][0]['objvel']))
        state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        state_dict['obj_pos'] = Test_data[k][0]['init']['obj_pos']
        state_dict['target_pos'] = Test_data[k][0]['init']['target_pos']  # fixed for each piece of demo data
        # {'q0': hand_joint_0 (torch_tensor), 'obj0': object_state_0 (torch_tensor), 'num_traj': num_traj (int), 'time_horizon': time_horizon (int)} 
        Initial_input = dict()
        Initial_input['hand_joint'] = torch.zeros(1, num_hand)
        Initial_input['hand_joint'][0] = torch.from_numpy(hand_OriState)
        Initial_input['object_states'] = torch.zeros(1, num_obj)
        Initial_input['object_states'][0] = torch.from_numpy(obj_OriState)
        traj = NGF_agent(Initial_input)[0].detach().numpy() # traj -> [T, dim]
        for t in range(len(Test_data[k]) - 1):
            if koopmanoption == 'Drafted':  
                x_t_1_computed = traj[t + 1]
            # ground-truth state values
            gt_hand_pos = Test_data[k][t + 1]['handpos']
            # calculated state values using Koopman rollouts vb
            hand_pos = x_t_1_computed[:num_handpos]
            if Velocity:
                hand_vel = x_t_1_computed[num_handpos:num_handpos + num_handvel]
            else:
                hand_vel = np.zeros(num_handvel)
            # compute the errors
            hand_pos_error_handJoint[t, k] = np.mean(np.abs(gt_hand_pos[6:] - hand_pos[6:]))  # probably not return the mean value?
            hand_pos_error_base[t, k] = np.mean(np.abs(gt_hand_pos[:6] - hand_pos[:6]))  # probably not return the mean value?
            Computed_joint_values[:, t + 1, k] = hand_pos
            Demo_joint_values[:, t + 1, k] = gt_hand_pos
            state_dict['qpos'] = x_t_1_computed[:36]
            state_dict['obj_pos'] = np.array([0.,0.,0.035])
            state_dict['qpos'][-6:-3] = np.array([0.,0.,0.])
            state_dict['qvel'] = np.append(hand_vel, x_t_1_computed[36:])
            # print("state_dict['obj_pos']:", state_dict['obj_pos'])
            # print("state_dict['qpos'][-6:-3]:", state_dict['qpos'][-6:-3])
    if error_type == 'demo':
        return [hand_pos_error_handJoint, hand_pos_error_base, Computed_joint_values, Demo_joint_values]
    else:
        return [obj_ori_error] # this is incorrect and should not be used, because state_dict['qpos'] is not correct(it requires the orien in robot frame, but it actually is defined in world frame)
        # no goal error for this visualization function

def NGF_policy_control(env_name, controller, NGF_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize):
    print("Begin to compute the NGF simulation errors!")
    hand_pos_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    demo_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error_koopman = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    Computed_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    Demo_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    success_threshold = 10
    success_list_sim = []
    success_list_demo = []
    success_rate = str()
    for k in tqdm(range(len(Test_data))):
        success_count_sim = np.zeros(len(Test_data[k]) - 1)
        success_count_RL = np.zeros(len(Test_data[k]) - 1)
        num_handpos = len(Test_data[k][0]['handpos'])
        if Velocity:
            hand_OriState = np.append(Test_data[k][0]['handpos'], Test_data[k][0]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k][0]['handpos']
        obj_OriState = np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objorient'], Test_data[k][0]['objvel']))
        init_state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        init_state_dict['obj_pos'] = Test_data[k][0]['init']['obj_pos']
        init_state_dict['target_pos'] = Test_data[k][0]['init']['target_pos']
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
            hand_pos_desired = hand_pos
            current = e.get_env_state()['hand_qpos'] # current state
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
            hand_pos = e.get_obs()[:num_handpos]     
            obj_obs = e.get_obs()
            # compute the errors
            hand_pos_error_handJoint[t, k] = np.mean(np.abs(gt_hand_pos[6:] - hand_pos[6:]))  # probably not return the mean value?
            hand_pos_PID_error_handJoint[t, k] = np.mean(np.abs(hand_pos_desired[6:] - hand_pos[6:]))  
            hand_pos_error_base[t, k] = np.mean(np.abs(gt_hand_pos[:6] - hand_pos[:6]))  # probably not return the mean value?
            hand_pos_PID_error_base[t, k] = np.mean(np.abs(hand_pos_desired[:6] - hand_pos[:6]))  
            obj_pos_error[t, k] = np.mean(np.abs(obj_obs[36:39]))  # obj_pos - tar_pos
            demo_pos_error[t, k] = np.mean(np.abs(Test_data[k][t]['observation']))
            if np.linalg.norm(obj_obs[36:39]) < 0.1:
                success_count_sim[t] = 1
            if np.linalg.norm(Test_data[k][t]['observation']) < 0.1:
                success_count_RL[t] = 1
        if sum(success_count_sim) > success_threshold:
            success_list_sim.append(1)
        if sum(success_count_RL) > success_threshold:
            success_list_demo.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f\n" % (len(success_list_sim) / len(Test_data))
    print("Success rate (RL) = %f" % (len(success_list_demo) / len(Test_data)))
    if error_type == 'demo':
        return [hand_pos_error_handJoint, hand_pos_PID_error_handJoint, hand_pos_error_base, hand_pos_PID_error_base, Computed_torque_values, Demo_torque_values, success_rate]
    else:
        return [obj_pos_error, obj_pos_error_koopman, demo_pos_error, success_rate]     

def NGF_policy_control_unseenTest(env_name, controller, NGF_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, Visualize):
    print("Testing the learned NGF dynamcis!")
    print("Begin to compute the simulation errors!")
    e = GymEnv(env_name)
    horizon = 100
    obj_ori_error = np.zeros([horizon - 1, len(Test_data)])
    obj_ori_error_koopman = np.zeros([horizon - 1, len(Test_data)])
    Computed_joint_values = np.zeros([num_hand, horizon, len(Test_data)])
    e.reset()
    init_state_dict = dict()
    success_threshold = 10
    success_list_sim = []
    success_rate = str()
    for k in tqdm(range(len(Test_data))):
        success_count_sim = np.zeros(horizon - 1)
        num_handpos = len(Test_data[k]['handpos'])
        if Velocity:
            hand_OriState = np.append(Test_data[k]['handpos'], Test_data[k]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k]['handpos']
        obj_OriState = np.append(Test_data[k]['objpos'], np.append(Test_data[k]['objorient'], Test_data[k]['objvel']))
        init_state_dict['qpos'] = Test_data[k]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k]['init']['qvel']
        init_state_dict['obj_pos'] = Test_data[k]['init']['obj_pos']
        init_state_dict['target_pos'] = Test_data[k]['init']['target_pos']
        Computed_joint_values[:, 0, k] = Test_data[k]['init']['qpos'][:num_hand]
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
            Computed_joint_values[:, t + 1, k] = hand_pos
            hand_pos_desired = hand_pos
            current = e.get_env_state()['hand_qpos'] # current state
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()   
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation)
            if Visualize:
                e.env.mj_render()
            # calculated state values using Koopman rollouts (in simulator)      
            obj_obs = e.get_obs()
            # compute the errors
            obj_ori_error[t, k] = np.mean(np.abs(obj_obs[36:39]))
            if np.linalg.norm(obj_obs[36:39]) < 0.1:
                success_count_sim[t] = 1
        if sum(success_count_sim) > success_threshold:
            success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))
    return obj_ori_error, obj_ori_error_koopman, Computed_joint_values, success_rate