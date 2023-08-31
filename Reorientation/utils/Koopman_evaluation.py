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
    hand_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_vel_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_ori_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    demo_ori_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_ori_error_koopman = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    Computed_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    Demo_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    Computed_joint_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    Joint_NN_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    success_threshold = 10
    success_list_sim = []
    success_list_koopman = []
    success_list_RL = []
    fall_list_sim = []
    fall_list_koopman = []
    fall_list_RL = []
    success_rate = str()
    # e.set_env_state(path['init_state_dict'])
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
        # init_state_dict['qpos'] = np.append(Test_data[k][0]['handpos'] + Test_data[k][0]['final_handpos'], np.zeros(6))
        init_state_dict['qpos'] = np.append(Test_data[k][0]['handpos'], np.zeros(6))
        # For hand control test, we can set the initial pos of pen to be (0.15, 0, 0, 0, 0, 0), so that there is no contact.
        # init_state_dict['qpos'][num_handpos] = 0.15  
        init_state_dict['qvel'] = np.append(Test_data[k][0]['handvel'], np.zeros(6))
        Computed_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        init_state_dict['desired_orien'] = euler2quat(Test_data[k][0]['pen_desired_orien'])
        e.set_env_state(init_state_dict)
        Joint_NN_values[:, 0, k] = e.get_obs()[:24]
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
                    obj_ori_world = ori_transform_inverse(x_t_1_computed[27:30], e.get_obs()[36:39]) # [27:30] obj_orientation
                    obj_pos_world = x_t_1_computed[24:27]
                else:
                    obj_ori_world = ori_transform_inverse(x_t_1_computed[51:54], e.get_obs()[36:39]) 
                    obj_pos_world = x_t_1_computed[48:51]
            else:
                x_t_1_computed = koopman_object.z_inverse(torch.from_numpy(z_t_1_computed)).numpy()
                if not Velocity:
                    obj_ori_world = ori_transform_inverse(x_t_1_computed[27:30], e.get_obs()[36:39])
                    obj_pos_world = x_t_1_computed[24:27]
                else:
                    obj_ori_world = ori_transform_inverse(x_t_1_computed[51:54], e.get_obs()[36:39]) 
                    obj_pos_world = x_t_1_computed[48:51]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint state
            Computed_joint_values[:, t + 1, k] = hand_pos
            # hand_pos_desired = hand_pos + Test_data[k][0]['final_handpos']
            hand_pos_desired = hand_pos
            z_t = z_t_1_computed
            current = e.get_env_state()['qpos'][:num_handpos] # current state
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()   
            Computed_torque_values[:, t, k] = NN_output
            Demo_torque_values[:, t, k] = Test_data[k][t]['action']
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation
            if Visualize:
                e.env.mj_render()
            # ground-truth state values (obtained from RL in simulator)
            # gt_hand_pos = Test_data[k][t + 1]['handpos'] + Test_data[k][0]['final_handpos']
            gt_hand_pos = Test_data[k][t + 1]['handpos']
            gt_obj_pos = Test_data[k][t + 1]['objpos']
            gt_obj_orient = Test_data[k][t + 1]['objorient']  # in demo data, it is defined in a new coordinate 
            gt_obj_orien_world_frame = ori_transform_inverse(gt_obj_orient, e.get_obs()[36:39])
            gt_obj_vel = Test_data[k][t + 1]['objvel']
            # calculated state values using Koopman rollouts (in simulator)
            hand_pos = e.get_obs()[:24]
            Joint_NN_values[:, t + 1, k] = hand_pos
            obj_pos = e.get_obs()[24:27]
            # Test_data[k][*]['desired_ori'] is the same value
            obj_orient = ori_transform(e.get_obs()[33:36], Test_data[k][t]['desired_ori'])  # change the coordinate
            obj_vel = e.get_obs()[27:33]          
            obj_obs = e.get_obs()
            # compute the errors
            hand_pos_error[t, k] = np.mean(np.abs(gt_hand_pos - hand_pos))  # probably not return the mean value?
            hand_pos_PID_error[t, k] = np.mean(np.abs(hand_pos_desired - hand_pos))  
            obj_pos_error[t, k] = np.mean(np.abs(gt_obj_pos - obj_pos) + np.abs(gt_obj_orient - obj_orient))
            obj_vel_error[t, k] = np.mean(np.abs(gt_obj_vel - obj_vel))
            obj_ori_error[t, k] = np.mean(np.abs(obj_obs[42:45]))  # obj_orien-desired_orien (goal error)
            obj_ori_error_koopman[t, k] = np.mean(np.abs(obj_ori_world - obj_obs[36:39]))  # goal error in koopman
            demo_ori_error[t, k] = np.mean(np.abs(Test_data[k][t]['observation']))
            # compute the success rate
            orien_similarity_sim = np.dot(obj_obs[33:36], obj_obs[36:39])
            orien_similarity_koopman = np.dot(obj_ori_world, obj_obs[36:39])
            orien_similarity_RL = np.dot(gt_obj_orien_world_frame, obj_obs[36:39])
            dist = np.linalg.norm(obj_obs[39:42])
            desired_pos = -obj_obs[39:42]+obj_obs[24:27]
            dist_koopman = np.linalg.norm(obj_pos_world - desired_pos)
            dist_RL = np.linalg.norm(gt_obj_pos - desired_pos)
            success_count_sim[t] = 1 if (orien_similarity_sim > 0.90) else 0
            success_count_koopman[t] = 1 if (orien_similarity_koopman > 0.90) else 0
            success_count_RL[t] = 1 if (orien_similarity_RL > 0.90) else 0
        if np.abs(obj_obs[39:42])[2] > 0.15:
            fall_list_sim.append(1)
        else:
            if sum(success_count_sim) > success_threshold and np.mean(np.abs(obj_vel)) < 1.: 
                success_list_sim.append(1)
        if np.abs(obj_pos_world - desired_pos)[2] > 0.15:
            fall_list_koopman.append(1)
        else:
            if sum(success_count_koopman) > success_threshold:
                success_list_koopman.append(1)
        if np.abs(gt_obj_pos - desired_pos)[2] > 0.15:
            fall_list_RL.append(1)
        else:
            if sum(success_count_RL) > success_threshold:
                success_list_RL.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f\n" % (len(success_list_sim) / len(Test_data))
    print("Success rate (koopman) = %f" % (len(success_list_koopman) / len(Test_data)))
    success_rate += "Success rate (koopman) = %f\n" % (len(success_list_koopman) / len(Test_data))
    print("Success rate (RL) = %f" % (len(success_list_RL) / len(Test_data)))
    print("Throw out rate (sim) = %f" % (len(fall_list_sim) / len(Test_data)))
    success_rate += "Throw out rate (sim) = %f\n" % (len(fall_list_sim) / len(Test_data))
    print("Throw out rate (koopman) = %f" % (len(fall_list_koopman) / len(Test_data)))
    success_rate += "Throw out rate (koopman) = %f\n" % (len(fall_list_koopman) / len(Test_data))
    print("Throw out rate (RL) = %f" % (len(fall_list_RL) / len(Test_data)))
    if error_type == 'demo':
        return [hand_pos_error, hand_pos_PID_error, Computed_torque_values, Demo_torque_values, Computed_joint_values, Joint_NN_values, success_rate]
    else:
        return [obj_ori_error, obj_ori_error_koopman, demo_ori_error, success_rate]     

def koopman_policy_control_unseenTest(env_name, controller, koopman_object, koopman_matrix, Test_data, Velocity, num_hand, num_obj, koopmanoption, Visualize):
    print("Testing the learned koopman dynamcis!")
    print("Begin to compute the simulation errors!")
    e = GymEnv(env_name)
    obj_ori_error = np.zeros([e.horizon - 1, len(Test_data)])
    obj_ori_error_koopman = np.zeros([e.horizon - 1, len(Test_data)])
    e.reset()
    init_state_dict = dict()
    success_threshold = 10
    success_list_sim = []
    success_list_koopman = []
    fall_list_sim = []
    fall_list_koopman = []
    success_rate = str()
    # e.set_env_state(path['init_state_dict'])
    for k in tqdm(range(len(Test_data))):
        success_count_sim = np.zeros(e.horizon - 1)
        success_count_koopman = np.zeros(e.horizon - 1)
        num_handpos = len(Test_data[k]['handpos'])
        if Velocity:
            hand_OriState = np.append(Test_data[k]['handpos'], Test_data[k]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k]['handpos']
        obj_OriState = np.append(Test_data[k]['objpos'], np.append(Test_data[k]['objorient'], Test_data[k]['objvel']))
        init_state_dict['qpos'] = np.append(Test_data[k]['handpos'], np.zeros(6))
        # For hand control test, we can set the initial pos of pen to be (0.15, 0, 0, 0, 0, 0), so that there is no contact.
        # init_state_dict['qpos'][num_handpos] = 0.15  
        init_state_dict['qvel'] = np.append(Test_data[k]['handvel'], np.zeros(6))
        init_state_dict['desired_orien'] = euler2quat(Test_data[k]['pen_desired_orien'])
        e.set_env_state(init_state_dict)
        if koopmanoption == 'Drafted': 
            z_t = koopman_object.z(hand_OriState, obj_OriState)  # initial states in lifted space
        else:
            input_state = np.append(hand_OriState, obj_OriState)
            input_state = torch.from_numpy(input_state)
            z_t = koopman_object.z(input_state).numpy()  # lift the initial states and then roll it out using Koopman operator
        for t in range(e.horizon - 1):  # this loop is for system evolution, open loop control, no feedback
            z_t_1_computed = np.dot(koopman_matrix, z_t)
            if koopmanoption == 'Drafted':  
                x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])
                if not Velocity:
                    obj_ori_world = ori_transform_inverse(x_t_1_computed[27:30], e.get_obs()[36:39]) # [27:30] obj_orientation
                    obj_pos_world = x_t_1_computed[24:27]
                else:
                    obj_ori_world = ori_transform_inverse(x_t_1_computed[51:54], e.get_obs()[36:39]) 
                    obj_pos_world = x_t_1_computed[48:51]
            else:
                x_t_1_computed = koopman_object.z_inverse(torch.from_numpy(z_t_1_computed)).numpy()
                if not Velocity:
                    obj_ori_world = ori_transform_inverse(x_t_1_computed[27:30], e.get_obs()[36:39]) # [27:30] obj_orientation
                    obj_pos_world = x_t_1_computed[24:27]
                else:
                    obj_ori_world = ori_transform_inverse(x_t_1_computed[51:54], e.get_obs()[36:39]) 
                    obj_pos_world = x_t_1_computed[48:51]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint states
            z_t = z_t_1_computed
            current = e.get_env_state()['qpos'][:num_handpos] # current state
            set_goal = hand_pos.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()   
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation
            if Visualize:
                e.env.mj_render()
            # calculated state values using Koopman rollouts (in simulator)      
            obj_vel = e.get_obs()[27:33]
            obj_obs = e.get_obs()
            # compute the errors
            obj_ori_error[t, k] = np.mean(np.abs(obj_obs[42:45]))  # obj_orien-desired_orien (goal error)
            obj_ori_error_koopman[t, k] = np.mean(np.abs(obj_ori_world - obj_obs[36:39]))  # goal error in koopman
            # compute the success rate
            orien_similarity_sim = np.dot(obj_obs[33:36], obj_obs[36:39])
            orien_similarity_koopman = np.dot(obj_ori_world, obj_obs[36:39])
            dist = np.linalg.norm(obj_obs[39:42])
            desired_pos = -obj_obs[39:42]+obj_obs[24:27]
            dist_koopman = np.linalg.norm(obj_pos_world - desired_pos)
            success_count_sim[t] = 1 if (orien_similarity_sim > 0.90) else 0
            success_count_koopman[t] = 1 if (orien_similarity_koopman > 0.90) else 0
        if np.abs(obj_obs[39:42])[2] > 0.15:
            fall_list_sim.append(1)
        else:
            if sum(success_count_sim) > success_threshold and np.mean(np.abs(obj_vel)) < 1.:
                success_list_sim.append(1)
        if np.abs(obj_pos_world - desired_pos)[2] > 0.15:
            fall_list_koopman.append(1)
        else:
            if sum(success_count_koopman) > success_threshold:
                success_list_koopman.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))
    print("Success rate (koopman) = %f" % (len(success_list_koopman) / len(Test_data)))
    print("Throw out rate (sim) = %f" % (len(fall_list_sim) / len(Test_data)))
    print("Throw out rate (koopman) = %f" % (len(fall_list_koopman) / len(Test_data)))
    return obj_ori_error, obj_ori_error_koopman, success_rate
    
def koopman_error_visualization(env_name, koopman_object, koopman_matrix, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize):
    print("Begin to compute the koopman rollout errors!")
    hand_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
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
        state_dict['desired_orien'] = Test_data[k][0]['pen_desired_orien']
        if Velocity:
            hand_OriState = np.append(Test_data[k][0]['handpos'], Test_data[k][0]['handvel'])  # at initial states
            state_dict['qvel'] = np.append(Test_data[k][0]['handvel'], np.zeros(6))
        else:
            hand_OriState = Test_data[k][0]['handpos']
            state_dict['qvel'] = np.append(np.zeros(num_hand), np.zeros(6))
        Demo_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        Computed_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        obj_OriState = np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objorient'], Test_data[k][0]['objvel']))
        state_dict['qpos'] = np.append(Test_data[k][0]['handpos'], np.zeros(6))
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
            # for object orientation, we should covert it back to the original coordinate
            Computed_joint_values[:, t + 1, k] = hand_pos
            Demo_joint_values[:, t + 1, k] = gt_hand_pos
            state_dict['qpos'] = np.append(hand_pos, np.zeros(6))
            state_dict['qvel'] = np.append(hand_vel, np.zeros(6))
            # compute the errors
            hand_pos_error[t, k] = np.mean(np.abs(gt_hand_pos - hand_pos))  # probably not return the mean value?
            obj_pos_error[t, k] = np.mean(np.abs(gt_obj_pos - obj_pos) + np.abs(gt_obj_orient - obj_orient))
            obj_vel_error[t, k] = np.mean(np.abs(gt_obj_vel - obj_vel))
            obj_obs = e.get_obs()
            obj_ori_error[t, k] = np.mean(np.abs(obj_obs[42:45])) # obj_orien-desired_orien
            if Visualize:
                e.env.mj_render()
    if error_type == 'demo':
        return [hand_pos_error, Computed_joint_values, Demo_joint_values]

def BC_error_visualization(env_name, BC_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize):
    print("Begin to compute the BC rollout errors!")
    hand_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
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
        state_dict['desired_orien'] = Test_data[k][0]['pen_desired_orien']
        if Velocity:
            hand_OriState = np.append(Test_data[k][0]['handpos'], Test_data[k][0]['handvel'])  # at initial states
            state_dict['qvel'] = np.append(Test_data[k][0]['handvel'], np.zeros(6))
        else:
            hand_OriState = Test_data[k][0]['handpos']
            state_dict['qvel'] = np.append(np.zeros(num_hand), np.zeros(6))
        Demo_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        Computed_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        obj_OriState = np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objorient'], Test_data[k][0]['objvel']))
        state_dict['qpos'] = np.append(Test_data[k][0]['handpos'], np.zeros(6))
        if koopmanoption == 'Drafted': 
            tmp_input = np.append(hand_OriState, obj_OriState)
            z_t = torch.from_numpy(tmp_input)
        for t in range(len(Test_data[k]) - 1):
            e.KoopmanVisualize(seed = None, state_dict = state_dict) # visualize the demo data without any actions (more like a visualization)
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
            # for object orientation, we should covert it back to the original coordinate
            Computed_joint_values[:, t + 1, k] = hand_pos
            Demo_joint_values[:, t + 1, k] = gt_hand_pos
            state_dict['qpos'] = np.append(hand_pos, np.zeros(6))
            state_dict['qvel'] = np.append(hand_vel, np.zeros(6))
            # compute the errors
            hand_pos_error[t, k] = np.mean(np.abs(gt_hand_pos - hand_pos))  # probably not return the mean value?
            obj_pos_error[t, k] = np.mean(np.abs(gt_obj_pos - obj_pos) + np.abs(gt_obj_orient - obj_orient))
            obj_vel_error[t, k] = np.mean(np.abs(gt_obj_vel - obj_vel))
            obj_obs = e.get_obs()
            obj_ori_error[t, k] = np.mean(np.abs(obj_obs[42:45])) # obj_orien-desired_orien
            if Visualize:
                e.env.mj_render()
    if error_type == 'demo':
        return [hand_pos_error, Computed_joint_values, Demo_joint_values]
    # else:
    #     return [obj_ori_error] # this is incorrect and should not be used, because state_dict['qpos'] is not correct(it requires the orien in robot frame, but it actually is defined in world frame)
    #     # no goal error for this visualization function

def BC_policy_control(env_name, controller, BC_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize):
    print("Begin to compute the simulation errors!")
    hand_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_vel_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_ori_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    demo_ori_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_ori_error_koopman = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    Computed_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    Demo_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    Computed_joint_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    Joint_NN_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    success_threshold = 10
    success_list_sim = []
    success_list_koopman = []
    success_list_RL = []
    fall_list_sim = []
    fall_list_koopman = []
    fall_list_RL = []
    success_rate = str()
    # e.set_env_state(path['init_state_dict'])
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
        # init_state_dict['qpos'] = np.append(Test_data[k][0]['handpos'] + Test_data[k][0]['final_handpos'], np.zeros(6))
        init_state_dict['qpos'] = np.append(Test_data[k][0]['handpos'], np.zeros(6))
        # For hand control test, we can set the initial pos of pen to be (0.15, 0, 0, 0, 0, 0), so that there is no contact.
        # init_state_dict['qpos'][num_handpos] = 0.15  
        init_state_dict['qvel'] = np.append(Test_data[k][0]['handvel'], np.zeros(6))
        Computed_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        init_state_dict['desired_orien'] = euler2quat(Test_data[k][0]['pen_desired_orien'])
        e.set_env_state(init_state_dict)
        Joint_NN_values[:, 0, k] = e.get_obs()[:24]
        if koopmanoption == 'Drafted': 
            tmp_input = np.append(hand_OriState, obj_OriState)
            z_t = torch.from_numpy(tmp_input)
        for t in range(len(Test_data[k]) - 1):  # this loop is for system evolution, open loop control, no feedback
            z_t_1_computed = BC_agent(z_t)
            if koopmanoption == 'Drafted':  
                x_t_1_computed = z_t_1_computed.detach().numpy()
                if not Velocity:
                    obj_ori_world = ori_transform_inverse(x_t_1_computed[27:30], e.get_obs()[36:39]) # [27:30] obj_orientation
                    obj_pos_world = x_t_1_computed[24:27]
                else:
                    obj_ori_world = ori_transform_inverse(x_t_1_computed[51:54], e.get_obs()[36:39]) 
                    obj_pos_world = x_t_1_computed[48:51]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint state
            Computed_joint_values[:, t + 1, k] = hand_pos
            # hand_pos_desired = hand_pos + Test_data[k][0]['final_handpos']
            hand_pos_desired = hand_pos
            z_t = z_t_1_computed
            current = e.get_env_state()['qpos'][:num_handpos] # current state
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()   
            Computed_torque_values[:, t, k] = NN_output
            Demo_torque_values[:, t, k] = Test_data[k][t]['action']
            NN_output[np.isnan(NN_output)] = 1
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation
            if Visualize:
                e.env.mj_render()
            # ground-truth state values (obtained from RL in simulator)
            # gt_hand_pos = Test_data[k][t + 1]['handpos'] + Test_data[k][0]['final_handpos']
            gt_hand_pos = Test_data[k][t + 1]['handpos']
            gt_obj_pos = Test_data[k][t + 1]['objpos']
            gt_obj_orient = Test_data[k][t + 1]['objorient']  # in demo data, it is defined in a new coordinate 
            gt_obj_orien_world_frame = ori_transform_inverse(gt_obj_orient, e.get_obs()[36:39])
            gt_obj_vel = Test_data[k][t + 1]['objvel']
            # calculated state values using Koopman rollouts (in simulator)
            hand_pos = e.get_obs()[:24]
            Joint_NN_values[:, t + 1, k] = hand_pos
            obj_pos = e.get_obs()[24:27]
            # Test_data[k][*]['desired_ori'] is the same value
            obj_orient = ori_transform(e.get_obs()[33:36], Test_data[k][t]['desired_ori'])  # change the coordinate
            obj_vel = e.get_obs()[27:33]          
            obj_obs = e.get_obs()
            # compute the errors
            hand_pos_error[t, k] = np.mean(np.abs(gt_hand_pos - hand_pos))  # probably not return the mean value?
            hand_pos_PID_error[t, k] = np.mean(np.abs(hand_pos_desired - hand_pos))  
            obj_pos_error[t, k] = np.mean(np.abs(gt_obj_pos - obj_pos) + np.abs(gt_obj_orient - obj_orient))
            obj_vel_error[t, k] = np.mean(np.abs(gt_obj_vel - obj_vel))
            obj_ori_error[t, k] = np.mean(np.abs(obj_obs[42:45]))  # obj_orien-desired_orien (goal error)
            obj_ori_error_koopman[t, k] = np.mean(np.abs(obj_ori_world - obj_obs[36:39]))  # goal error in koopman
            demo_ori_error[t, k] = np.mean(np.abs(Test_data[k][t]['observation']))
            # compute the success rate
            orien_similarity_sim = np.dot(obj_obs[33:36], obj_obs[36:39])
            orien_similarity_koopman = np.dot(obj_ori_world, obj_obs[36:39])
            orien_similarity_RL = np.dot(gt_obj_orien_world_frame, obj_obs[36:39])
            dist = np.linalg.norm(obj_obs[39:42])
            desired_pos = -obj_obs[39:42]+obj_obs[24:27]
            dist_koopman = np.linalg.norm(obj_pos_world - desired_pos)
            dist_RL = np.linalg.norm(gt_obj_pos - desired_pos)
            success_count_sim[t] = 1 if (orien_similarity_sim > 0.90) else 0
            success_count_koopman[t] = 1 if (orien_similarity_koopman > 0.90) else 0
            success_count_RL[t] = 1 if (orien_similarity_RL > 0.90) else 0
        if np.abs(obj_obs[39:42])[2] > 0.15:
            fall_list_sim.append(1)
        else:
            if sum(success_count_sim) > success_threshold and np.mean(np.abs(obj_vel)) < 1.:
                success_list_sim.append(1)
        if np.abs(obj_pos_world - desired_pos)[2] > 0.15:
            fall_list_koopman.append(1)
        else:
            if sum(success_count_koopman) > success_threshold:
                success_list_koopman.append(1)
        if np.abs(gt_obj_pos - desired_pos)[2] > 0.15:
            fall_list_RL.append(1)
        else:
            if sum(success_count_RL) > success_threshold:
                success_list_RL.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f\n" % (len(success_list_sim) / len(Test_data))
    print("Success rate (koopman) = %f" % (len(success_list_koopman) / len(Test_data)))
    success_rate += "Success rate (koopman) = %f\n" % (len(success_list_koopman) / len(Test_data))
    print("Success rate (RL) = %f" % (len(success_list_RL) / len(Test_data)))
    print("Throw out rate (sim) = %f" % (len(fall_list_sim) / len(Test_data)))
    success_rate += "Throw out rate (sim) = %f\n" % (len(fall_list_sim) / len(Test_data))
    print("Throw out rate (koopman) = %f" % (len(fall_list_koopman) / len(Test_data)))
    success_rate += "Throw out rate (koopman) = %f\n" % (len(fall_list_koopman) / len(Test_data))
    print("Throw out rate (RL) = %f" % (len(fall_list_RL) / len(Test_data)))
    if error_type == 'demo':
        return [hand_pos_error, hand_pos_PID_error, Computed_torque_values, Demo_torque_values, Computed_joint_values, Joint_NN_values, success_rate]
    else:
        return [obj_ori_error, obj_ori_error_koopman, demo_ori_error, success_rate]     

def BC_policy_control_unseenTest(env_name, controller, BC_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, Visualize):
    print("Testing the learned koopman dynamcis!")
    print("Begin to compute the simulation errors!")
    e = GymEnv(env_name)
    obj_ori_error = np.zeros([e.horizon - 1, len(Test_data)])
    obj_ori_error_koopman = np.zeros([e.horizon - 1, len(Test_data)])
    e.reset()
    init_state_dict = dict()
    success_threshold = 10
    success_list_sim = []
    success_list_koopman = []
    fall_list_sim = []
    fall_list_koopman = []
    success_rate = str()
    # e.set_env_state(path['init_state_dict'])
    for k in tqdm(range(len(Test_data))):
        success_count_sim = np.zeros(e.horizon - 1)
        success_count_koopman = np.zeros(e.horizon - 1)
        num_handpos = len(Test_data[k]['handpos'])
        if Velocity:
            hand_OriState = np.append(Test_data[k]['handpos'], Test_data[k]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k]['handpos']
        obj_OriState = np.append(Test_data[k]['objpos'], np.append(Test_data[k]['objorient'], Test_data[k]['objvel']))
        init_state_dict['qpos'] = np.append(Test_data[k]['handpos'], np.zeros(6))
        # For hand control test, we can set the initial pos of pen to be (0.15, 0, 0, 0, 0, 0), so that there is no contact.
        # init_state_dict['qpos'][num_handpos] = 0.15  
        init_state_dict['qvel'] = np.append(Test_data[k]['handvel'], np.zeros(6))
        init_state_dict['desired_orien'] = euler2quat(Test_data[k]['pen_desired_orien'])
        e.set_env_state(init_state_dict)
        if koopmanoption == 'Drafted': 
            tmp_input = np.append(hand_OriState, obj_OriState)
            z_t = torch.from_numpy(tmp_input)
        for t in range(e.horizon - 1):  # this loop is for system evolution, open loop control, no feedback
            z_t_1_computed = BC_agent(z_t)
            if koopmanoption == 'Drafted':  
                x_t_1_computed = z_t_1_computed.detach().numpy()
                if not Velocity:
                    obj_ori_world = ori_transform_inverse(x_t_1_computed[27:30], e.get_obs()[36:39]) # [27:30] obj_orientation
                    obj_pos_world = x_t_1_computed[24:27]
                else:
                    obj_ori_world = ori_transform_inverse(x_t_1_computed[51:54], e.get_obs()[36:39]) 
                    obj_pos_world = x_t_1_computed[48:51]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint states
            z_t = z_t_1_computed
            current = e.get_env_state()['qpos'][:num_handpos] # current state
            set_goal = hand_pos.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()  
            NN_output[np.isnan(NN_output)] = 1 
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation
            if Visualize:
                e.env.mj_render()
            # calculated state values using Koopman rollouts (in simulator)      
            obj_obs = e.get_obs()
            obj_vel = e.get_obs()[27:33]
            # compute the errors
            obj_ori_error[t, k] = np.mean(np.abs(obj_obs[42:45]))  # obj_orien-desired_orien (goal error)
            obj_ori_error_koopman[t, k] = np.mean(np.abs(obj_ori_world - obj_obs[36:39]))  # goal error in koopman
            # compute the success rate
            orien_similarity_sim = np.dot(obj_obs[33:36], obj_obs[36:39])
            orien_similarity_koopman = np.dot(obj_ori_world, obj_obs[36:39])
            dist = np.linalg.norm(obj_obs[39:42])
            desired_pos = -obj_obs[39:42]+obj_obs[24:27]
            dist_koopman = np.linalg.norm(obj_pos_world - desired_pos)
            success_count_sim[t] = 1 if (orien_similarity_sim > 0.90) else 0
            success_count_koopman[t] = 1 if (orien_similarity_koopman > 0.90) else 0
        if np.abs(obj_obs[39:42])[2] > 0.15:
            fall_list_sim.append(1)
        else:
            if sum(success_count_sim) > success_threshold and np.mean(np.abs(obj_vel)) < 1.:
                success_list_sim.append(1)
        if np.abs(obj_pos_world - desired_pos)[2] > 0.15:
            fall_list_koopman.append(1)
        else:
            if sum(success_count_koopman) > success_threshold:
                success_list_koopman.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))
    print("Success rate (koopman) = %f" % (len(success_list_koopman) / len(Test_data)))
    print("Throw out rate (sim) = %f" % (len(fall_list_sim) / len(Test_data)))
    print("Throw out rate (koopman) = %f" % (len(fall_list_koopman) / len(Test_data)))
    return obj_ori_error, obj_ori_error_koopman, success_rate

def LSTM_error_visualization(env_name, LSTM_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize):
    print("Begin to compute the LSTM rollout errors!")
    hand_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
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
        state_dict['desired_orien'] = Test_data[k][0]['pen_desired_orien']
        if Velocity:
            hand_OriState = np.append(Test_data[k][0]['handpos'], Test_data[k][0]['handvel'])  # at initial states
            state_dict['qvel'] = np.append(Test_data[k][0]['handvel'], np.zeros(6))
        else:
            hand_OriState = Test_data[k][0]['handpos']
            state_dict['qvel'] = np.append(np.zeros(num_hand), np.zeros(6))
        Demo_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        Computed_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        obj_OriState = np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objorient'], Test_data[k][0]['objvel']))
        state_dict['qpos'] = np.append(Test_data[k][0]['handpos'], np.zeros(6))
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
            # for object orientation, we should covert it back to the original coordinate
            Computed_joint_values[:, t + 1, k] = hand_pos
            Demo_joint_values[:, t + 1, k] = gt_hand_pos
            state_dict['qpos'] = np.append(hand_pos, np.zeros(6))
            state_dict['qvel'] = np.append(hand_vel, np.zeros(6))
            # compute the errors
            hand_pos_error[t, k] = np.mean(np.abs(gt_hand_pos - hand_pos))  # probably not return the mean value?
            obj_pos_error[t, k] = np.mean(np.abs(gt_obj_pos - obj_pos) + np.abs(gt_obj_orient - obj_orient))
            obj_vel_error[t, k] = np.mean(np.abs(gt_obj_vel - obj_vel))
            obj_obs = e.get_obs()
            obj_ori_error[t, k] = np.mean(np.abs(obj_obs[42:45])) # obj_orien-desired_orien
            if Visualize:
                e.env.mj_render()
    if error_type == 'demo':
        return [hand_pos_error, Computed_joint_values, Demo_joint_values]
    # else:
    #     return [obj_ori_error] # this is incorrect and should not be used, because state_dict['qpos'] is not correct(it requires the orien in robot frame, but it actually is defined in world frame)
    #     # no goal error for this visualization function

def LSTM_policy_control(env_name, controller, LSTM_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize):
    print("Begin to compute the simulation errors!")
    hand_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_vel_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_ori_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    demo_ori_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_ori_error_koopman = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    Computed_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    Demo_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    Computed_joint_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    Joint_NN_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    success_threshold = 10
    success_list_sim = []
    success_list_koopman = []
    success_list_RL = []
    fall_list_sim = []
    fall_list_koopman = []
    fall_list_RL = []
    # e.set_env_state(path['init_state_dict'])
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
        # init_state_dict['qpos'] = np.append(Test_data[k][0]['handpos'] + Test_data[k][0]['final_handpos'], np.zeros(6))
        init_state_dict['qpos'] = np.append(Test_data[k][0]['handpos'], np.zeros(6))
        # For hand control test, we can set the initial pos of pen to be (0.15, 0, 0, 0, 0, 0), so that there is no contact.
        # init_state_dict['qpos'][num_handpos] = 0.15  
        init_state_dict['qvel'] = np.append(Test_data[k][0]['handvel'], np.zeros(6))
        Computed_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        init_state_dict['desired_orien'] = euler2quat(Test_data[k][0]['pen_desired_orien'])
        e.set_env_state(init_state_dict)
        Joint_NN_values[:, 0, k] = e.get_obs()[:24]
        if koopmanoption == 'Drafted': 
            tmp_input = np.append(hand_OriState, obj_OriState)
            z_t = torch.from_numpy(tmp_input)
        hidden_state = (np.zeros((1, 1, LSTM_agent.model.rnn_cell.hidden_size)), np.zeros((1, 1, LSTM_agent.model.rnn_cell.hidden_size)))
        for t in range(len(Test_data[k]) - 1):  # this loop is for system evolution, open loop control, no feedback
            z_t_1_computed, hidden_state = LSTM_agent.get_action(z_t, hidden_state)
            if koopmanoption == 'Drafted':  
                x_t_1_computed = z_t_1_computed
                if not Velocity:
                    obj_ori_world = ori_transform_inverse(x_t_1_computed[27:30], e.get_obs()[36:39]) # [27:30] obj_orientation
                    obj_pos_world = x_t_1_computed[24:27]
                else:
                    obj_ori_world = ori_transform_inverse(x_t_1_computed[51:54], e.get_obs()[36:39]) 
                    obj_pos_world = x_t_1_computed[48:51]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint state
            Computed_joint_values[:, t + 1, k] = hand_pos
            # hand_pos_desired = hand_pos + Test_data[k][0]['final_handpos']
            hand_pos_desired = hand_pos
            z_t = z_t_1_computed
            current = e.get_env_state()['qpos'][:num_handpos] # current state
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()   
            Computed_torque_values[:, t, k] = NN_output
            Demo_torque_values[:, t, k] = Test_data[k][t]['action']
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation
            if Visualize:
                e.env.mj_render()
            # ground-truth state values (obtained from RL in simulator)
            # gt_hand_pos = Test_data[k][t + 1]['handpos'] + Test_data[k][0]['final_handpos']
            gt_hand_pos = Test_data[k][t + 1]['handpos']
            gt_obj_pos = Test_data[k][t + 1]['objpos']
            gt_obj_orient = Test_data[k][t + 1]['objorient']  # in demo data, it is defined in a new coordinate 
            gt_obj_orien_world_frame = ori_transform_inverse(gt_obj_orient, e.get_obs()[36:39])
            gt_obj_vel = Test_data[k][t + 1]['objvel']
            # calculated state values using Koopman rollouts (in simulator)
            hand_pos = e.get_obs()[:24]
            Joint_NN_values[:, t + 1, k] = hand_pos
            obj_pos = e.get_obs()[24:27]
            # Test_data[k][*]['desired_ori'] is the same value
            obj_orient = ori_transform(e.get_obs()[33:36], Test_data[k][t]['desired_ori'])  # change the coordinate
            obj_vel = e.get_obs()[27:33]          
            obj_obs = e.get_obs()
            # compute the errors
            hand_pos_error[t, k] = np.mean(np.abs(gt_hand_pos - hand_pos))  # probably not return the mean value?
            hand_pos_PID_error[t, k] = np.mean(np.abs(hand_pos_desired - hand_pos))  
            obj_pos_error[t, k] = np.mean(np.abs(gt_obj_pos - obj_pos) + np.abs(gt_obj_orient - obj_orient))
            obj_vel_error[t, k] = np.mean(np.abs(gt_obj_vel - obj_vel))
            obj_ori_error[t, k] = np.mean(np.abs(obj_obs[42:45]))  # obj_orien-desired_orien (goal error)
            obj_ori_error_koopman[t, k] = np.mean(np.abs(obj_ori_world - obj_obs[36:39]))  # goal error in koopman
            demo_ori_error[t, k] = np.mean(np.abs(Test_data[k][t]['observation']))
            # compute the success rate
            orien_similarity_sim = np.dot(obj_obs[33:36], obj_obs[36:39])
            orien_similarity_koopman = np.dot(obj_ori_world, obj_obs[36:39])
            orien_similarity_RL = np.dot(gt_obj_orien_world_frame, obj_obs[36:39])
            dist = np.linalg.norm(obj_obs[39:42])
            desired_pos = -obj_obs[39:42]+obj_obs[24:27]
            dist_koopman = np.linalg.norm(obj_pos_world - desired_pos)
            dist_RL = np.linalg.norm(gt_obj_pos - desired_pos)
            success_count_sim[t] = 1 if (orien_similarity_sim > 0.90) else 0
            success_count_koopman[t] = 1 if (orien_similarity_koopman > 0.90) else 0
            success_count_RL[t] = 1 if (orien_similarity_RL > 0.90) else 0
        if np.abs(obj_obs[39:42])[2] > 0.15:
            fall_list_sim.append(1)
        else:
            if sum(success_count_sim) > success_threshold and np.mean(np.abs(obj_vel)) < 1.:
                success_list_sim.append(1)
        if np.abs(obj_pos_world - desired_pos)[2] > 0.15:
            fall_list_koopman.append(1)
        else:
            if sum(success_count_koopman) > success_threshold:
                success_list_koopman.append(1)
        if np.abs(gt_obj_pos - desired_pos)[2] > 0.15:
            fall_list_RL.append(1)
        else:
            if sum(success_count_RL) > success_threshold:
                success_list_RL.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f\n" % (len(success_list_sim) / len(Test_data))
    print("Success rate (koopman) = %f" % (len(success_list_koopman) / len(Test_data)))
    success_rate += "Success rate (koopman) = %f\n" % (len(success_list_koopman) / len(Test_data))
    print("Success rate (RL) = %f" % (len(success_list_RL) / len(Test_data)))
    print("Throw out rate (sim) = %f" % (len(fall_list_sim) / len(Test_data)))
    success_rate += "Throw out rate (sim) = %f\n" % (len(fall_list_sim) / len(Test_data))
    print("Throw out rate (koopman) = %f" % (len(fall_list_koopman) / len(Test_data)))
    success_rate += "Throw out rate (koopman) = %f\n" % (len(fall_list_koopman) / len(Test_data))
    print("Throw out rate (RL) = %f" % (len(fall_list_RL) / len(Test_data)))
    if error_type == 'demo':
        return [hand_pos_error, hand_pos_PID_error, Computed_torque_values, Demo_torque_values, Computed_joint_values, Joint_NN_values, success_rate]
    else:
        return [obj_ori_error, obj_ori_error_koopman, demo_ori_error, success_rate]     

def LSTM_policy_control_unseenTest(env_name, controller, LSTM_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, Visualize):
    print("Testing the learned koopman dynamcis!")
    print("Begin to compute the simulation errors!")
    e = GymEnv(env_name)
    obj_ori_error = np.zeros([e.horizon - 1, len(Test_data)])
    obj_ori_error_koopman = np.zeros([e.horizon - 1, len(Test_data)])
    e.reset()
    init_state_dict = dict()
    success_threshold = 10
    success_list_sim = []
    success_list_koopman = []
    fall_list_sim = []
    fall_list_koopman = []
    success_rate = str()
    # e.set_env_state(path['init_state_dict'])
    for k in tqdm(range(len(Test_data))):
        success_count_sim = np.zeros(e.horizon - 1)
        success_count_koopman = np.zeros(e.horizon - 1)
        num_handpos = len(Test_data[k]['handpos'])
        if Velocity:
            hand_OriState = np.append(Test_data[k]['handpos'], Test_data[k]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k]['handpos']
        obj_OriState = np.append(Test_data[k]['objpos'], np.append(Test_data[k]['objorient'], Test_data[k]['objvel']))
        init_state_dict['qpos'] = np.append(Test_data[k]['handpos'], np.zeros(6))
        # For hand control test, we can set the initial pos of pen to be (0.15, 0, 0, 0, 0, 0), so that there is no contact.
        # init_state_dict['qpos'][num_handpos] = 0.15  
        init_state_dict['qvel'] = np.append(Test_data[k]['handvel'], np.zeros(6))
        init_state_dict['desired_orien'] = euler2quat(Test_data[k]['pen_desired_orien'])
        e.set_env_state(init_state_dict)
        if koopmanoption == 'Drafted': 
            tmp_input = np.append(hand_OriState, obj_OriState)
            z_t = torch.from_numpy(tmp_input)
        hidden_state = (np.zeros((1, 1, LSTM_agent.model.rnn_cell.hidden_size)), np.zeros((1, 1, LSTM_agent.model.rnn_cell.hidden_size)))
        for t in range(e.horizon - 1):  # this loop is for system evolution, open loop control, no feedback
            z_t_1_computed, hidden_state = LSTM_agent.get_action(z_t, hidden_state)
            if koopmanoption == 'Drafted':  
                x_t_1_computed = z_t_1_computed
                if not Velocity:
                    obj_ori_world = ori_transform_inverse(x_t_1_computed[27:30], e.get_obs()[36:39]) # [27:30] obj_orientation
                    obj_pos_world = x_t_1_computed[24:27]
                else:
                    obj_ori_world = ori_transform_inverse(x_t_1_computed[51:54], e.get_obs()[36:39]) 
                    obj_pos_world = x_t_1_computed[48:51]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint states
            z_t = z_t_1_computed
            current = e.get_env_state()['qpos'][:num_handpos] # current state
            set_goal = hand_pos.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()   
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation
            if Visualize:
                e.env.mj_render()
            # calculated state values using Koopman rollouts (in simulator)      
            obj_obs = e.get_obs()
            obj_vel = e.get_obs()[27:33]
            # compute the errors
            obj_ori_error[t, k] = np.mean(np.abs(obj_obs[42:45]))  # obj_orien-desired_orien (goal error)
            obj_ori_error_koopman[t, k] = np.mean(np.abs(obj_ori_world - obj_obs[36:39]))  # goal error in koopman
            # compute the success rate
            orien_similarity_sim = np.dot(obj_obs[33:36], obj_obs[36:39])
            orien_similarity_koopman = np.dot(obj_ori_world, obj_obs[36:39])
            dist = np.linalg.norm(obj_obs[39:42])
            desired_pos = -obj_obs[39:42]+obj_obs[24:27]
            dist_koopman = np.linalg.norm(obj_pos_world - desired_pos)
            success_count_sim[t] = 1 if (orien_similarity_sim > 0.90) else 0
            success_count_koopman[t] = 1 if (orien_similarity_koopman > 0.90) else 0
        if np.abs(obj_obs[39:42])[2] > 0.15:
            fall_list_sim.append(1)
        else:
            if sum(success_count_sim) > success_threshold and np.mean(np.abs(obj_vel)) < 1.:
                success_list_sim.append(1)
        if np.abs(obj_pos_world - desired_pos)[2] > 0.15:
            fall_list_koopman.append(1)
        else:
            if sum(success_count_koopman) > success_threshold:
                success_list_koopman.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))
    print("Success rate (koopman) = %f" % (len(success_list_koopman) / len(Test_data)))
    print("Throw out rate (sim) = %f" % (len(fall_list_sim) / len(Test_data)))
    print("Throw out rate (koopman) = %f" % (len(fall_list_koopman) / len(Test_data)))
    return obj_ori_error, obj_ori_error_koopman, success_rate

def NDP_error_visualization(env_name, NDP_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize):
    print("Begin to compute the NDP rollout errors!")
    hand_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
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
        state_dict['desired_orien'] = Test_data[k][0]['pen_desired_orien']
        if Velocity:
            hand_OriState = np.append(Test_data[k][0]['handpos'], Test_data[k][0]['handvel'])  # at initial states
            state_dict['qvel'] = np.append(Test_data[k][0]['handvel'], np.zeros(6))
        else:
            hand_OriState = Test_data[k][0]['handpos']
            state_dict['qvel'] = np.append(np.zeros(num_hand), np.zeros(6))
        Demo_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        Computed_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        obj_OriState = np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objorient'], Test_data[k][0]['objvel']))
        state_dict['qpos'] = np.append(Test_data[k][0]['handpos'], np.zeros(6))
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
            # for object orientation, we should covert it back to the original coordinate
            Computed_joint_values[:, t + 1, k] = hand_pos
            Demo_joint_values[:, t + 1, k] = gt_hand_pos
            state_dict['qpos'] = np.append(hand_pos, np.zeros(6))
            state_dict['qvel'] = np.append(hand_vel, np.zeros(6))
            # compute the errors
            hand_pos_error[t, k] = np.mean(np.abs(gt_hand_pos - hand_pos))  # probably not return the mean value?
            obj_pos_error[t, k] = np.mean(np.abs(gt_obj_pos - obj_pos) + np.abs(gt_obj_orient - obj_orient))
            obj_vel_error[t, k] = np.mean(np.abs(gt_obj_vel - obj_vel))
            obj_obs = e.get_obs()
            obj_ori_error[t, k] = np.mean(np.abs(obj_obs[42:45])) # obj_orien-desired_orien
            if Visualize:
                e.env.mj_render()
    if error_type == 'demo':
        return [hand_pos_error, Computed_joint_values, Demo_joint_values]
    # else:
    #     return [obj_ori_error] # this is incorrect and should not be used, because state_dict['qpos'] is not correct(it requires the orien in robot frame, but it actually is defined in world frame)
    #     # no goal error for this visualization function

def NDP_policy_control(env_name, controller, NDP_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize):
    print("Begin to compute the simulation errors!")
    hand_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_vel_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_ori_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    demo_ori_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_ori_error_koopman = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    Computed_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    Demo_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    Computed_joint_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    Joint_NN_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    success_threshold = 10
    success_list_sim = []
    success_list_koopman = []
    success_list_RL = []
    fall_list_sim = []
    fall_list_koopman = []
    fall_list_RL = []
    success_rate = str()
    # e.set_env_state(path['init_state_dict'])
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
        # init_state_dict['qpos'] = np.append(Test_data[k][0]['handpos'] + Test_data[k][0]['final_handpos'], np.zeros(6))
        init_state_dict['qpos'] = np.append(Test_data[k][0]['handpos'], np.zeros(6))
        # For hand control test, we can set the initial pos of pen to be (0.15, 0, 0, 0, 0, 0), so that there is no contact.
        # init_state_dict['qpos'][num_handpos] = 0.15  
        init_state_dict['qvel'] = np.append(Test_data[k][0]['handvel'], np.zeros(6))
        Computed_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        init_state_dict['desired_orien'] = euler2quat(Test_data[k][0]['pen_desired_orien'])
        e.set_env_state(init_state_dict)
        Joint_NN_values[:, 0, k] = e.get_obs()[:24]
        if koopmanoption == 'Drafted': 
            tmp_input = np.append(hand_OriState, obj_OriState)
            z_t = torch.from_numpy(tmp_input)
        traj = NDP_agent.execute(z_t, z_t) # traj -> [T, dim]
        for t in range(len(Test_data[k]) - 1):  # this loop is for system evolution, open loop control, no feedback
            if koopmanoption == 'Drafted':  
                x_t_1_computed = traj[t + 1]
                if not Velocity:
                    obj_ori_world = ori_transform_inverse(x_t_1_computed[27:30], e.get_obs()[36:39]) # [27:30] obj_orientation
                    obj_pos_world = x_t_1_computed[24:27]
                else:
                    obj_ori_world = ori_transform_inverse(x_t_1_computed[51:54], e.get_obs()[36:39]) 
                    obj_pos_world = x_t_1_computed[48:51]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint state
            Computed_joint_values[:, t + 1, k] = hand_pos
            # hand_pos_desired = hand_pos + Test_data[k][0]['final_handpos']
            hand_pos_desired = hand_pos
            current = e.get_env_state()['qpos'][:num_handpos] # current state
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()   
            Computed_torque_values[:, t, k] = NN_output
            Demo_torque_values[:, t, k] = Test_data[k][t]['action']
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation
            if Visualize:
                e.env.mj_render()
            # ground-truth state values (obtained from RL in simulator)
            # gt_hand_pos = Test_data[k][t + 1]['handpos'] + Test_data[k][0]['final_handpos']
            gt_hand_pos = Test_data[k][t + 1]['handpos']
            gt_obj_pos = Test_data[k][t + 1]['objpos']
            gt_obj_orient = Test_data[k][t + 1]['objorient']  # in demo data, it is defined in a new coordinate 
            gt_obj_orien_world_frame = ori_transform_inverse(gt_obj_orient, e.get_obs()[36:39])
            gt_obj_vel = Test_data[k][t + 1]['objvel']
            # calculated state values using Koopman rollouts (in simulator)
            hand_pos = e.get_obs()[:24]
            Joint_NN_values[:, t + 1, k] = hand_pos
            obj_pos = e.get_obs()[24:27]
            # Test_data[k][*]['desired_ori'] is the same value
            obj_orient = ori_transform(e.get_obs()[33:36], Test_data[k][t]['desired_ori'])  # change the coordinate
            obj_vel = e.get_obs()[27:33]          
            obj_obs = e.get_obs()
            # compute the errors
            hand_pos_error[t, k] = np.mean(np.abs(gt_hand_pos - hand_pos))  # probably not return the mean value?
            hand_pos_PID_error[t, k] = np.mean(np.abs(hand_pos_desired - hand_pos))  
            obj_pos_error[t, k] = np.mean(np.abs(gt_obj_pos - obj_pos) + np.abs(gt_obj_orient - obj_orient))
            obj_vel_error[t, k] = np.mean(np.abs(gt_obj_vel - obj_vel))
            obj_ori_error[t, k] = np.mean(np.abs(obj_obs[42:45]))  # obj_orien-desired_orien (goal error)
            obj_ori_error_koopman[t, k] = np.mean(np.abs(obj_ori_world - obj_obs[36:39]))  # goal error in koopman
            demo_ori_error[t, k] = np.mean(np.abs(Test_data[k][t]['observation']))
            # compute the success rate
            orien_similarity_sim = np.dot(obj_obs[33:36], obj_obs[36:39])
            orien_similarity_koopman = np.dot(obj_ori_world, obj_obs[36:39])
            orien_similarity_RL = np.dot(gt_obj_orien_world_frame, obj_obs[36:39])
            dist = np.linalg.norm(obj_obs[39:42])
            desired_pos = -obj_obs[39:42]+obj_obs[24:27]
            dist_koopman = np.linalg.norm(obj_pos_world - desired_pos)
            dist_RL = np.linalg.norm(gt_obj_pos - desired_pos)
            success_count_sim[t] = 1 if (orien_similarity_sim > 0.90) else 0
            success_count_koopman[t] = 1 if (orien_similarity_koopman > 0.90) else 0
            success_count_RL[t] = 1 if (orien_similarity_RL > 0.90) else 0
        if np.abs(obj_obs[39:42])[2] > 0.15:
            fall_list_sim.append(1)
        else:
            if sum(success_count_sim) > success_threshold and np.mean(np.abs(obj_vel)) < 1.:
                success_list_sim.append(1)
        if np.abs(obj_pos_world - desired_pos)[2] > 0.15:
            fall_list_koopman.append(1)
        else:
            if sum(success_count_koopman) > success_threshold:
                success_list_koopman.append(1)
        if np.abs(gt_obj_pos - desired_pos)[2] > 0.15:
            fall_list_RL.append(1)
        else:
            if sum(success_count_RL) > success_threshold:
                success_list_RL.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f\n" % (len(success_list_sim) / len(Test_data))
    print("Success rate (koopman) = %f" % (len(success_list_koopman) / len(Test_data)))
    success_rate += "Success rate (koopman) = %f\n" % (len(success_list_koopman) / len(Test_data))
    print("Success rate (RL) = %f" % (len(success_list_RL) / len(Test_data)))
    print("Throw out rate (sim) = %f" % (len(fall_list_sim) / len(Test_data)))
    success_rate += "Throw out rate (sim) = %f\n" % (len(fall_list_sim) / len(Test_data))
    print("Throw out rate (koopman) = %f" % (len(fall_list_koopman) / len(Test_data)))
    success_rate += "Throw out rate (koopman) = %f\n" % (len(fall_list_koopman) / len(Test_data))
    print("Throw out rate (RL) = %f" % (len(fall_list_RL) / len(Test_data)))
    if error_type == 'demo':
        return [hand_pos_error, hand_pos_PID_error, Computed_torque_values, Demo_torque_values, Computed_joint_values, Joint_NN_values, success_rate]
    else:
        return [obj_ori_error, obj_ori_error_koopman, demo_ori_error, success_rate]     

def NDP_policy_control_unseenTest(env_name, controller, NDP_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, Visualize):
    print("Testing the learned koopman dynamcis!")
    print("Begin to compute the simulation errors!")
    e = GymEnv(env_name)
    obj_ori_error = np.zeros([e.horizon - 1, len(Test_data)])
    obj_ori_error_koopman = np.zeros([e.horizon - 1, len(Test_data)])
    e.reset()
    init_state_dict = dict()
    success_threshold = 10
    success_list_sim = []
    success_list_koopman = []
    fall_list_sim = []
    fall_list_koopman = []
    success_rate = str()
    # e.set_env_state(path['init_state_dict'])
    for k in tqdm(range(len(Test_data))):
        success_count_sim = np.zeros(e.horizon - 1)
        success_count_koopman = np.zeros(e.horizon - 1)
        num_handpos = len(Test_data[k]['handpos'])
        if Velocity:
            hand_OriState = np.append(Test_data[k]['handpos'], Test_data[k]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k]['handpos']
        obj_OriState = np.append(Test_data[k]['objpos'], np.append(Test_data[k]['objorient'], Test_data[k]['objvel']))
        init_state_dict['qpos'] = np.append(Test_data[k]['handpos'], np.zeros(6))
        # For hand control test, we can set the initial pos of pen to be (0.15, 0, 0, 0, 0, 0), so that there is no contact.
        # init_state_dict['qpos'][num_handpos] = 0.15  
        init_state_dict['qvel'] = np.append(Test_data[k]['handvel'], np.zeros(6))
        init_state_dict['desired_orien'] = euler2quat(Test_data[k]['pen_desired_orien'])
        e.set_env_state(init_state_dict)
        if koopmanoption == 'Drafted': 
            tmp_input = np.append(hand_OriState, obj_OriState)
            z_t = torch.from_numpy(tmp_input)
        traj = NDP_agent.execute(z_t, z_t) # traj -> [T, dim]
        for t in range(e.horizon - 1):  # this loop is for system evolution, open loop control, no feedback
            if koopmanoption == 'Drafted':  
                x_t_1_computed = traj[t + 1]
                if not Velocity:
                    obj_ori_world = ori_transform_inverse(x_t_1_computed[27:30], e.get_obs()[36:39]) # [27:30] obj_orientation
                    obj_pos_world = x_t_1_computed[24:27]
                else:
                    obj_ori_world = ori_transform_inverse(x_t_1_computed[51:54], e.get_obs()[36:39]) 
                    obj_pos_world = x_t_1_computed[48:51]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint states
            current = e.get_env_state()['qpos'][:num_handpos] # current state
            set_goal = hand_pos.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()   
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation
            if Visualize:
                e.env.mj_render()
            # calculated state values using Koopman rollouts (in simulator)      
            obj_obs = e.get_obs()
            obj_vel = e.get_obs()[27:33]
            # compute the errors
            obj_ori_error[t, k] = np.mean(np.abs(obj_obs[42:45]))  # obj_orien-desired_orien (goal error)
            obj_ori_error_koopman[t, k] = np.mean(np.abs(obj_ori_world - obj_obs[36:39]))  # goal error in koopman
            # compute the success rate
            orien_similarity_sim = np.dot(obj_obs[33:36], obj_obs[36:39])
            orien_similarity_koopman = np.dot(obj_ori_world, obj_obs[36:39])
            dist = np.linalg.norm(obj_obs[39:42])
            desired_pos = -obj_obs[39:42]+obj_obs[24:27]
            dist_koopman = np.linalg.norm(obj_pos_world - desired_pos)
            success_count_sim[t] = 1 if (orien_similarity_sim > 0.90) else 0
            success_count_koopman[t] = 1 if (orien_similarity_koopman > 0.90) else 0
        if np.abs(obj_obs[39:42])[2] > 0.15:
            fall_list_sim.append(1)
        else:
            if sum(success_count_sim) > success_threshold and np.mean(np.abs(obj_vel)) < 1.:
                success_list_sim.append(1)
        if np.abs(obj_pos_world - desired_pos)[2] > 0.15:
            fall_list_koopman.append(1)
        else:
            if sum(success_count_koopman) > success_threshold:
                success_list_koopman.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))
    print("Success rate (koopman) = %f" % (len(success_list_koopman) / len(Test_data)))
    print("Throw out rate (sim) = %f" % (len(fall_list_sim) / len(Test_data)))
    print("Throw out rate (koopman) = %f" % (len(fall_list_koopman) / len(Test_data)))
    return obj_ori_error, obj_ori_error_koopman, success_rate


def NGF_error_visualization(env_name, NGF_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize):
    print("Begin to compute the NGF rollout errors!")
    hand_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_ori_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    Computed_joint_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    Demo_joint_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    e = GymEnv(env_name)
    e.reset()
    state_dict = {}
    for k in tqdm(range(len(Test_data))):
        num_handpos = len(Test_data[k][0]['handpos'])
        num_handvel = len(Test_data[k][0]['handvel'])
        state_dict['desired_orien'] = Test_data[k][0]['pen_desired_orien']
        if Velocity:
            hand_OriState = np.append(Test_data[k][0]['handpos'], Test_data[k][0]['handvel'])  # at initial states
            state_dict['qvel'] = np.append(Test_data[k][0]['handvel'], np.zeros(6))
        else:
            hand_OriState = Test_data[k][0]['handpos']
            state_dict['qvel'] = np.append(np.zeros(num_hand), np.zeros(6))
        Demo_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        Computed_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        obj_OriState = np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objorient'], Test_data[k][0]['objvel']))
        state_dict['qpos'] = np.append(Test_data[k][0]['handpos'], np.zeros(6))
        # {'q0': hand_joint_0 (torch_tensor), 'obj0': object_state_0 (torch_tensor), 'num_traj': num_traj (int), 'time_horizon': time_horizon (int)} 
        Initial_input = dict()
        Initial_input['hand_joint'] = torch.zeros(1, num_hand)
        Initial_input['hand_joint'][0] = torch.from_numpy(hand_OriState)
        Initial_input['object_states'] = torch.zeros(1, num_obj)
        Initial_input['object_states'][0] = torch.from_numpy(obj_OriState)
        traj = NGF_agent(Initial_input)[0].detach().numpy() # traj -> [T, dim]
        for t in range(len(Test_data[k]) - 1):
            e.KoopmanVisualize(seed = None, state_dict = state_dict) # visualize the demo data without any actions (more like a visualization)
            if koopmanoption == 'Drafted':  
                x_t_1_computed = traj[t + 1] # only hand positions
            # ground-truth state values
            gt_hand_pos = Test_data[k][t + 1]['handpos']
            # calculated state values using Koopman rollouts vb
            hand_pos = x_t_1_computed[:num_handpos]
            if Velocity:
                hand_vel = x_t_1_computed[num_handpos:num_handpos + num_handvel]
            else:
                hand_vel = np.zeros(num_handvel)
            # for object orientation, we should covert it back to the original coordinate
            Computed_joint_values[:, t + 1, k] = hand_pos
            Demo_joint_values[:, t + 1, k] = gt_hand_pos
            state_dict['qpos'] = np.append(hand_pos, np.zeros(6))
            state_dict['qvel'] = np.append(hand_vel, np.zeros(6))
            # compute the errors
            hand_pos_error[t, k] = np.mean(np.abs(gt_hand_pos - hand_pos))  # probably not return the mean value?
            obj_obs = e.get_obs()
            obj_ori_error[t, k] = np.mean(np.abs(obj_obs[42:45])) # obj_orien-desired_orien
            if Visualize:
                e.env.mj_render()
    if error_type == 'demo':
        return [hand_pos_error, Computed_joint_values, Demo_joint_values]
    # else:
    #     return [obj_ori_error] # this is incorrect and should not be used, because state_dict['qpos'] is not correct(it requires the orien in robot frame, but it actually is defined in world frame)
    #     # no goal error for this visualization function

def NGF_policy_control(env_name, controller, NGF_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize):
    print("Begin to compute the simulation errors!")
    hand_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_vel_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_ori_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    demo_ori_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_ori_error_koopman = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    Computed_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    Demo_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    Computed_joint_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    Joint_NN_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    success_threshold = 10
    success_list_sim = []
    success_list_RL = []
    fall_list_sim = []
    fall_list_RL = []
    success_rate = str()
    # e.set_env_state(path['init_state_dict'])
    for k in tqdm(range(len(Test_data))):
        success_count_sim = np.zeros(len(Test_data[k]) - 1)
        success_count_RL = np.zeros(len(Test_data[k]) - 1)
        num_handpos = len(Test_data[k][0]['handpos'])
        if Velocity:
            hand_OriState = np.append(Test_data[k][0]['handpos'], Test_data[k][0]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k][0]['handpos']
        obj_OriState = np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objorient'], Test_data[k][0]['objvel']))
        # init_state_dict['qpos'] = np.append(Test_data[k][0]['handpos'] + Test_data[k][0]['final_handpos'], np.zeros(6))
        init_state_dict['qpos'] = np.append(Test_data[k][0]['handpos'], np.zeros(6))
        # For hand control test, we can set the initial pos of pen to be (0.15, 0, 0, 0, 0, 0), so that there is no contact.
        # init_state_dict['qpos'][num_handpos] = 0.15  
        init_state_dict['qvel'] = np.append(Test_data[k][0]['handvel'], np.zeros(6))
        Computed_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        init_state_dict['desired_orien'] = euler2quat(Test_data[k][0]['pen_desired_orien'])
        e.set_env_state(init_state_dict)
        Joint_NN_values[:, 0, k] = e.get_obs()[:24]
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
            Computed_joint_values[:, t + 1, k] = hand_pos
            # hand_pos_desired = hand_pos + Test_data[k][0]['final_handpos']
            hand_pos_desired = hand_pos
            current = e.get_env_state()['qpos'][:num_handpos] # current state
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()   
            Computed_torque_values[:, t, k] = NN_output
            Demo_torque_values[:, t, k] = Test_data[k][t]['action']
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation
            if Visualize:
                e.env.mj_render()
            # ground-truth state values (obtained from RL in simulator)
            # gt_hand_pos = Test_data[k][t + 1]['handpos'] + Test_data[k][0]['final_handpos']
            gt_hand_pos = Test_data[k][t + 1]['handpos']
            gt_obj_pos = Test_data[k][t + 1]['objpos']
            gt_obj_orient = Test_data[k][t + 1]['objorient']  # in demo data, it is defined in a new coordinate 
            gt_obj_orien_world_frame = ori_transform_inverse(gt_obj_orient, e.get_obs()[36:39])
            gt_obj_vel = Test_data[k][t + 1]['objvel']
            # calculated state values using Koopman rollouts (in simulator)
            hand_pos = e.get_obs()[:24]
            Joint_NN_values[:, t + 1, k] = hand_pos
            obj_pos = e.get_obs()[24:27]
            # Test_data[k][*]['desired_ori'] is the same value
            obj_orient = ori_transform(e.get_obs()[33:36], Test_data[k][t]['desired_ori'])  # change the coordinate
            obj_vel = e.get_obs()[27:33]          
            obj_obs = e.get_obs()
            # compute the errors
            hand_pos_error[t, k] = np.mean(np.abs(gt_hand_pos - hand_pos))  # probably not return the mean value?
            hand_pos_PID_error[t, k] = np.mean(np.abs(hand_pos_desired - hand_pos))  
            obj_pos_error[t, k] = np.mean(np.abs(gt_obj_pos - obj_pos) + np.abs(gt_obj_orient - obj_orient))
            obj_vel_error[t, k] = np.mean(np.abs(gt_obj_vel - obj_vel))
            obj_ori_error[t, k] = np.mean(np.abs(obj_obs[42:45]))  # obj_orien-desired_orien (goal error)
            demo_ori_error[t, k] = np.mean(np.abs(Test_data[k][t]['observation']))
            # compute the success rate
            orien_similarity_sim = np.dot(obj_obs[33:36], obj_obs[36:39])
            orien_similarity_RL = np.dot(gt_obj_orien_world_frame, obj_obs[36:39])
            dist = np.linalg.norm(obj_obs[39:42])
            desired_pos = -obj_obs[39:42]+obj_obs[24:27]
            dist_RL = np.linalg.norm(gt_obj_pos - desired_pos)
            success_count_sim[t] = 1 if (orien_similarity_sim > 0.90) else 0
            success_count_RL[t] = 1 if (orien_similarity_RL > 0.90) else 0
        if np.abs(obj_obs[39:42])[2] > 0.15:
            fall_list_sim.append(1)
        else:
            if sum(success_count_sim) > success_threshold and np.mean(np.abs(obj_vel)) < 1.:
                success_list_sim.append(1)
        if np.abs(gt_obj_pos - desired_pos)[2] > 0.15:
            fall_list_RL.append(1)
        else:
            if sum(success_count_RL) > success_threshold:
                success_list_RL.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f\n" % (len(success_list_sim) / len(Test_data))
    print("Success rate (RL) = %f" % (len(success_list_RL) / len(Test_data)))
    print("Throw out rate (sim) = %f" % (len(fall_list_sim) / len(Test_data)))
    success_rate += "Throw out rate (sim) = %f\n" % (len(fall_list_sim) / len(Test_data))
    print("Throw out rate (RL) = %f" % (len(fall_list_RL) / len(Test_data)))
    if error_type == 'demo':
        return [hand_pos_error, hand_pos_PID_error, Computed_torque_values, Demo_torque_values, Computed_joint_values, Joint_NN_values, success_rate]
    else:
        return [obj_ori_error, obj_ori_error_koopman, demo_ori_error, success_rate]     

def NGF_policy_control_unseenTest(env_name, controller, NGF_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, Visualize):
    print("Testing the learned koopman dynamcis!")
    print("Begin to compute the simulation errors!")
    e = GymEnv(env_name)
    obj_ori_error = np.zeros([e.horizon - 1, len(Test_data)])
    obj_ori_error_koopman = np.zeros([e.horizon - 1, len(Test_data)])
    e.reset()
    init_state_dict = dict()
    success_threshold = 10
    success_list_sim = []
    fall_list_sim = []
    success_rate = str()
    # e.set_env_state(path['init_state_dict'])
    for k in tqdm(range(len(Test_data))):
        success_count_sim = np.zeros(e.horizon - 1)
        num_handpos = len(Test_data[k]['handpos'])
        if Velocity:
            hand_OriState = np.append(Test_data[k]['handpos'], Test_data[k]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k]['handpos']
        obj_OriState = np.append(Test_data[k]['objpos'], np.append(Test_data[k]['objorient'], Test_data[k]['objvel']))
        init_state_dict['qpos'] = np.append(Test_data[k]['handpos'], np.zeros(6))
        # For hand control test, we can set the initial pos of pen to be (0.15, 0, 0, 0, 0, 0), so that there is no contact.
        # init_state_dict['qpos'][num_handpos] = 0.15  
        init_state_dict['qvel'] = np.append(Test_data[k]['handvel'], np.zeros(6))
        init_state_dict['desired_orien'] = euler2quat(Test_data[k]['pen_desired_orien'])
        e.set_env_state(init_state_dict)
        Initial_input = dict()
        Initial_input['hand_joint'] = torch.zeros(1, num_hand)
        Initial_input['hand_joint'][0] = torch.from_numpy(hand_OriState)
        Initial_input['object_states'] = torch.zeros(1, num_obj)
        Initial_input['object_states'][0] = torch.from_numpy(obj_OriState)
        traj = NGF_agent(Initial_input)[0].detach().numpy() # traj -> [T, dim]
        for t in range(e.horizon - 1):  # this loop is for system evolution, open loop control, no feedback
            if koopmanoption == 'Drafted':  
                x_t_1_computed = traj[t + 1]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint states
            current = e.get_env_state()['qpos'][:num_handpos] # current state
            set_goal = hand_pos.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()   
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation
            if Visualize:
                e.env.mj_render()
            # calculated state values using Koopman rollouts (in simulator)      
            obj_obs = e.get_obs()
            obj_vel = e.get_obs()[27:33]
            # compute the errors
            obj_ori_error[t, k] = np.mean(np.abs(obj_obs[42:45]))  # obj_orien-desired_orien (goal error)
            # compute the success rate
            orien_similarity_sim = np.dot(obj_obs[33:36], obj_obs[36:39])
            success_count_sim[t] = 1 if (orien_similarity_sim > 0.90) else 0
        if np.abs(obj_obs[39:42])[2] > 0.15:
            fall_list_sim.append(1)
        else:
            if sum(success_count_sim) > success_threshold and np.mean(np.abs(obj_vel)) < 1.:
                success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))
    print("Throw out rate (sim) = %f" % (len(fall_list_sim) / len(Test_data)))
    return obj_ori_error, obj_ori_error_koopman, success_rate















# def gnn_koopman_error_visualization(env_name, koopman_object, koopman_matrix, Test_data, Velocity, num_hand, num_obj, param, error_type):
#     print("Begin to compute the koopman rollout errors!")
#     hand_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
#     obj_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
#     obj_vel_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
#     obj_ori_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
#     hand_dict, gnn_num_obj, gnn_num_relation = koopman_object.Create_states()
#     e = GymEnv(env_name)
#     e.reset()
#     state_dict = {}
#     for k in tqdm(range(len(Test_data))):
#         num_handvel = len(Test_data[k][0]['handvel'])
#         state_dict['desired_orien'] = Test_data[k][0]['pen_desired_orien']
#         if Velocity:
#                 plam_OriState = np.append(Test_data[k][0]['handpos'][hand_dict['palm'][0]:hand_dict['palm'][1]], Test_data[k][0]['handvel'][hand_dict['palm'][0]:hand_dict['palm'][1]])
#                 forfinger_OriState = np.append(Test_data[k][0]['handpos'][hand_dict['forfinger'][0]:hand_dict['forfinger'][1]], Test_data[k][0]['handvel'][hand_dict['forfinger'][0]:hand_dict['forfinger'][1]])
#                 middlefinger_OriState = np.append(Test_data[k][0]['handpos'][hand_dict['middlefinger'][0]:hand_dict['middlefinger'][1]], Test_data[k][0]['handvel'][hand_dict['middlefinger'][0]:hand_dict['middlefinger'][1]])
#                 ringfinger_OriState = np.append(Test_data[k][0]['handpos'][hand_dict['ringfinger'][0]:hand_dict['ringfinger'][1]], Test_data[k][0]['handvel'][hand_dict['ringfinger'][0]:hand_dict['ringfinger'][1]])
#                 littlefinger_OriState = np.append(Test_data[k][0]['handpos'][hand_dict['littlefinger'][0]:hand_dict['littlefinger'][1]], Test_data[k][0]['handvel'][hand_dict['littlefinger'][0]:hand_dict['littlefinger'][1]])
#                 thumb_OriState = np.append(Test_data[k][0]['handpos'][hand_dict['thumb'][0]:hand_dict['thumb'][1]], Test_data[k][0]['handvel'][hand_dict['thumb'][0]:hand_dict['thumb'][1]])
#                 state_dict['qvel'] = np.append(Test_data[k][0]['handvel'], np.zeros(6))
#         else:
#             plam_OriState = Test_data[k][0]['handpos'][hand_dict['palm'][0]:hand_dict['palm'][1]]
#             forfinger_OriState = Test_data[k][0]['handpos'][hand_dict['forfinger'][0]:hand_dict['forfinger'][1]]
#             middlefinger_OriState = Test_data[k][0]['handpos'][hand_dict['middlefinger'][0]:hand_dict['middlefinger'][1]]
#             ringfinger_OriState = Test_data[k][0]['handpos'][hand_dict['ringfinger'][0]:hand_dict['ringfinger'][1]]
#             littlefinger_OriState = Test_data[k][0]['handpos'][hand_dict['littlefinger'][0]:hand_dict['littlefinger'][1]]
#             thumb_OriState = Test_data[k][0]['handpos'][hand_dict['thumb'][0]:hand_dict['thumb'][1]]
#             state_dict['qvel'] = np.append(np.zeros(num_hand), np.zeros(6))
#         obj_OriState = np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objorient'], Test_data[k][0]['objvel']))
#         state_dict['qpos'] = np.append(Test_data[k][0]['handpos'], np.zeros(6))
#         Init_state = np.zeros([gnn_num_obj, num_obj])
#         Init_state[0, 0:plam_OriState.shape[0]] = plam_OriState
#         Init_state[1, 0:forfinger_OriState.shape[0]] = forfinger_OriState
#         Init_state[2, 0:middlefinger_OriState.shape[0]] = middlefinger_OriState
#         Init_state[3, 0:ringfinger_OriState.shape[0]] = ringfinger_OriState
#         Init_state[4, 0:littlefinger_OriState.shape[0]] = littlefinger_OriState
#         Init_state[5, 0:thumb_OriState.shape[0]] = thumb_OriState
#         Init_state[6, 0:obj_OriState.shape[0]] = obj_OriState
#         relations, _ = koopman_object.Create_relations(1, gnn_num_obj, gnn_num_relation, param, 1)
#         Init_state = torch.from_numpy(Init_state).to(torch.float32)[None,:]
#         z_t = koopman_object.z(Init_state, relations, koopman_object._rollout)[0].view(-1).numpy()  # initial states in lifted space
#         for t in range(len(Test_data[k]) - 1):
#             e.KoopmanVisualize(seed = None, state_dict = state_dict) # visualize the demo data without any actions (more like a visualization)
#             z_t_1_computed = np.dot(koopman_matrix, z_t)
#             tmp = torch.from_numpy(z_t_1_computed).view(gnn_num_obj, -1)[None,:]
#             x_t_1_computed = koopman_object.z_inverse(tmp, relations, koopman_object._rollout).numpy()[0]
#             # ground-truth state values
#             gt_hand_pos = Test_data[k][t + 1]['handpos']
#             gt_obj_pos = Test_data[k][t + 1]['objpos']
#             gt_obj_orient = Test_data[k][t + 1]['objorient']
#             gt_obj_vel = Test_data[k][t + 1]['objvel']
#             # calculated state values using Koopman rollouts vb
#             plam_OriState_t = x_t_1_computed[0, 0:plam_OriState.shape[0]]
#             forfinger_OriState_t = x_t_1_computed[1, 0:forfinger_OriState.shape[0]]
#             hand_pos = np.append(plam_OriState_t, forfinger_OriState_t)
#             middlefinger_OriState_t = x_t_1_computed[2, 0:middlefinger_OriState.shape[0]]
#             hand_pos = np.append(hand_pos, middlefinger_OriState_t)
#             ringfinger_OriState_t = x_t_1_computed[3, 0:ringfinger_OriState.shape[0]]
#             hand_pos = np.append(hand_pos, ringfinger_OriState_t)
#             littlefinger_OriState_t = x_t_1_computed[4, 0:littlefinger_OriState.shape[0]]
#             hand_pos = np.append(hand_pos, littlefinger_OriState_t)
#             thumb_OriState_t = x_t_1_computed[5, 0:thumb_OriState.shape[0]]
#             hand_pos = np.append(hand_pos, thumb_OriState_t)
#             obj_OriState_t = x_t_1_computed[6, 0:obj_OriState.shape[0]]
#             obj_pos = obj_OriState_t[:3]
#             obj_orient = obj_OriState_t[3:6]
#             obj_vel = obj_OriState_t[6:]
#             z_t = z_t_1_computed
#             hand_vel = np.zeros(num_handvel)
#             state_dict['qpos'] = np.append(hand_pos, np.append(obj_pos, ori_transform_inverse(obj_orient, Test_data[k][t]['desired_ori'])))
#             state_dict['qvel'] = np.append(hand_vel, obj_vel)
#             # compute the errors
#             hand_pos_error[t, k] = np.mean(np.abs(gt_hand_pos - hand_pos))  # probably not return the mean value?
#             obj_pos_error[t, k] = np.mean(np.abs(gt_obj_pos - obj_pos) + np.abs(gt_obj_orient - obj_orient))
#             obj_vel_error[t, k] = np.mean(np.abs(gt_obj_vel - obj_vel))
#             # compare current state with goal state
#             obj_obs = e.get_obs()
#             obj_ori_error[t, k] = np.mean(np.abs(obj_obs[42:45]))
#     if error_type == 'demo':
#         return [hand_pos_error]
#     else:
#         return [obj_ori_error]  # this is incorrect and should not be used

# def gnn_koopman_policy_control(env_name, controller, koopman_object, koopman_matrix, Test_data, Velocity, num_hand, num_obj, param, error_type, Visualize):
#     print("Begin to compute the simulation errors!")
#     hand_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
#     hand_pos_PID_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
#     obj_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
#     obj_vel_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
#     obj_ori_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
#     demo_ori_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
#     obj_ori_error_koopman = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
#     e = GymEnv(env_name)
#     e.reset()
#     init_state_dict = dict()
#     hand_dict, gnn_num_obj, gnn_num_relation = koopman_object.Create_states()
#     # e.set_env_state(path['init_state_dict'])
#     for k in tqdm(range(len(Test_data))):
#         if Velocity:
#                 plam_OriState = np.append(Test_data[k][0]['handpos'][hand_dict['palm'][0]:hand_dict['palm'][1]], Test_data[k][0]['handvel'][hand_dict['palm'][0]:hand_dict['palm'][1]])
#                 forfinger_OriState = np.append(Test_data[k][0]['handpos'][hand_dict['forfinger'][0]:hand_dict['forfinger'][1]], Test_data[k][0]['handvel'][hand_dict['forfinger'][0]:hand_dict['forfinger'][1]])
#                 middlefinger_OriState = np.append(Test_data[k][0]['handpos'][hand_dict['middlefinger'][0]:hand_dict['middlefinger'][1]], Test_data[k][0]['handvel'][hand_dict['middlefinger'][0]:hand_dict['middlefinger'][1]])
#                 ringfinger_OriState = np.append(Test_data[k][0]['handpos'][hand_dict['ringfinger'][0]:hand_dict['ringfinger'][1]], Test_data[k][0]['handvel'][hand_dict['ringfinger'][0]:hand_dict['ringfinger'][1]])
#                 littlefinger_OriState = np.append(Test_data[k][0]['handpos'][hand_dict['littlefinger'][0]:hand_dict['littlefinger'][1]], Test_data[k][0]['handvel'][hand_dict['littlefinger'][0]:hand_dict['littlefinger'][1]])
#                 thumb_OriState = np.append(Test_data[k][0]['handpos'][hand_dict['thumb'][0]:hand_dict['thumb'][1]], Test_data[k][0]['handvel'][hand_dict['thumb'][0]:hand_dict['thumb'][1]])
#         else:
#             plam_OriState = Test_data[k][0]['handpos'][hand_dict['palm'][0]:hand_dict['palm'][1]]
#             forfinger_OriState = Test_data[k][0]['handpos'][hand_dict['forfinger'][0]:hand_dict['forfinger'][1]]
#             middlefinger_OriState = Test_data[k][0]['handpos'][hand_dict['middlefinger'][0]:hand_dict['middlefinger'][1]]
#             ringfinger_OriState = Test_data[k][0]['handpos'][hand_dict['ringfinger'][0]:hand_dict['ringfinger'][1]]
#             littlefinger_OriState = Test_data[k][0]['handpos'][hand_dict['littlefinger'][0]:hand_dict['littlefinger'][1]]
#             thumb_OriState = Test_data[k][0]['handpos'][hand_dict['thumb'][0]:hand_dict['thumb'][1]]
#         obj_OriState = np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objorient'], Test_data[k][0]['objvel']))
#         Init_state = np.zeros([gnn_num_obj, num_obj])
#         Init_state[0, 0:plam_OriState.shape[0]] = plam_OriState
#         Init_state[1, 0:forfinger_OriState.shape[0]] = forfinger_OriState
#         Init_state[2, 0:middlefinger_OriState.shape[0]] = middlefinger_OriState
#         Init_state[3, 0:ringfinger_OriState.shape[0]] = ringfinger_OriState
#         Init_state[4, 0:littlefinger_OriState.shape[0]] = littlefinger_OriState
#         Init_state[5, 0:thumb_OriState.shape[0]] = thumb_OriState
#         Init_state[6, 0:obj_OriState.shape[0]] = obj_OriState
#         init_state_dict['qpos'] = np.append(Test_data[k][0]['handpos'], np.zeros(6))
#         # For hand control test, we can set the initial pos of pen to be (0.15, 0, 0, 0, 0, 0), so that there is no contact.
#         # init_state_dict['qpos'][num_handpos] = 0.15  
#         init_state_dict['qvel'] = np.append(Test_data[k][0]['handvel'], np.zeros(6))
#         init_state_dict['desired_orien'] = euler2quat(Test_data[k][0]['pen_desired_orien']) # for various goal orientations, this is different
#         e.set_env_state(init_state_dict)
#         relations, _ = koopman_object.Create_relations(1, gnn_num_obj, gnn_num_relation, param, 1)
#         Init_state = torch.from_numpy(Init_state).to(torch.float32)[None,:]
#         z_t = koopman_object.z(Init_state, relations, koopman_object._rollout)[0].view(-1).numpy()  # initial states in lifted space
#         for t in range(len(Test_data[k]) - 1):  # this loop is for system evolution, open loop control, no feedback
#             z_t_1_computed = np.dot(koopman_matrix, z_t)
#             tmp = torch.from_numpy(z_t_1_computed).view(gnn_num_obj, -1)[None,:]
#             x_t_1_computed = koopman_object.z_inverse(tmp, relations, koopman_object._rollout).numpy()[0]
#             plam_OriState_t = x_t_1_computed[0, 0:plam_OriState.shape[0]]
#             forfinger_OriState_t = x_t_1_computed[1, 0:forfinger_OriState.shape[0]]
#             hand_pos = np.append(plam_OriState_t, forfinger_OriState_t)
#             middlefinger_OriState_t = x_t_1_computed[2, 0:middlefinger_OriState.shape[0]]
#             hand_pos = np.append(hand_pos, middlefinger_OriState_t)
#             ringfinger_OriState_t = x_t_1_computed[3, 0:ringfinger_OriState.shape[0]]
#             hand_pos = np.append(hand_pos, ringfinger_OriState_t)
#             littlefinger_OriState_t = x_t_1_computed[4, 0:littlefinger_OriState.shape[0]]
#             hand_pos = np.append(hand_pos, littlefinger_OriState_t)
#             thumb_OriState_t = x_t_1_computed[5, 0:thumb_OriState.shape[0]]
#             hand_pos = np.append(hand_pos, thumb_OriState_t)
#             hand_pos_desired = hand_pos
#             obj_OriState_t = x_t_1_computed[6, 0:obj_OriState.shape[0]]
#             obj_orient = obj_OriState_t[3:6]
#             obj_ori_world = ori_transform_inverse(obj_orient, e.get_obs()[36:39]) # [27:30] obj_orientation
#             z_t = z_t_1_computed
#             controller.set_goal(hand_pos)
#             torque_action = controller(e.get_env_state()['qpos'][:num_hand])
#             e.step(torque_action)  # Visualize the demo using the actions (more like a simulation)
#             if Visualize:
#                 e.env.mj_render()
#             # ground-truth state values (obtained from RL in simulator)
#             gt_hand_pos = Test_data[k][t + 1]['handpos']
#             gt_obj_pos = Test_data[k][t + 1]['objpos']
#             gt_obj_orient = Test_data[k][t + 1]['objorient']
#             gt_obj_vel = Test_data[k][t + 1]['objvel']
#             # calculated state values using Koopman rollouts (in simulator)
#             hand_pos = e.get_obs()[:24]
#             obj_pos = e.get_obs()[24:27]
#             obj_orient = ori_transform(e.get_obs()[33:36], Test_data[k][t]['desired_ori'])  # change the coordinate
#             obj_vel = e.get_obs()[27:33] 
#             obj_obs = e.get_obs()
#             # compute the errors
#             # compare current states with ground truths
#             hand_pos_error[t, k] = np.mean(np.abs(gt_hand_pos - hand_pos))  # probably not return the mean value?
#             hand_pos_PID_error[t, k] = np.mean(np.abs(hand_pos_desired - hand_pos))  
#             obj_pos_error[t, k] = np.mean(np.abs(gt_obj_pos - obj_pos) + np.abs(gt_obj_orient - obj_orient))
#             obj_vel_error[t, k] = np.mean(np.abs(gt_obj_vel - obj_vel))
#             # compare current state with goal state
#             obj_ori_error[t, k] = np.mean(np.abs(obj_obs[42:45])) # obj_orien-desired_orien
#             obj_ori_error_koopman[t, k] = np.mean(np.abs(obj_ori_world - obj_obs[36:39]))  # goal error in koopman
#             demo_ori_error[t, k] = np.mean(np.abs(Test_data[k][t]['observation']))
#     if error_type == 'demo':
#         return [hand_pos_error, hand_pos_PID_error]
#     else:
#         return [obj_ori_error, obj_ori_error_koopman, demo_ori_error]  
    
# def gnn_koopman_policy_control_unseenTest(env_name, controller, koopman_object, koopman_matrix, Test_data, Velocity, num_hand, num_obj, param, Visualize):
#     print("Begin to compute the simulation errors!")
#     e = GymEnv(env_name)
#     obj_ori_error = np.zeros([e.horizon - 1, len(Test_data)])
#     obj_ori_error_koopman = np.zeros([e.horizon - 1, len(Test_data)])
#     e.reset()
#     init_state_dict = dict()
#     hand_dict, gnn_num_obj, gnn_num_relation = koopman_object.Create_states()
#     # e.set_env_state(path['init_state_dict'])
#     for k in tqdm(range(len(Test_data))):
#         if Velocity:
#                 plam_OriState = np.append(Test_data[k]['handpos'][hand_dict['palm'][0]:hand_dict['palm'][1]], Test_data[k]['handvel'][hand_dict['palm'][0]:hand_dict['palm'][1]])
#                 forfinger_OriState = np.append(Test_data[k]['handpos'][hand_dict['forfinger'][0]:hand_dict['forfinger'][1]], Test_data[k]['handvel'][hand_dict['forfinger'][0]:hand_dict['forfinger'][1]])
#                 middlefinger_OriState = np.append(Test_data[k]['handpos'][hand_dict['middlefinger'][0]:hand_dict['middlefinger'][1]], Test_data[k]['handvel'][hand_dict['middlefinger'][0]:hand_dict['middlefinger'][1]])
#                 ringfinger_OriState = np.append(Test_data[k]['handpos'][hand_dict['ringfinger'][0]:hand_dict['ringfinger'][1]], Test_data[k]['handvel'][hand_dict['ringfinger'][0]:hand_dict['ringfinger'][1]])
#                 littlefinger_OriState = np.append(Test_data[k]['handpos'][hand_dict['littlefinger'][0]:hand_dict['littlefinger'][1]], Test_data[k]['handvel'][hand_dict['littlefinger'][0]:hand_dict['littlefinger'][1]])
#                 thumb_OriState = np.append(Test_data[k]['handpos'][hand_dict['thumb'][0]:hand_dict['thumb'][1]], Test_data[k]['handvel'][hand_dict['thumb'][0]:hand_dict['thumb'][1]])
#         else:
#             plam_OriState = Test_data[k]['handpos'][hand_dict['palm'][0]:hand_dict['palm'][1]]
#             forfinger_OriState = Test_data[k]['handpos'][hand_dict['forfinger'][0]:hand_dict['forfinger'][1]]
#             middlefinger_OriState = Test_data[k]['handpos'][hand_dict['middlefinger'][0]:hand_dict['middlefinger'][1]]
#             ringfinger_OriState = Test_data[k]['handpos'][hand_dict['ringfinger'][0]:hand_dict['ringfinger'][1]]
#             littlefinger_OriState = Test_data[k]['handpos'][hand_dict['littlefinger'][0]:hand_dict['littlefinger'][1]]
#             thumb_OriState = Test_data[k]['handpos'][hand_dict['thumb'][0]:hand_dict['thumb'][1]]
#         obj_OriState = np.append(Test_data[k]['objpos'], np.append(Test_data[k]['objorient'], Test_data[k]['objvel']))
#         Init_state = np.zeros([gnn_num_obj, num_obj])
#         Init_state[0, 0:plam_OriState.shape[0]] = plam_OriState
#         Init_state[1, 0:forfinger_OriState.shape[0]] = forfinger_OriState
#         Init_state[2, 0:middlefinger_OriState.shape[0]] = middlefinger_OriState
#         Init_state[3, 0:ringfinger_OriState.shape[0]] = ringfinger_OriState
#         Init_state[4, 0:littlefinger_OriState.shape[0]] = littlefinger_OriState
#         Init_state[5, 0:thumb_OriState.shape[0]] = thumb_OriState
#         Init_state[6, 0:obj_OriState.shape[0]] = obj_OriState
#         init_state_dict['qpos'] = np.append(Test_data[k]['handpos'], np.zeros(6))
#         # For hand control test, we can set the initial pos of pen to be (0.15, 0, 0, 0, 0, 0), so that there is no contact.
#         # init_state_dict['qpos'][num_handpos] = 0.15  
#         init_state_dict['qvel'] = np.append(Test_data[k]['handvel'], np.zeros(6))
#         init_state_dict['desired_orien'] = euler2quat(Test_data[k]['pen_desired_orien']) # for various goal orientations, this is different
#         e.set_env_state(init_state_dict)
#         relations, _ = koopman_object.Create_relations(1, gnn_num_obj, gnn_num_relation, param, 1)
#         Init_state = torch.from_numpy(Init_state).to(torch.float32)[None,:]
#         z_t = koopman_object.z(Init_state, relations, koopman_object._rollout)[0].view(-1).numpy()  # initial states in lifted space
#         for t in range(e.horizon - 1):  # this loop is for system evolution, open loop control, no feedback
#             z_t_1_computed = np.dot(koopman_matrix, z_t)
#             tmp = torch.from_numpy(z_t_1_computed).view(gnn_num_obj, -1)[None,:]
#             x_t_1_computed = koopman_object.z_inverse(tmp, relations, koopman_object._rollout).numpy()[0]
#             plam_OriState_t = x_t_1_computed[0, 0:plam_OriState.shape[0]]
#             forfinger_OriState_t = x_t_1_computed[1, 0:forfinger_OriState.shape[0]]
#             hand_pos = np.append(plam_OriState_t, forfinger_OriState_t)
#             middlefinger_OriState_t = x_t_1_computed[2, 0:middlefinger_OriState.shape[0]]
#             hand_pos = np.append(hand_pos, middlefinger_OriState_t)
#             ringfinger_OriState_t = x_t_1_computed[3, 0:ringfinger_OriState.shape[0]]
#             hand_pos = np.append(hand_pos, ringfinger_OriState_t)
#             littlefinger_OriState_t = x_t_1_computed[4, 0:littlefinger_OriState.shape[0]]
#             hand_pos = np.append(hand_pos, littlefinger_OriState_t)
#             thumb_OriState_t = x_t_1_computed[5, 0:thumb_OriState.shape[0]]
#             hand_pos = np.append(hand_pos, thumb_OriState_t)
#             obj_OriState_t = x_t_1_computed[6, 0:obj_OriState.shape[0]]
#             obj_orient = obj_OriState_t[3:6]
#             obj_ori_world = ori_transform_inverse(obj_orient, e.get_obs()[36:39]) # [27:30] obj_orientation
#             z_t = z_t_1_computed
#             controller.set_goal(hand_pos)
#             torque_action = controller(e.get_env_state()['qpos'][:num_hand])
#             e.step(torque_action)  # Visualize the demo using the actions (more like a simulation)
#             if Visualize:
#                 e.env.mj_render()
#             obj_obs = e.get_obs()
#             # compute the errors
#             # compare current state with goal state
#             obj_ori_error[t, k] = np.mean(np.abs(obj_obs[42:45])) # obj_orien-desired_orien
#             obj_ori_error_koopman[t, k] = np.mean(np.abs(obj_ori_world - obj_obs[36:39]))  # goal error in koopman
#     return obj_ori_error, obj_ori_error_koopman
