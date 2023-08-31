from cProfile import label
from glob import escape
from attr import asdict
import torch
import mj_envs
import click 
import json
import os
import numpy as np
import gym
import pickle
from tqdm import tqdm
from utils.gym_env import GymEnv
from utils.Observables import *
from utils.Koopman_evaluation import *
from utils.Controller import *
from utils.quatmath import quat2euler, euler2quat
from utils.coord_trans import ori_transform, ori_transform_inverse
from utils.gaussian_rnn import RNN
from utils.fc_network import FCNetwork
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.dmp_layer import DMPIntegrator, DMPParameters
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.linalg import logm
import sys
import os
import random
import time
import shutil

DESC = '''
Helper script to visualize demonstrations.\n
USAGE:\n
    Visualizes demonstrations on the env\n
    $ python utils/visualize_demos --env_name relocate-v0\n
'''

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', required= True)
@click.option('--demo_file', type=str, help='demo file to load', default=None)
@click.option('--num_demo', type=str, help='define the number of demo', default='0') 
@click.option('--koopmanoption', type=str, help='Indicate the koopman choice (Drafted, MLP, GNN)', default=None)   
@click.option('--velocity', type=str, help='If using hand velocity', default=None)
@click.option('--save_matrix', type=str, help='If save the koopman matrix after training', default=None)
@click.option('--matrix_file', type=str, help='If loading the saved matrix/model for test', default='')
@click.option('--config', type=str, help='load the network info for MLP and GNN', default='')
@click.option('--control', type=str, help='apply with a controller', default='')
@click.option('--error_type', type=str, help='define how to calculate the errors', default='demo') # two options: demo; goal
@click.option('--visualize', type=str, help='define whether or not to visualze the manipulation results', default='') # two options: demo; goal
@click.option('--unseen_test', type=str, help='define if we generate unseen data for testing the learned dynamics model', default='') # generate new testing data
@click.option('--rl_policy', type=str, help='define the file location of the well-trained policy', default='') 
@click.option('--folder_name', type=str, help='define the location to put trained models', default='') 
@click.option('--matrix_list', type=str, help='define the file location of all learned koopman matrix', default='')
#Separate is needed because we have to modify the z function for each drafted policy, so we use this to store the intermediate results for each one
@click.option('--separate', type=str, help='For comparing the number of observables, we separate the test by storing the results locally', default='') 
@click.option('--seed', type=str, help='random seed', default='1') 
@click.option('--first_demo', type=str, help='random seed', default='1') 

def main(env_name, demo_file, num_demo, koopmanoption, velocity, save_matrix, matrix_file, config, control, error_type, visualize, unseen_test, rl_policy, folder_name, matrix_list, separate, seed, first_demo):
    num_demo = int(num_demo) # if num_demo != 0 -> we manually define the num of demo for testing the sample efficiency
    Velocity = True if velocity == 'True' else False
    save_matrix = True if save_matrix == 'True' else False  # save the Koopman matrix for Drafted method and trained model for MLP/GNN
    controller = True if control == 'True' else False
    Visualize = True if visualize == 'True' else False
    Unseen_test = True if unseen_test == 'True' else False
    number_sample = 100  # num of unseen samples used for test
    seed = int(seed)
    first_demo = int(first_demo)
    # Keep using this controller
    Controller_loc = 'Results/Drafted/Hammer/11_10_2022_22_07_15/PID_NN/NN_controller_best.pt'
    multiple_test = True
    if env_name is "":
        print("Unknown env.")
        return
    if demo_file is None:
        demos = pickle.load(open('./demonstrations/'+env_name+'_demos.pickle', 'rb'))
    else:
        demos = pickle.load(open(demo_file, 'rb'))
    if num_demo == 0 or num_demo > len(demos):
        num_demo = len(demos)
    # using the recommendated fig params from https://github.com/jbmouret/matplotlib_for_papers#pylab-vs-matplotlib
    fig_params = {
    'axes.labelsize': 10,
    'font.size': 10,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': False,
    'figure.figsize': [5, 4.5]
    }
    mpl.rcParams.update(fig_params)
    error_calc = 'median' # median error or mean error
    unit_train = False
    if len(folder_name) == 0: # we do not explicitly define the folder name
        folder_name = '/Hammer/NDP_cloning/parameter_tuning/'  # define the folder name for training
        unit_train = True
    if not os.path.exists(matrix_file):  # loading data for training
        Training_data = demo_playback(env_name, demos, num_demo) # len(Training_data) = num_demo
        num_handpos = len(Training_data[0][0]['handpos'])
        num_handvel = len(Training_data[0][0]['handvel'])
        num_objpos = len(Training_data[0][0]['objpos']) + len(Training_data[0][0]['objorient'])
        num_objvel = len(Training_data[0][0]['objvel'])
        num_goal_pose = len(Training_data[0][0]['nail_goal'])
        num_obj = num_objpos + num_objvel + num_goal_pose
        if Velocity: # hand velocities are also used as the original states
            num_hand = num_handpos + num_handvel
        else:
            num_hand = num_handpos
    # Load the trained matrix/model and roll it out in simulation only
    else:
        if Unseen_test: # test the performance on unseen data
            e = GymEnv(env_name)
            rl_policy = pickle.load(open(rl_policy, 'rb'))  # load the well-trained RL policy
            # Remember to change the pitch&yaw range for each task specification (under reset_model())
            Training_data = e.generate_unseen_data_hammer(number_sample)  
            num_handpos = len(Training_data[0]['handpos'])
            num_handvel = len(Training_data[0]['handvel'])
            num_objpos = len(Training_data[0]['objpos']) + len(Training_data[0]['objorient'])
            num_objvel = len(Training_data[0]['objvel'])
            num_goal_pose = len(Training_data[0]['nail_goal'])
            num_obj = num_objpos + num_objvel + num_goal_pose
            if Velocity: # hand velocities are also used as the original states
                num_hand = num_handpos + num_handvel
            else:
                num_hand = num_handpos
        else: # test the performance on demo data
            Training_data = demo_playback(env_name, demos, len(demos)) # Test on all the demo data
            num_handpos = len(Training_data[0][0]['handpos'])
            num_handvel = len(Training_data[0][0]['handvel'])
            num_objpos = len(Training_data[0][0]['objpos']) + len(Training_data[0][0]['objorient'])
            num_objvel = len(Training_data[0][0]['objvel'])
            num_goal_pose = len(Training_data[0][0]['nail_goal'])
            num_obj = num_objpos + num_objvel + num_goal_pose
            if Velocity: # hand velocities are also used as the original states
                num_hand = num_handpos + num_handvel
            else:
                num_hand = num_handpos
        # define the input and outputs of the neural network
        NN_Input_size = 2 * num_hand
        NN_Output_size = num_hand
        # NN_size may vary
        NN_size = [NN_Input_size, 2 * NN_Input_size, 2 * NN_Input_size, NN_Input_size, NN_Output_size]  # define a neural network to learn the PID mapping
        Controller = FCNetwork(NN_size, nonlinearity='relu') # load the controller
        Controller.load_state_dict(torch.load(Controller_loc))
        Controller.eval() # eval mode
        # define the input and outputs of the NDPs
        k = 1 
        # only one rollout (only one input image, but the whole traj with T steps)
        # the number of rollout = the number of DMP/NDP trained in each traj.
        # if we only have one rollout, only one NDP, so all the demo data along each traj is used to train the param for this NDP.
        # define the input and outputs of NDPs    
        NN_Input_size = num_obj + num_hand 
        separate = 2 # large model for testing
        if int(separate) == 0:
            N = 10
            hidden_layer = [NN_Input_size, 32, 64, 32]
        elif int(separate) == 1:
            N = 20
            hidden_layer = [NN_Input_size, 64, 128, 64]
        else:
            N = 30
            hidden_layer = [NN_Input_size, 128, 256, 128]
        NDP_policy = NdpCnn(T=100,l=1, N=N, state_index=np.arange(NN_Input_size), layer_sizes = hidden_layer)
        NDP_policy.load_state_dict(torch.load(matrix_file))
        NDP_policy.eval()
        fig_path_tmp = [ele + '/' for ele in matrix_file.split('/')]
        fig_path_visual_demo = ''.join(fig_path_tmp[:-1]) + 'RolloutError_visual(DemoError).png'
        fig_path_simu_demo_handJoint = ''.join(fig_path_tmp[:-1]) + 'RolloutError_simu(DemoError_on_Handjoints).png'
        fig_path_simu_demo_handRT = ''.join(fig_path_tmp[:-1]) + 'RolloutError_simu(DemoError_on_Base).png'
        fig_path_simu_goal = ''.join(fig_path_tmp[:-1]) + 'RolloutError_simu(GoalError).png'
        current_time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        fig_path_simu_goal_unseen = ''.join(fig_path_tmp[:-1]) + 'RolloutError_simu_unseen(GoalError)_' + current_time + '.png'
        fig_hand_joints_demo = [''.join(fig_path_tmp[:-1]) + 'Joint/joints_' + str(i) + '.png' for i in range(num_hand)]
        fig_hand_torques_demo = [''.join(fig_path_tmp[:-1]) + 'Torque/torques_' + str(i) + '.png' for i in range(num_hand)]
        fig_hand_joints_unseen_demo = [''.join(fig_path_tmp[:-1]) + 'Joint_unseen/torques_' + str(i) + '.png' for i in range(num_hand)]
        if koopmanoption == "Drafted":
            print("Trained drafted koopman matrix loaded!")
            if not Unseen_test:
                errors = NDP_error_visualization(env_name, NDP_policy, Training_data, Velocity, num_hand, num_obj, koopmanoption, error_type, False)
                if error_type == 'demo':
                    hand_pos_error_joints = errors[0]
                    hand_pos_error_base = errors[1]
                    nail_goal_pos_error = errors[2]
                    computed_joints = errors[3]
                    demo_joints = errors[4]
                if controller:
                    errors_simu = NDP_policy_control(env_name, Controller, NDP_policy, Training_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize)
                    if error_type == 'demo':
                        hand_pos_error_simu_joint = errors_simu[0]
                        hand_pos_error_PID_simu_joint = errors_simu[1]
                        hand_pos_error_simu_base = errors_simu[2]
                        hand_pos_error_PID_simu_base = errors_simu[3]
                        computed_torques = errors_simu[4]
                        demo_torques = errors_simu[5]
                        success_rate = errors_simu[6]
                    else:  # goal
                        obj_ori_error_simu = errors_simu[0]
                        obj_ori_error_simu_koopman = errors_simu[1]
                        demo_ori_error_simu = errors_simu[2]
            else: # If we choose to test the learned dynamics on the unseen data, make sure 'error_type' is set to be 'goal' and enable the controller
                if not controller or error_type == 'demo':
                    print("Please make sure 'error_type' is set to be 'goal' and enable the controller when testing the learned koopman on unseen data")
                    sys.exit()
                else:
                    obj_ori_error_simu, obj_ori_error_simu_koopman, Computed_joint_unseen, success_rate = NDP_policy_control_unseenTest(env_name, Controller, NDP_policy, Training_data, Velocity, num_hand, num_obj, koopmanoption, Visualize)      
                    demo_ori_error_simu, RL_joint_unseen = e.visualize_policy_on_demos(rl_policy, Training_data, Visualize, 31)
        # plot the figures
        if not Unseen_test:  # on demo data                 
            if error_type == 'demo':  # visualize the errors wrt demo data (pure visualization)
                x = np.arange(0, hand_pos_error_joints.shape[0])
                plt.figure(1)
                plt.axes(frameon=0)
                plt.grid()
                if error_calc == 'median':  # plot median/percentile 
                    plt.plot(x, np.median(hand_pos_error_joints, axis = 1), linewidth=2, label = 'hand joint error (koopman)', color='#B22400')
                    plt.fill_between(x, np.percentile(hand_pos_error_joints, 25, axis = 1), np.percentile(hand_pos_error_joints, 75, axis = 1), alpha = 0.15, linewidth = 0, color='#B22400')
                    plt.plot(x, np.median(hand_pos_error_base, axis = 1), linewidth=2, label = 'base error (koopman)', color='#F22BB2')
                    plt.fill_between(x, np.percentile(hand_pos_error_base, 25, axis = 1), np.percentile(hand_pos_error_base, 75, axis = 1), alpha = 0.15, linewidth = 0, color='#F22BB2')
                    # In demo data, the nail goal pos are fixed for each piece
                    plt.plot(x, np.median(nail_goal_pos_error, axis = 1), linewidth=2, label = 'nail goal pos error', color='#006BB2')
                    plt.fill_between(x, np.percentile(nail_goal_pos_error, 25, axis = 1), np.percentile(nail_goal_pos_error, 75, axis = 1), alpha = 0.15, linewidth = 0, color='#006BB2')
                else:  # plot mean 
                    plt.plot(x, np.mean(hand_pos_error_joints, axis = 1), linewidth=2, label = 'hand joint error', color='#B22400')
                    plt.plot(x, np.mean(hand_pos_error_base, axis = 1), linewidth=2, label = 'base error', color='#F22BB2')
                    plt.plot(x, np.mean(nail_goal_pos_error, axis = 1), linewidth=2, label = 'nail goal pos error', color='#006BB2')
                plt.xlabel('Time step')
                plt.ylabel('Difference error wrt demo')
                legend = plt.legend()
                frame = legend.get_frame()
                frame.set_facecolor('0.9')
                frame.set_edgecolor('0.9')
                plt.savefig(fig_path_visual_demo)
                if multiple_test:
                    np.save(''.join(fig_path_tmp[:-1]) + 'computed_joints.npy', computed_joints)
                    np.save(''.join(fig_path_tmp[:-1]) + 'demo_joints.npy', demo_joints)
                else:
                    x = np.arange(0, computed_joints.shape[1])
                    for i in range(num_hand):
                        plt.figure(20 + i)
                        plt.axes(frameon=0)
                        plt.grid()
                        if error_calc == 'median':  # plot median/percentile 
                            plt.plot(x, np.median(computed_joints[i], axis = 1), linewidth=2, label = 'Koopman rollout', color='#B22400')
                            plt.fill_between(x, np.percentile(computed_joints[i], 25, axis = 1), np.percentile(computed_joints[i], 75, axis = 1), alpha = 0.15, linewidth = 0, color='#B22400')
                            plt.plot(x, np.median(demo_joints[i], axis = 1), linewidth=2, label = 'Demo', color='#F22BB2')
                            plt.fill_between(x, np.percentile(demo_joints[i], 25, axis = 1), np.percentile(demo_joints[i], 75, axis = 1), alpha = 0.15, linewidth = 0, color='#F22BB2')
                        else:  # plot mean 
                            plt.plot(x, np.mean(computed_joints[i], axis = 1), linewidth=2, label = 'Koopman rollout', color='#B22400')
                            plt.plot(x, np.mean(demo_joints[i], axis = 1), linewidth=2, label = 'Demo', color='#F22BB2')
                        plt.xlabel('Time step')
                        plt.ylabel('Joint%d'%(i))
                        legend = plt.legend()
                        frame = legend.get_frame()
                        frame.set_facecolor('0.9')
                        frame.set_edgecolor('0.9')
                        plt.savefig(fig_hand_joints_demo[i]) 
            # goal error wrt visualization is not available for now.
            if controller: # using the controller
                if error_type == 'demo': # visualize the errors wrt demo data
                    x_simu = np.arange(0, hand_pos_error_simu_joint.shape[0])
                    plt.figure(2)
                    plt.axes(frameon=0)
                    plt.grid()
                    if error_calc == 'median':  # plot median/percentile 
                        plt.plot(x_simu, np.median(hand_pos_error_simu_joint, axis = 1), linewidth=2, label = 'hand joint error (simu)', color='#B22400')
                        plt.fill_between(x_simu, np.percentile(hand_pos_error_simu_joint, 25, axis = 1), np.percentile(hand_pos_error_simu_joint, 75, axis = 1), alpha = 0.15, linewidth = 0, color='#B22400')
                        plt.plot(x_simu, np.median(hand_pos_error_joints, axis = 1), linewidth=2, label = 'hand joint error (koopman)', color='#F22BB2')
                        plt.fill_between(x_simu, np.percentile(hand_pos_error_joints, 25, axis = 1), np.percentile(hand_pos_error_joints, 75, axis = 1), alpha = 0.15, linewidth = 0, color='#F22BB2')
                        plt.plot(x_simu, np.median(hand_pos_error_PID_simu_joint, axis = 1), linewidth=2, label = 'hand joint error (PID tracking)', color='#006BB2')
                        plt.fill_between(x_simu, np.percentile(hand_pos_error_PID_simu_joint, 25, axis = 1), np.percentile(hand_pos_error_PID_simu_joint, 75, axis = 1), alpha = 0.15, linewidth = 0, color='#006BB2')
                    else:  # plot mean 
                        plt.plot(x_simu, np.mean(hand_pos_error_simu_joint, axis = 1), linewidth=2, label = 'hand joint error (simu)', color='#B22400')
                        plt.plot(x_simu, np.mean(hand_pos_error_joints, axis = 1), linewidth=2, label = 'hand joint error (koopman)', color='#F22BB2')
                        plt.plot(x_simu, np.mean(hand_pos_error_PID_simu_joint, axis = 1), linewidth=2, label = 'hand joint error (PID tracking)', color='#006BB2')
                    plt.xlabel('Time step')
                    plt.ylabel('joint tracking error wrt demo')
                    legend = plt.legend()
                    frame = legend.get_frame()
                    frame.set_facecolor('0.9')
                    frame.set_edgecolor('0.9')
                    plt.savefig(fig_path_simu_demo_handJoint)
                    x_simu = np.arange(0, hand_pos_error_simu_base.shape[0])
                    plt.figure(3)
                    plt.axes(frameon=0)
                    plt.grid()
                    if error_calc == 'median':  # plot median/percentile 
                        plt.plot(x_simu, np.median(hand_pos_error_simu_base, axis = 1), linewidth=2, label = 'base error (simu)', color='#B22400')
                        plt.fill_between(x_simu, np.percentile(hand_pos_error_simu_base, 25, axis = 1), np.percentile(hand_pos_error_simu_base, 75, axis = 1), alpha = 0.15, linewidth = 0, color='#B22400')
                        plt.plot(x_simu, np.median(hand_pos_error_base, axis = 1), linewidth=2, label = 'base error (koopman)', color='#F22BB2')
                        plt.fill_between(x_simu, np.percentile(hand_pos_error_base, 25, axis = 1), np.percentile(hand_pos_error_base, 75, axis = 1), alpha = 0.15, linewidth = 0, color='#F22BB2')
                        plt.plot(x_simu, np.median(hand_pos_error_PID_simu_base, axis = 1), linewidth=2, label = 'base error (PID tracking)', color='#006BB2')
                        plt.fill_between(x_simu, np.percentile(hand_pos_error_PID_simu_base, 25, axis = 1), np.percentile(hand_pos_error_PID_simu_base, 75, axis = 1), alpha = 0.15, linewidth = 0, color='#006BB2')
                    else:  # plot mean 
                        plt.plot(x_simu, np.mean(hand_pos_error_simu_base, axis = 1), linewidth=2, label = 'base error (simu)', color='#B22400')
                        plt.plot(x_simu, np.mean(hand_pos_error_base, axis = 1), linewidth=2, label = 'base error (koopman)', color='#F22BB2')
                        plt.plot(x_simu, np.mean(hand_pos_error_PID_simu_base, axis = 1), linewidth=2, label = 'base error (PID tracking)', color='#006BB2')
                    plt.xlabel('Time step')
                    plt.ylabel('Base tracking error wrt demo')
                    legend = plt.legend()
                    frame = legend.get_frame()
                    frame.set_facecolor('0.9')
                    frame.set_edgecolor('0.9')
                    plt.savefig(fig_path_simu_demo_handRT)
                    if multiple_test:
                        np.save(''.join(fig_path_tmp[:-1]) + 'computed_torques.npy', computed_torques)
                        np.save(''.join(fig_path_tmp[:-1]) + 'demo_torques.npy', demo_torques)
                        with open(''.join(fig_path_tmp[:-1]) + 'success.txt', 'w') as f:
                            f.write(success_rate)
                    else:
                        x = np.arange(0, computed_torques.shape[1])
                        for i in range(num_hand):
                            plt.figure(60 + i)
                            plt.axes(frameon=0)
                            plt.grid()
                            if error_calc == 'median':  # plot median/percentile 
                                plt.plot(x, np.median(computed_torques[i], axis = 1), linewidth=2, label = 'PID', color='#B22400')
                                plt.fill_between(x, np.percentile(computed_torques[i], 25, axis = 1), np.percentile(computed_torques[i], 75, axis = 1), alpha = 0.15, linewidth = 0, color='#B22400')
                                plt.plot(x, np.median(demo_torques[i], axis = 1), linewidth=2, label = 'Demo', color='#F22BB2')
                                plt.fill_between(x, np.percentile(demo_torques[i], 25, axis = 1), np.percentile(demo_torques[i], 75, axis = 1), alpha = 0.15, linewidth = 0, color='#F22BB2')
                            else:  # plot mean 
                                plt.plot(x, np.mean(computed_torques[i], axis = 1), linewidth=2, label = 'PID', color='#B22400')
                                plt.plot(x, np.mean(demo_torques[i], axis = 1), linewidth=2, label = 'Demo', color='#F22BB2')
                            plt.xlabel('Time step')
                            plt.ylabel('Joint%d'%(i))
                            legend = plt.legend()
                            frame = legend.get_frame()
                            frame.set_facecolor('0.9')
                            frame.set_edgecolor('0.9')
                            plt.savefig(fig_hand_torques_demo[i])
                else: # visualize the errors wrt task goal
                    x_simu = np.arange(0, obj_ori_error_simu.shape[0])
                    plt.figure(2)
                    plt.axes(frameon=0)
                    plt.grid()
                    if error_calc == 'median':  # plot median/percentile 
                        plt.plot(x_simu, np.median(obj_ori_error_simu, axis = 1), linewidth=2, label = 'object position error (simu)', color='#B22400')
                        plt.plot(x_simu, np.median(obj_ori_error_simu_koopman, axis = 1), linewidth=2, label = 'object position error (koopman)', color='#F22BB2')
                        plt.plot(x_simu, np.median(demo_ori_error_simu, axis = 1), linewidth=2, label = 'object position error (RL)', color='#006BB2')
                        plt.fill_between(x_simu, np.percentile(obj_ori_error_simu, 25, axis = 1), np.percentile(obj_ori_error_simu, 75, axis = 1), alpha = 0.15, linewidth = 0, color='#B22400')
                        plt.fill_between(x_simu, np.percentile(obj_ori_error_simu_koopman, 25, axis = 1), np.percentile(obj_ori_error_simu_koopman, 75, axis = 1), alpha = 0.15, linewidth = 0, color='#F22BB2')
                        plt.fill_between(x_simu, np.percentile(demo_ori_error_simu, 25, axis = 1), np.percentile(demo_ori_error_simu, 75, axis = 1), alpha = 0.15, linewidth = 0, color='#006BB2')
                    else:  # plot mean 
                        plt.plot(x_simu, np.mean(obj_ori_error_simu, axis = 1), linewidth=2, label = 'object position error (simu)', color='#B22400')
                        plt.plot(x_simu, np.mean(obj_ori_error_simu_koopman, axis = 1), linewidth=2, label = 'object position error (koopman)', color='#F22BB2')
                        plt.plot(x_simu, np.mean(demo_ori_error_simu, axis = 1), linewidth=2, label = 'object position error (RL)', color='#006BB2')
                    plt.xlabel('Time step')
                    plt.ylabel('Position error wrt goal')
                    legend = plt.legend()
                    frame = legend.get_frame()
                    frame.set_facecolor('0.9')
                    frame.set_edgecolor('0.9')
                    plt.savefig(fig_path_simu_goal)
        else:  # test on unseen data
            if not controller or error_type == 'demo':
                print("Please make sure 'error_type' is set to be 'goal' and enable the controller when testing the learned koopman on unseen data")
                sys.exit()
            else:
                x_simu = np.arange(0, obj_ori_error_simu.shape[0])
                plt.figure(2)
                plt.axes(frameon=0)
                plt.grid()
                if error_calc == 'median':  # plot median/percentile 
                    plt.plot(x_simu, np.median(obj_ori_error_simu, axis = 1), linewidth=2, label = 'object position error (simu_unseen)', color='#B22400')
                    plt.plot(x_simu, np.median(obj_ori_error_simu_koopman, axis = 1), linewidth=2, label = 'object position error (koopman_unseen)', color='#F22BB2')
                    plt.plot(x_simu, np.median(demo_ori_error_simu, axis = 1), linewidth=2, label = 'object position error (RL_unseen)', color='#006BB2')
                    plt.fill_between(x_simu, np.percentile(obj_ori_error_simu, 25, axis = 1), np.percentile(obj_ori_error_simu, 75, axis = 1), alpha = 0.15, linewidth = 0, color='#B22400')
                    plt.fill_between(x_simu, np.percentile(obj_ori_error_simu_koopman, 25, axis = 1), np.percentile(obj_ori_error_simu_koopman, 75, axis = 1), alpha = 0.15, linewidth = 0, color='#F22BB2')
                    plt.fill_between(x_simu, np.percentile(demo_ori_error_simu, 25, axis = 1), np.percentile(demo_ori_error_simu, 75, axis = 1), alpha = 0.15, linewidth = 0, color='#006BB2')
                else:  # plot mean 
                    plt.plot(x_simu, np.mean(obj_ori_error_simu, axis = 1), linewidth=2, label = 'object position error (simu_unseen)', color='#B22400')
                    plt.plot(x_simu, np.mean(obj_ori_error_simu_koopman, axis = 1), linewidth=2, label = 'object position error (koopman_unseen)', color='#F22BB2')
                    plt.plot(x_simu, np.mean(demo_ori_error_simu, axis = 1), linewidth=2, label = 'object position error (RL_unseen)', color='#006BB2')
                plt.xlabel('Time step')
                plt.ylabel('Position error wrt goal')
                legend = plt.legend()
                frame = legend.get_frame()
                frame.set_facecolor('0.9')
                frame.set_edgecolor('0.9')
                plt.savefig(fig_path_simu_goal_unseen)
                with open(''.join(fig_path_tmp[:-1]) + 'success_1.txt', 'w') as f:
                    f.write(success_rate)
                # x = np.arange(0, Computed_joint_unseen.shape[1])
                # for i in range(num_hand):
                #     plt.figure(120 + i)
                #     plt.axes(frameon=0)
                #     plt.grid()
                #     if error_calc == 'median':  # plot median/percentile 
                #         plt.plot(x, np.median(Computed_joint_unseen[i], axis = 1), linewidth=2, label = 'Koopman rollout', color='#B22400')
                #         plt.fill_between(x, np.percentile(Computed_joint_unseen[i], 25, axis = 1), np.percentile(Computed_joint_unseen[i], 75, axis = 1), alpha = 0.15, linewidth = 0, color='#B22400')
                #         plt.plot(x, np.median(RL_joint_unseen[i], axis = 1), linewidth=2, label = 'Demo', color='#F22BB2')
                #         plt.fill_between(x, np.percentile(RL_joint_unseen[i], 25, axis = 1), np.percentile(RL_joint_unseen[i], 75, axis = 1), alpha = 0.15, linewidth = 0, color='#F22BB2')
                #     else:  # plot mean 
                #         plt.plot(x, np.mean(Computed_joint_unseen[i], axis = 1), linewidth=2, label = 'Koopman rollout', color='#B22400')
                #         plt.plot(x, np.mean(RL_joint_unseen[i], axis = 1), linewidth=2, label = 'Demo', color='#F22BB2')
                #     plt.xlabel('Time step')
                #     plt.ylabel('Joint%d'%(i))
                #     legend = plt.legend()
                #     frame = legend.get_frame()
                #     frame.set_facecolor('0.9')
                #     frame.set_edgecolor('0.9')
                #     plt.savefig(fig_hand_joints_unseen_demo[i])
        # plt.show()
        print("Finish the evaluation!")
        sys.exit()
    '''
    Above: Test the trained koopman dynamics
    Below: Train the koopman dynamics from demo data
    '''
    # record the training errors
    # Drafted: error over num of used trajectories 
    # Start the training process for each mode
    if unit_train:
        parent_dir = './Results/' + koopmanoption + folder_name
        current_time = 'seperate' + str(separate) + '_' + 'seed' + str(seed)
        # current_time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        os.mkdir(os.path.join(parent_dir, current_time))
        results_record = open('./Results/' + koopmanoption + folder_name + current_time + '/results.txt', 'w+')
    else:
        if not os.path.exists(os.path.join(folder_name, "NDP")):
            os.mkdir(os.path.join(folder_name, "NDP"))
        else:
            shutil.rmtree(os.path.join(folder_name, "NDP"))   # Removes all the subdirectories!
            os.mkdir(os.path.join(folder_name, "NDP"))
        results_record = open(folder_name + "NDP" + '/results.txt', 'w+')
    start_time = time.time()
    if koopmanoption == "Drafted":  # define the BC agent
        k = 1 # only one rollout (only one input image, but the whole traj with T steps)
        # the number of rollout = the number of DMP/NDP trained in each traj.
        # if we only have one rollout, only one NDP, so all the demo data along each traj is used to train the param for this NDP.
        T = int(len(Training_data[0])/k) # simulation steps
        batch_size = 4
        lr = 0.001
        iter = 200
        # define the input and outputs of NDPs    
        NN_Input_size = num_obj + num_hand 
        separate = 2 # large model for training
        if int(separate) == 0:
            N = 10
            hidden_layer = [NN_Input_size, 32, 64, 32]
        elif int(separate) == 1:
            N = 20
            hidden_layer = [NN_Input_size, 64, 128, 64]
        else:
            N = 30
            hidden_layer = [NN_Input_size, 128, 256, 128]
        policy = NdpCnn(T=T,l=1, N=N, state_index=np.arange(NN_Input_size), layer_sizes = hidden_layer, seed = seed)
        trainable_params = list(policy.parameters())
        # for i in trainable_params:
        #     print(i.shape)
        num_params = sum(p.numel() for p in trainable_params if p.requires_grad)
        # print model #params
        print("model #params: %d" % num_params)
        Raw_Input_data = np.zeros([first_demo, len(Training_data[0]), NN_Input_size])  
        Validation_input_data = np.zeros([50, len(Training_data[0]), NN_Input_size])  
        NDP_loss = []
        Best_loss = 10000
        print("NDP training starts!\n")
        for k in tqdm(range(first_demo)):
            for t in range(len(Training_data[k])):
                if Velocity:
                    hand_OriState = np.append(Training_data[k][t]['handpos'], Training_data[k][t]['handvel'])
                else:
                    hand_OriState = Training_data[k][t]['handpos']
                obj_OriState = np.append(np.append(Training_data[k][t]['objpos'], np.append(Training_data[k][t]['objorient'], Training_data[k][t]['objvel'])), Training_data[k][t]['nail_goal'])
                Raw_Input_data[k, t] = np.append(hand_OriState, obj_OriState)
        for k in tqdm(range(200, 250)):
            for t in range(len(Training_data[k])):
                if Velocity:
                    hand_OriState = np.append(Training_data[k][t]['handpos'], Training_data[k][t]['handvel'])
                else:
                    hand_OriState = Training_data[k][t]['handpos']
                obj_OriState = np.append(np.append(Training_data[k][t]['objpos'], np.append(Training_data[k][t]['objorient'], Training_data[k][t]['objvel'])), Training_data[k][t]['nail_goal'])
                Validation_input_data[k - 200, t] = np.append(hand_OriState, obj_OriState)
        Raw_Output_data = Raw_Input_data.copy()
        Validation_output_data = Validation_input_data.copy()
        batch_iter = int(Raw_Input_data.shape[0] / batch_size)
        shuffle_index = np.arange(0, batch_iter * batch_size)
        for t in tqdm(range(iter)):
            # if t != 0 and t % 200 == 0:
            #     lr *= 0.9
            print("This is %d iteration and the learning rate is %f."%(t, lr))
            results_record.write("This is %d iteration and the learning rate is %f.\n"%(t, lr))
            np.random.seed(t)  # fixed the initial seed value for reproducible results
            np.random.shuffle(shuffle_index)     
            policy.train()
            optimizer = torch.optim.Adam(trainable_params, lr=lr)  # try with momentum term
            for mb in range(batch_iter):
                rand_idx = shuffle_index[mb * batch_size: (mb + 1)* batch_size]
                batch_input = torch.from_numpy(Raw_Input_data[rand_idx])
                batch_label = torch.from_numpy(Raw_Output_data[rand_idx])
                optimizer.zero_grad()  
                batch_output = policy(batch_input[:, 0], batch_input[:, 0]) # batch_output -> [batch_size, T, dim]
                # batch_output is the full trajectory 
                loss_criterion = torch.nn.MSELoss()
                loss = loss_criterion(batch_output, batch_label.detach().to(torch.float32))
                loss.backward()
                optimizer.step()
            policy.eval() # Evaluate the current model
            eval_input = torch.from_numpy(Validation_input_data)
            eval_label = torch.from_numpy(Validation_output_data)
            eval_output = policy(eval_input[:, 0], eval_input[:, 0]) # batch_output -> [batch_size, T, dim]
            loss_criterion = torch.nn.MSELoss()
            eval_loss = loss_criterion(eval_output, eval_label.detach().to(torch.float32)).item()
            NDP_loss.append(eval_loss)
            print("After %d iteration, the total loss on the training demo is %f.\n"%(t, eval_loss))
            results_record.write("After %d iteration, the total loss on the training demo is %f.\n"%(t, eval_loss))
            results_record.flush()
            if save_matrix:
                if eval_loss < Best_loss:
                    Best_loss = eval_loss
                    if unit_train:
                        torch.save(policy.state_dict(), './Results/' + koopmanoption + folder_name + current_time + '/NDP_agent.pt')
                    else:
                        torch.save(policy.state_dict(), folder_name + "NDP" + '/NDP_agent.pt')
        print("NDP training ends!\n")      
    results_record.write("Training time is: %f seconds.\n"%(time.time() - start_time))   
    results_record.close()
    plt.figure(1)
    plt.axes(frameon=0)
    plt.grid()
    plt.plot(NDP_loss, linewidth=2, label = 'NDP agent Loss', color='#B22400')
    plt.xlabel('Training iteration')
    plt.ylabel('Loss')
    legend = plt.legend()
    frame = legend.get_frame()
    frame.set_facecolor('0.9')
    frame.set_edgecolor('0.9')
    if unit_train:
        plt.savefig('./Results/' + koopmanoption + folder_name + current_time + '/Training_loss.png')
    else:
        plt.savefig(folder_name + "NDP" + '/Training_loss.png')

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def demo_playback(env_name, demo_paths, num_demo):
    e = GymEnv(env_name)
    e.reset()
    Training_data = []
    state_dict = {}
    print("Begin loading demo data!")
    print(num_demo)
    sample_index = np.arange(num_demo)
    Selected_timesteps = 31  # reduced horizon (only for test, for training, we use full-horizon data, T = 100)
    for t in tqdm(sample_index):
        path = demo_paths[t]
        path_data = []
        e.set_env_state(path['init_state_dict'])
        actions = path['actions']
        observations = path['observations']  
        observations_visualize = path['observations_visualization'] 
        handVelocity = path['handVelocity'] 
        for t in range(Selected_timesteps):
            tmp = dict()
            if t == 0:
                tmp['init'] = path['init_state_dict']
            obs = observations[t] 
            obs_visual = observations_visualize[t]
            allvel = handVelocity[t] 
            handpos = obs[:26]  
            tmp['handpos'] = handpos
            tmp['handvel'] = allvel[:26]
            objpos = obs[49:52] + obs[42:45] 
            tmp['objpos'] = objpos 
            tmp['objorient'] = obs[39:42] 
            tmp['objvel'] = obs[27:33]
            tmp['nail_goal'] = obs[46:49] 
            tmp['observation'] = obs[49:52]
            tmp['action'] = actions[t]
            if False:
                e.env.mj_render()  # render the visualizer
            path_data.append(tmp)
        Training_data.append(path_data)
    return Training_data

def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class NdpCnn(nn.Module):
    # ptu.fanin_init -> uniform distribution
    def __init__(
            self,
            init_w=3e-3,
            layer_sizes=[784, 200, 100],
            hidden_activation=F.relu,
            output_activation=None,
            hidden_init=fanin_init, 
            b_init_value=0.1,
            state_index=np.arange(1),
            N = 5, 
            T = 10, 
            l = 10,
            seed = 0,
            *args,
            **kwargs
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.N = N 
        self.l = l
        self.input_size = len(state_index)
        self.output_size = N*len(state_index) + 2*len(state_index) # 30 * 2 + 2 * 2 = 64
        # output_size should be: omage + goal (params for NDP)
        output_size = self.output_size
        self.T = T
        self.output_activation=torch.tanh
        self.state_index = state_index
        self.output_dim = output_size
        tau = 1
        dt = 0.01  # read from Mujuco env
        self.output_activation=torch.tanh
        self.DMPparam = DMPParameters(N, tau, dt, len(state_index), None, T)  # non-trainable params
        self.func = DMPIntegrator()
        self.register_buffer('DMPp', self.DMPparam.data_tensor)
        self.register_buffer('param_grad', self.DMPparam.grad_tensor)
        # we can fine-tine the layer sizes - NN to learn the parameters for dynamics policy
        self.hidden_activation = hidden_activation
        self.N = N
        self.middle_layers = []
        for i in range(len(layer_sizes)-1): 
            layer = torch.nn.Linear(layer_sizes[i], layer_sizes[i+1])
            hidden_init(layer.weight)
            layer.bias.data.fill_(b_init_value)
            self.middle_layers.append(layer)
            self.add_module("middle_layer_" + str(i), layer)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.last_fc = init_(nn.Linear(layer_sizes[-1], output_size)) # output_size -> output_size

    def forward(self, input, y0, return_preactivations=False):
        x = input
        x = x.view(-1, self.input_size).float()
        activation_fn = self.hidden_activation  # F.relu
        # x.shape = (batch_size, self.input_size)
        for layer in self.middle_layers:
            x = activation_fn(layer(x))
        # x.size = output_size = N*len(state_index) + 2*len(state_index)
        output = self.last_fc(x)*1000  
        y0 = y0.reshape(-1, 1)[:, 0] # y0.shape -> [dim*batch_size]
        dy0 = torch.zeros_like(y0) + 0.01 # set to be small values
        # self.DMPp -> DMP parameters, including: \alpha, \beta, +a_x
        y, dy, ddy = self.func.forward(output, self.DMPp, self.param_grad, None, y0, dy0) # y -> # [batch_size * dim, T]
        y = y.view(input.shape[0], len(self.state_index), -1) # project back to [batch_size, dim, T]
        return y.transpose(2, 1) # [batch_size, T, dim]

    def execute(self, input, y0, return_preactivations=False):
        x = input
        x = x.view(-1, self.input_size).float()
        activation_fn = self.hidden_activation  # F.relu
        # x.shape = (batch_size, self.input_size)
        for layer in self.middle_layers:
            x = activation_fn(layer(x))
        # x.size = output_size = N*len(state_index) + 2*len(state_index)
        output = self.last_fc(x)*1000  
        y0 = y0.reshape(-1, 1)[:, 0] # y0.shape -> [dim*batch_size]
        dy0 = torch.zeros_like(y0) + 0.01 # set to be small values
        # self.DMPp -> DMP parameters, including: \alpha, \beta, a_x
        y, dy, ddy = self.func.forward(output, self.DMPp, self.param_grad, None, y0, dy0) # y -> # [batch_size * dim, T]
        return y.transpose(1, 0).detach().numpy() # [batch_size, T, dim]
if __name__ == '__main__':
    main()

