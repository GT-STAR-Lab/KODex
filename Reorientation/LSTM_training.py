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
    Controller_loc = 'Results/Controller/NN_controller_best.pt'
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
        folder_name = '/obj_in_world_Transformed_frame_diff_goals_1/LSTM_cloning/parameter_tuning/'  # define the folder name for training
        unit_train = True
    if not os.path.exists(matrix_file):  # loading data for training
        Training_data = demo_playback(env_name, demos, num_demo) # len(Training_data) = num_demo
        num_handpos = len(Training_data[0][0]['handpos'])
        num_handvel = len(Training_data[0][0]['handvel'])
        num_objpos = len(Training_data[0][0]['objpos']) + len(Training_data[0][0]['objorient'])
        num_objvel = len(Training_data[0][0]['objvel'])
        dt = 1  # simulation time for each step, assuming it should be 1
        num_obj = num_objpos + num_objvel
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
            Training_data = e.generate_unseen_data(number_sample)  
            num_handpos = len(Training_data[0]['handpos'])
            num_handvel = len(Training_data[0]['handvel'])
            num_objpos = len(Training_data[0]['objpos']) + len(Training_data[0]['objorient'])
            num_objvel = len(Training_data[0]['objvel'])
            num_obj = num_objpos + num_objvel
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
            dt = 1  # simulation time for each step, assuming it should be 1
            num_obj = num_objpos + num_objvel
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
        # define the input and outputs of the neural network(LSTM)    
        NN_Input_size = num_obj + num_hand 
        NN_Output_size = NN_Input_size
        separate = 0 # small model for testing
        if int(separate) == 0:
            LSTM_policy = RNN(NN_Input_size, NN_Output_size, input_sizes=100, hidden_state=200, LSTM_layer=1)   
        elif int(separate) == 1:
            LSTM_policy = RNN(NN_Input_size, NN_Output_size, input_sizes=175, hidden_state=250, LSTM_layer=1)   
        else:
            LSTM_policy = RNN(NN_Input_size, NN_Output_size, input_sizes=250, hidden_state=300, LSTM_layer=1)   
        LSTM_policy.model.load_state_dict(torch.load(matrix_file))
        LSTM_policy.model.eval()
        fig_path_tmp = [ele + '/' for ele in matrix_file.split('/')]
        fig_path_visual_demo = ''.join(fig_path_tmp[:-1]) + 'RolloutError_visual(DemoError).png'
        fig_path_simu_demo = ''.join(fig_path_tmp[:-1]) + 'RolloutError_simu(DemoError).png'
        fig_path_simu_goal = ''.join(fig_path_tmp[:-1]) + 'RolloutError_simu(GoalError).png'
        current_time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        fig_path_simu_goal_unseen = ''.join(fig_path_tmp[:-1]) + 'RolloutError_simu_unseen(GoalError)_' + current_time + '.png'
        fig_hand_joints_demo = [''.join(fig_path_tmp[:-1]) + 'Joint/joints_' + str(i) + '.png' for i in range(num_hand)]
        fig_hand_torques_demo = [''.join(fig_path_tmp[:-1]) + 'Torque/torques_' + str(i) + '.png' for i in range(num_hand)]
        fig_hand_joints_tracking_demo = [''.join(fig_path_tmp[:-1]) + 'Joint_tracking/torques_' + str(i) + '.png' for i in range(num_hand)]
        if koopmanoption == "Drafted":
            print("Trained drafted koopman matrix loaded!")
            if not Unseen_test:
                errors = LSTM_error_visualization(env_name, LSTM_policy, Training_data, Velocity, num_hand, num_obj, koopmanoption, error_type, False)
                if error_type == 'demo':
                    hand_pos_error = errors[0]
                    computed_joints = errors[1]
                    demo_joints = errors[2]
                if controller:
                    errors_simu = LSTM_policy_control(env_name, Controller, LSTM_policy, Training_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize)
                    if error_type == 'demo':
                        hand_pos_error_simu = errors_simu[0]
                        hand_pos_error_PID_simu = errors_simu[1]
                        computed_torques = errors_simu[2]
                        demo_torques = errors_simu[3]
                        Koopman_joints = errors_simu[4]
                        NN_joints = errors_simu[5]
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
                    obj_ori_error_simu, obj_ori_error_simu_koopman, success_rate = LSTM_policy_control_unseenTest(env_name, Controller, LSTM_policy, Training_data, Velocity, num_hand, num_obj, koopmanoption, Visualize)      
                    demo_ori_error_simu, success_rate_RL = e.visualize_policy_on_demos(rl_policy, Training_data, Visualize, e.horizon)
        # plot the figures
        if not Unseen_test:  # on demo data                 
            if error_type == 'demo':  # visualize the errors wrt demo data (pure visualization)
                x = np.arange(0, hand_pos_error.shape[0])
                plt.figure(1)
                plt.axes(frameon=0)
                plt.grid()
                if error_calc == 'median':  # plot median/percentile 
                    plt.plot(x, np.median(hand_pos_error, axis = 1), linewidth=2, label = 'hand joint error (koopman)', color='#B22400')
                    plt.fill_between(x, np.percentile(hand_pos_error, 25, axis = 1), np.percentile(hand_pos_error, 75, axis = 1), alpha = 0.15, linewidth = 0, color='#B22400')
                else:  # plot mean 
                    plt.plot(x, np.mean(hand_pos_error, axis = 1), linewidth=2, label = 'hand joint error', color='#B22400')
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
                        if  error_calc == 'median':  # plot median/percentile 
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
                    x_simu = np.arange(0, hand_pos_error_simu.shape[0])
                    plt.figure(2)
                    plt.axes(frameon=0)
                    plt.grid()
                    if error_calc == 'median':  # plot median/percentile 
                        plt.plot(x_simu, np.median(hand_pos_error_simu, axis = 1), linewidth=2, label = 'hand joint error (simu)', color='#B22400')
                        plt.fill_between(x_simu, np.percentile(hand_pos_error_simu, 25, axis = 1), np.percentile(hand_pos_error_simu, 75, axis = 1), alpha = 0.15, linewidth = 0, color='#B22400')
                        plt.plot(x_simu, np.median(hand_pos_error, axis = 1), linewidth=2, label = 'hand joint error (koopman)', color='#F22BB2')
                        plt.fill_between(x_simu, np.percentile(hand_pos_error, 25, axis = 1), np.percentile(hand_pos_error, 75, axis = 1), alpha = 0.15, linewidth = 0, color='#F22BB2')
                        plt.plot(x_simu, np.median(hand_pos_error_PID_simu, axis = 1), linewidth=2, label = 'hand joint error (PID tracking)', color='#006BB2')
                        plt.fill_between(x_simu, np.percentile(hand_pos_error_PID_simu, 25, axis = 1), np.percentile(hand_pos_error_PID_simu, 75, axis = 1), alpha = 0.15, linewidth = 0, color='#006BB2')
                    else:  # plot mean 
                        plt.plot(x_simu, np.mean(hand_pos_error_simu, axis = 1), linewidth=2, label = 'hand joint error (simu)', color='#B22400')
                        plt.plot(x_simu, np.mean(hand_pos_error, axis = 1), linewidth=2, label = 'hand joint error (koopman)', color='#F22BB2')
                        plt.plot(x_simu, np.mean(hand_pos_error_PID_simu, axis = 1), linewidth=2, label = 'hand joint error (PID tracking)', color='#006BB2')
                    plt.xlabel('Time step')
                    plt.ylabel('Difference error wrt demo')
                    legend = plt.legend()
                    frame = legend.get_frame()
                    frame.set_facecolor('0.9')
                    frame.set_edgecolor('0.9')
                    plt.savefig(fig_path_simu_demo)
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
                        x = np.arange(0, Koopman_joints.shape[1])
                        for i in range(num_hand):
                            plt.figure(110 + i)
                            plt.axes(frameon=0)
                            plt.grid()
                            if error_calc == 'median':  # plot median/percentile 
                                plt.plot(x, np.median(Koopman_joints[i], axis = 1), linewidth=2, label = 'Koopman_Joint', color='#B22400')
                                plt.fill_between(x, np.percentile(Koopman_joints[i], 25, axis = 1), np.percentile(Koopman_joints[i], 75, axis = 1), alpha = 0.15, linewidth = 0, color='#B22400')
                                plt.plot(x, np.median(NN_joints[i], axis = 1), linewidth=2, label = 'NN_Joint', color='#F22BB2')
                                plt.fill_between(x, np.percentile(NN_joints[i], 25, axis = 1), np.percentile(NN_joints[i], 75, axis = 1), alpha = 0.15, linewidth = 0, color='#F22BB2')
                            else:  # plot mean 
                                plt.plot(x, np.mean(Koopman_joints[i], axis = 1), linewidth=2, label = 'Koopman_Joint', color='#B22400')
                                plt.plot(x, np.mean(NN_joints[i], axis = 1), linewidth=2, label = 'NN_Joint', color='#F22BB2')
                            plt.xlabel('Time step')
                            plt.ylabel('Joint%d'%(i))
                            legend = plt.legend()
                            frame = legend.get_frame()
                            frame.set_facecolor('0.9')
                            frame.set_edgecolor('0.9')
                            plt.savefig(fig_hand_joints_tracking_demo[i])
                else: # visualize the errors wrt task goal
                    x_simu = np.arange(0, obj_ori_error_simu.shape[0])
                    plt.figure(2)
                    plt.axes(frameon=0)
                    plt.grid()
                    if error_calc == 'median':  # plot median/percentile 
                        plt.plot(x_simu, np.median(obj_ori_error_simu, axis = 1), linewidth=2, label = 'object orientation error (simu)', color='#B22400')
                        plt.plot(x_simu, np.median(obj_ori_error_simu_koopman, axis = 1), linewidth=2, label = 'object orientation error (koopman)', color='#F22BB2')
                        plt.plot(x_simu, np.median(demo_ori_error_simu, axis = 1), linewidth=2, label = 'object orientation error (RL)', color='#006BB2')
                        plt.fill_between(x_simu, np.percentile(obj_ori_error_simu, 25, axis = 1), np.percentile(obj_ori_error_simu, 75, axis = 1), alpha = 0.15, linewidth = 0, color='#B22400')
                        plt.fill_between(x_simu, np.percentile(obj_ori_error_simu_koopman, 25, axis = 1), np.percentile(obj_ori_error_simu_koopman, 75, axis = 1), alpha = 0.15, linewidth = 0, color='#F22BB2')
                        plt.fill_between(x_simu, np.percentile(demo_ori_error_simu, 25, axis = 1), np.percentile(demo_ori_error_simu, 75, axis = 1), alpha = 0.15, linewidth = 0, color='#006BB2')
                    else:  # plot mean 
                        plt.plot(x_simu, np.mean(obj_ori_error_simu, axis = 1), linewidth=2, label = 'object orientation error (simu)', color='#B22400')
                        plt.plot(x_simu, np.mean(obj_ori_error_simu_koopman, axis = 1), linewidth=2, label = 'object orientation error (koopman)', color='#F22BB2')
                        plt.plot(x_simu, np.mean(demo_ori_error_simu, axis = 1), linewidth=2, label = 'object orientation error (RL)', color='#006BB2')
                    plt.xlabel('Time step')
                    plt.ylabel('Orientation error wrt goal')
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
                    plt.plot(x_simu, np.median(obj_ori_error_simu, axis = 1), linewidth=2, label = 'object orientation error (simu_unseen)', color='#B22400')
                    plt.plot(x_simu, np.median(obj_ori_error_simu_koopman, axis = 1), linewidth=2, label = 'object orientation error (koopman_unseen)', color='#F22BB2')
                    plt.plot(x_simu, np.median(demo_ori_error_simu, axis = 1), linewidth=2, label = 'object orientation error (RL_unseen)', color='#006BB2')
                    plt.fill_between(x_simu, np.percentile(obj_ori_error_simu, 25, axis = 1), np.percentile(obj_ori_error_simu, 75, axis = 1), alpha = 0.15, linewidth = 0, color='#B22400')
                    plt.fill_between(x_simu, np.percentile(obj_ori_error_simu_koopman, 25, axis = 1), np.percentile(obj_ori_error_simu_koopman, 75, axis = 1), alpha = 0.15, linewidth = 0, color='#F22BB2')
                    plt.fill_between(x_simu, np.percentile(demo_ori_error_simu, 25, axis = 1), np.percentile(demo_ori_error_simu, 75, axis = 1), alpha = 0.15, linewidth = 0, color='#006BB2')
                else:  # plot mean 
                    plt.plot(x_simu, np.mean(obj_ori_error_simu, axis = 1), linewidth=2, label = 'object orientation error (simu_unseen)', color='#B22400')
                    plt.plot(x_simu, np.mean(obj_ori_error_simu_koopman, axis = 1), linewidth=2, label = 'object orientation error (koopman_unseen)', color='#F22BB2')
                    plt.plot(x_simu, np.mean(demo_ori_error_simu, axis = 1), linewidth=2, label = 'object orientation error (RL_unseen)', color='#006BB2')
                plt.xlabel('Time step')
                plt.ylabel('Orientation error wrt goal')
                legend = plt.legend()
                frame = legend.get_frame()
                frame.set_facecolor('0.9')
                frame.set_edgecolor('0.9')
                plt.savefig(fig_path_simu_goal_unseen)
                with open(''.join(fig_path_tmp[:-1]) + 'success_1.txt', 'w') as f:
                    f.write(success_rate)
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
        if not os.path.exists(os.path.join(folder_name, "LSTM")):
            os.mkdir(os.path.join(folder_name, "LSTM"))
        else:
            shutil.rmtree(os.path.join(folder_name, "LSTM"))   # Removes all the subdirectories!
            os.mkdir(os.path.join(folder_name, "LSTM"))
        results_record = open(folder_name + "LSTM" + '/results.txt', 'w+')
    start_time = time.time()
    if koopmanoption == "Drafted":  # define the BC agent
        batch_size = 4
        lr = 0.001
        iter = 200 
        # define the input and outputs of the neural network(LSTM)    
        NN_Input_size = num_obj + num_hand 
        NN_Output_size = NN_Input_size
        separate = 0 # small model for training
        if int(separate) == 0:
            policy = RNN(NN_Input_size, NN_Output_size, input_sizes=100, hidden_state=200, LSTM_layer=1, seed = seed)   
        elif int(separate) == 1:
            policy = RNN(NN_Input_size, NN_Output_size, input_sizes=175, hidden_state=250, LSTM_layer=1, seed = seed)   
        else:
            policy = RNN(NN_Input_size, NN_Output_size, input_sizes=250, hidden_state=300, LSTM_layer=1, seed = seed)   
        # input_sizes, hidden_states, LSTM_layer are the hyperparamters for LSTM
        trainable_params = policy.trainable_params
        num_params = sum(p.numel() for p in trainable_params if p.requires_grad)
        # print model #params
        print("model #params: %d" % num_params)
        Raw_Input_data = np.zeros([first_demo, len(Training_data[0]) - 1, NN_Input_size])  
        Raw_Output_data = np.zeros([first_demo, len(Training_data[0]) - 1, NN_Input_size])  
        Validation_input_data = np.zeros([50, len(Training_data[0]) - 1, NN_Input_size])  
        Validation_output_data = np.zeros([50, len(Training_data[0]) - 1, NN_Input_size])  
        LSTM_loss = []
        Best_loss = 10000
        print("LSTM training starts!\n")
        for k in tqdm(range(first_demo)):
            for t in range(len(Training_data[k]) - 1):
                if Velocity:
                    hand_OriState = np.append(Training_data[k][t]['handpos'], Training_data[k][t]['handvel'])
                    hand_oriState_ = np.append(Training_data[k][t + 1]['handpos'], Training_data[k][t + 1]['handvel'])
                else:
                    hand_OriState = Training_data[k][t]['handpos']
                    hand_oriState_ = Training_data[k][t + 1]['handpos']
                obj_OriState = np.append(Training_data[k][t]['objpos'], np.append(Training_data[k][t]['objorient'], Training_data[k][t]['objvel']))
                obj_OriState_ = np.append(Training_data[k][t + 1]['objpos'], np.append(Training_data[k][t + 1]['objorient'], Training_data[k][t + 1]['objvel']))
                Raw_Input_data[k, t] = np.append(hand_OriState, obj_OriState)
                Raw_Output_data[k, t] = np.append(hand_oriState_, obj_OriState_)
        for k in tqdm(range(200, 250)):
            for t in range(len(Training_data[k]) - 1):
                if Velocity:
                    hand_OriState = np.append(Training_data[k][t]['handpos'], Training_data[k][t]['handvel'])
                    hand_oriState_ = np.append(Training_data[k][t + 1]['handpos'], Training_data[k][t + 1]['handvel'])
                else:
                    hand_OriState = Training_data[k][t]['handpos']
                    hand_oriState_ = Training_data[k][t + 1]['handpos']
                obj_OriState = np.append(Training_data[k][t]['objpos'], np.append(Training_data[k][t]['objorient'], Training_data[k][t]['objvel']))
                obj_OriState_ = np.append(Training_data[k][t + 1]['objpos'], np.append(Training_data[k][t + 1]['objorient'], Training_data[k][t + 1]['objvel']))
                Validation_input_data[k - 200, t] = np.append(hand_OriState, obj_OriState)
                Validation_output_data[k - 200, t] = np.append(hand_oriState_, obj_OriState_)
        batch_iter = int(Raw_Input_data.shape[0] / batch_size)
        shuffle_index = np.arange(0, batch_iter * batch_size)
        for t in tqdm(range(iter)):
            # if t != 0 and t % 200 == 0:
            #     lr *= 0.9
            print("This is %d iteration and the learning rate is %f."%(t, lr))
            results_record.write("This is %d iteration and the learning rate is %f.\n"%(t, lr))
            np.random.seed(t)  # fixed the initial seed value for reproducible results
            np.random.shuffle(shuffle_index)     
            policy.model.train()
            for mb in range(batch_iter):
                rand_idx = shuffle_index[mb * batch_size: (mb + 1)* batch_size]
                batch_input = torch.from_numpy(Raw_Input_data[rand_idx])
                batch_label = torch.from_numpy(Raw_Output_data[rand_idx])
                optimizer = torch.optim.Adam(trainable_params, lr=lr)  # try with momentum term
                optimizer.zero_grad()  
                loss = loss_func(policy, batch_input, batch_label)
                # all_loss:[Encoder_loss_torch, Encoder_loss_numpy, Pred_loss_torch, Pred_loss_numpy]
                loss.backward()
                optimizer.step()
            policy.model.eval() # Evaluate the current model
            eval_input = torch.from_numpy(Validation_input_data)
            eval_label = torch.from_numpy(Validation_output_data)
            eval_loss = eval_func(policy, eval_input, eval_label)
            LSTM_loss.append(eval_loss)
            print("After %d iteration, the total loss on the training demo is %f.\n"%(t, eval_loss))
            results_record.write("After %d iteration, the total loss on the training demo is %f.\n"%(t, eval_loss))
            results_record.flush()
            if save_matrix:
                if eval_loss < Best_loss:
                    Best_loss = eval_loss
                    if unit_train:
                        torch.save(policy.model.state_dict(), './Results/' + koopmanoption + folder_name + current_time + '/LSTM_agent.pt')
                    else:
                        torch.save(policy.model.state_dict(), folder_name + "LSTM" + '/LSTM_agent.pt')
        print("LSTM training ends!\n")  
    results_record.write("Training time is: %f seconds.\n"%(time.time() - start_time))
    results_record.close()
    plt.figure(1)
    plt.axes(frameon=0)
    plt.grid()
    plt.plot(LSTM_loss, linewidth=2, label = 'LSTM agent Loss', color='#B22400')
    plt.xlabel('Training iteration')
    plt.ylabel('Loss')
    legend = plt.legend()
    frame = legend.get_frame()
    frame.set_facecolor('0.9')
    frame.set_edgecolor('0.9')
    if unit_train:
        plt.savefig('./Results/' + koopmanoption + folder_name + current_time + '/Training_loss.png')
    else:
        plt.savefig(folder_name + "LSTM" + '/Training_loss.png')

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def loss_func(LSTM_policy, batch_input, batch_label):
    loss_criterion = torch.nn.MSELoss()
    batch_output = LSTM_policy.model(batch_input)
    loss = loss_criterion(batch_output, batch_label.detach().to(torch.float32))
    return loss

def eval_func(LSTM_policy, eval_input, eval_label):
    loss_criterion = torch.nn.MSELoss()
    batch_output = LSTM_policy.model(eval_input)
    loss = loss_criterion(batch_output, eval_label.detach().to(torch.float32)).item()
    return loss

def demo_playback(env_name, demo_paths, num_demo):
    e = GymEnv(env_name)
    e.reset()
    Training_data = []
    state_dict = {}
    print("Begin loading demo data!")
    print(num_demo)
    sample_index = np.arange(num_demo)
    for t in tqdm(sample_index):
        path = demo_paths[t]
        path_data = []
        e.set_env_state(path['init_state_dict'])
        actions = path['actions']
        observations = path['observations']  
        observations_visualization = path['observations_visualization']
        handVelocity = path['handVelocity']  
        for t in range(len(actions)):
            tmp = dict()
            obs = observations[t] 
            obs_visual = observations_visualization[t] 
            state_dict['desired_orien'] = quat2euler(path['init_state_dict']['desired_orien'])
            state_dict['qpos'] = obs_visual[:30]
            state_dict['qvel'] = obs_visual[30:]
            handpos = obs[:24]
            tmp['handpos'] = handpos
            handvel = handVelocity[t]
            tmp['handvel'] = handvel
            objpos = obs[24:27]
            tmp['objpos'] = objpos
            objvel = obs[27:33] 
            tmp['objvel'] = objvel
            tmp['desired_ori'] = obs[36:39] 
            objorient = obs[33:36] 
            tmp['objorient'] = ori_transform(objorient, tmp['desired_ori']) 
            tmp['observation'] = obs[42:45]  
            tmp['action'] = actions[t]
            tmp['pen_desired_orien'] = path['desired_orien']
            if False:
                e.env.mj_render()  # render the visualizer
            path_data.append(tmp)
        Training_data.append(path_data)
    return Training_data
if __name__ == '__main__':
    main()

