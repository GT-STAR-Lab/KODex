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
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.linalg import logm
import scipy
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
@click.option('--control', type=str, help='apply with a controller', default='')
@click.option('--error_type', type=str, help='define how to calculate the errors', default='demo') # two options: demo; goal
@click.option('--visualize', type=str, help='define whether or not to visualze the manipulation results', default='') # two options: demo; goal
@click.option('--unseen_test', type=str, help='define if we generate unseen data for testing the learned dynamics model', default='') # generate new testing data
@click.option('--rl_policy', type=str, help='define the file location of the well-trained policy', default='') 
@click.option('--folder_name', type=str, help='define the location to put trained models', default='') 
@click.option('--separate', type=str, help='For comparing the number of observables, we separate the test by storing the results locally', default='') 

def main(env_name, demo_file, num_demo, koopmanoption, velocity, save_matrix, matrix_file, control, error_type, visualize, unseen_test, rl_policy, folder_name, separate):
    num_demo = int(num_demo) # if num_demo != 0 -> we manually define the num of demo for testing the sample efficiency
    Velocity = True if velocity == 'True' else False
    save_matrix = True if save_matrix == 'True' else False  # save the Koopman matrix for Drafted method and trained model for MLP/GNN
    controller = True if control == 'True' else False
    Visualize = True if visualize == 'True' else False
    Unseen_test = True if unseen_test == 'True' else False
    Separate = True if separate == 'True' else False
    number_sample = 200  # num of unseen samples used for test
    Controller_loc = 'Results/Controller/NN_controller_best.pt'
    multiple_test = True  # if we are going to have 
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
    if not os.path.exists(matrix_file):  # loading data for training
        Training_data = demo_playback(env_name, demos, num_demo) # len(Training_data) = num_demo
        num_handpos = len(Training_data[0][0]['handpos'])
        num_handvel = len(Training_data[0][0]['handvel'])
        num_objpos = len(Training_data[0][0]['objpos'])
        num_objvel = len(Training_data[0][0]['objvel'])
        num_init_pos = len(Training_data[0][0]['handle_init'])
        dt = 1  # simulation time for each step, assuming it should be 1
        num_obj = num_objpos + num_objvel + num_init_pos
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
            Training_data = e.generate_unseen_data_door(number_sample)  
            num_handpos = len(Training_data[0]['handpos'])
            num_handvel = len(Training_data[0]['handvel'])
            num_objpos = len(Training_data[0]['objpos'])
            num_objvel = len(Training_data[0]['objvel'])
            num_init_pos = len(Training_data[0]['handle_init'])
            num_obj = num_objpos + num_objvel + num_init_pos
            if Velocity: # hand velocities are also used as the original states
                num_hand = num_handpos + num_handvel
            else:
                num_hand = num_handpos
        else: # test the performance on demo data
            Training_data = demo_playback(env_name, demos, len(demos)) # Test on all the demo data
            num_handpos = len(Training_data[0][0]['handpos'])
            num_handvel = len(Training_data[0][0]['handvel'])
            num_objpos = len(Training_data[0][0]['objpos'])
            num_objvel = len(Training_data[0][0]['objvel'])
            num_init_pos = len(Training_data[0][0]['handle_init'])
            dt = 1  # simulation time for each step, assuming it should be 1
            num_obj = num_objpos + num_objvel + num_init_pos
            if Velocity: # hand velocities are also used as the original states
                num_hand = num_handpos + num_handvel
            else:
                num_hand = num_handpos
        # define the input and outputs of the neural network
        NN_Input_size = 2 * num_hand
        NN_Output_size = num_hand
        NN_size = [NN_Input_size, 2 * NN_Input_size, 2 * NN_Input_size, NN_Input_size, NN_Output_size]  # define a neural network to learn the PID mapping
        Controller = FCNetwork(NN_size, nonlinearity='relu')
        Controller.load_state_dict(torch.load(Controller_loc))
        Controller.eval() # eval mode
        fig_path_tmp = [ele + '/' for ele in matrix_file.split('/')]
        current_time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        fig_path_visual_demo = ''.join(fig_path_tmp[:-1]) + 'RolloutError_visual(DemoError).png'
        fig_path_simu_demo_handJoint = ''.join(fig_path_tmp[:-1]) + current_time + 'RolloutError_simu(DemoError_on_Handjoints).png'
        fig_path_simu_demo_handRT = ''.join(fig_path_tmp[:-1]) + current_time + 'RolloutError_simu(DemoError_on_Base).png'
        fig_path_simu_goal = ''.join(fig_path_tmp[:-1]) + current_time + 'RolloutError_simu(GoalError).png'
        fig_path_simu_goal_unseen = ''.join(fig_path_tmp[:-1]) + 'RolloutError_simu_unseen(GoalError)_' + current_time + '.png'
        fig_hand_joints_demo = [''.join(fig_path_tmp[:-1]) + 'Joint/joints_' + str(i) + '.png' for i in range(num_hand)]
        fig_hand_torques_demo = [''.join(fig_path_tmp[:-1]) + 'Torque/torques_' + str(i) + '.png' for i in range(num_hand)]
        if koopmanoption == "Drafted":
            print("Trained drafted koopman matrix loaded!")
            cont_koopman_operator = np.load(matrix_file)
            Koopman = DraftedObservable(num_hand, num_obj)
            if not Unseen_test:
                errors = koopman_error_visualization(env_name, Koopman, cont_koopman_operator, Training_data, Velocity, num_hand, num_obj, koopmanoption, error_type, False)
                if error_type == 'demo':
                    hand_pos_error_joints = errors[0]
                    hand_pos_error_base = errors[1]
                    computed_joints = errors[2]
                    demo_joints = errors[3]
                if controller:
                    errors_simu = koopman_policy_control(env_name, Controller, Koopman, cont_koopman_operator, Training_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize)
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
                    obj_ori_error_simu, obj_ori_error_simu_koopman, success_rate = koopman_policy_control_unseenTest(env_name, Controller, Koopman, cont_koopman_operator, Training_data, Velocity, num_hand, num_obj, koopmanoption, Visualize)      
                    demo_ori_error_simu = e.visualize_policy_on_demos(rl_policy, Training_data, Visualize, 70)
            if Separate: # store the results for comparing the different policys
                print("Save the results for post-process!")
                np.save(''.join(fig_path_tmp[:-1]) + 'test_error.npy', obj_ori_error_simu)
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
                else:  # plot mean 
                    plt.plot(x, np.mean(hand_pos_error_joints, axis = 1), linewidth=2, label = 'hand joint error', color='#B22400')
                    plt.plot(x, np.mean(hand_pos_error_base, axis = 1), linewidth=2, label = 'base error', color='#F22BB2')
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
                                plt.plot(x, np.median(computed_torques[i], axis = 1), linewidth=2, label = 'NN', color='#B22400')
                                plt.fill_between(x, np.percentile(computed_torques[i], 25, axis = 1), np.percentile(computed_torques[i], 75, axis = 1), alpha = 0.15, linewidth = 0, color='#B22400')
                                plt.plot(x, np.median(demo_torques[i], axis = 1), linewidth=2, label = 'Demo', color='#F22BB2')
                                plt.fill_between(x, np.percentile(demo_torques[i], 25, axis = 1), np.percentile(demo_torques[i], 75, axis = 1), alpha = 0.15, linewidth = 0, color='#F22BB2')
                            else:  # plot mean 
                                plt.plot(x, np.mean(computed_torques[i], axis = 1), linewidth=2, label = 'NN', color='#B22400')
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
                with open(''.join(fig_path_tmp[:-1]) + 'success.txt', 'w') as f:
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
    # MLP: error over num of training iterations
    # GNN: error over num of training iterations
    # Start the training process for each mode
    if not os.path.exists(os.path.join(folder_name, "koopman")):
        os.mkdir(os.path.join(folder_name, "koopman"))
    else:
        shutil.rmtree(os.path.join(folder_name, "koopman"))   # Removes all the subdirectories!
        os.mkdir(os.path.join(folder_name, "koopman"))
    results_record = open(folder_name + "koopman" + '/results.txt', 'w+')
    start_time = time.time()
    if koopmanoption == "Drafted":
        Koopman = DraftedObservable(num_hand, num_obj)
        num_obs = Koopman.compute_observable(num_hand, num_obj)
        print("number of observables:", num_obs)
        A = np.zeros((num_obs, num_obs))  
        G = np.zeros((num_obs, num_obs))
        ## loop to collect data
        print("Drafted koopman training starts!\n")
        for k in tqdm(range(len(Training_data))):
            if Velocity:
                hand_OriState = np.append(Training_data[k][0]['handpos'], Training_data[k][0]['handvel'])
            else:
                hand_OriState = Training_data[k][0]['handpos']
            obj_OriState = np.append(Training_data[k][0]['objpos'], np.append(Training_data[k][0]['objvel'], Training_data[k][0]['handle_init']))
            z_t = Koopman.z(hand_OriState, obj_OriState)  # initial states in lifted space            
            for t in range(len(Training_data[k]) - 1):
                if Velocity:
                    hand_OriState = np.append(Training_data[k][t + 1]['handpos'], Training_data[k][t + 1]['handvel'])
                else:
                    hand_OriState = Training_data[k][t + 1]['handpos']
                obj_OriState = np.append(Training_data[k][t + 1]['objpos'], np.append(Training_data[k][t + 1]['objvel'], Training_data[k][t + 1]['handle_init']))
                z_t_1 = Koopman.z(hand_OriState, obj_OriState) # states in lifted space at next time step
                # A and G are cumulated through all the demo data
                A += np.outer(z_t_1, z_t)
                G += np.outer(z_t, z_t)
                z_t = z_t_1
        M = len(Training_data) * (len(Training_data[0]) - 1)
        A /= M
        G /= M
        koopman_operator = np.dot(A, scipy.linalg.pinv(G)) # do not use np.linalg.pinv, it may include all large singular values
        # cont_koopman_operator = logm(koopman_operator) / dt
        cont_koopman_operator = koopman_operator
        results_record.write("Training time is: %f seconds.\n"%(time.time() - start_time))
        # generate another matrix with similar matrix magnitude to verify the correctness of the learnt koopman matrix
        # we want to see that each element in the matrix does matter
        koopman_mean = np.mean(cont_koopman_operator)
        print("Koopman mean:%f"%(koopman_mean))
        koopman_std = np.std(cont_koopman_operator)
        print("Koopman std:%f"%(koopman_std))
        Test_matrix = np.random.normal(loc = koopman_mean, scale = koopman_std, size = cont_koopman_operator.shape)
        print("Fake matrix mean:%f"%(np.mean(Test_matrix)))
        print("Fake matrix std:%f"%(np.std(Test_matrix)))
        print("Drafted koopman training ends!\n")
        # print("The drafted koopman matrix is: ", cont_koopman_operator)
        # print("The drafted koopman matrix shape is: ", koopman_operator.shape)
                # save the trained koopman matrix
        if save_matrix:
            np.save(folder_name + "koopman" + '/koopmanMatrix.npy', cont_koopman_operator)
            print("Koopman matrix is saved!\n")
        print("Koopman final testing starts!\n")
        ErrorInLifted, ErrorInOriginal = koopman_evaluation(Koopman, cont_koopman_operator, Training_data, Velocity, num_hand, num_obj)
        Fake_ErrorInLifted, Fake_ErrorInOriginal = koopman_evaluation(Koopman, Test_matrix, Training_data, Velocity, num_hand, num_obj)
        print("Koopman final testing ends!\n")
        if error_calc == 'median': # compute the median 
            print("The final test accuracy in lifted space is: %f, and the accuracy in original space is: %f."%(np.median(ErrorInLifted), np.median(ErrorInOriginal)))
            print("The fake test accuracy in lifted space is: %f, and the fake accuracy in original space is: %f."%(np.median(Fake_ErrorInLifted), np.median(Fake_ErrorInOriginal)))
            results_record.write("The final test accuracy in lifted space is: %f, and the accuracy in original space is: %f.\n"%(np.median(ErrorInLifted), np.median(ErrorInOriginal)))
            results_record.write("The fake test accuracy in lifted space is: %f, and the fake accuracy in original space is: %f.\n"%(np.median(Fake_ErrorInLifted), np.median(Fake_ErrorInOriginal)))
        else:
            print("The final test accuracy in lifted space is: %f, and the accuracy in original space is: %f."%(np.mean(ErrorInLifted), np.mean(ErrorInOriginal)))
            print("The fake test accuracy in lifted space is: %f, and the fake accuracy in original space is: %f."%(np.mean(Fake_ErrorInLifted), np.mean(Fake_ErrorInOriginal)))
            results_record.write("The final test accuracy in lifted space is: %f, and the accuracy in original space is: %f.\n"%(np.mean(ErrorInLifted), np.mean(ErrorInOriginal)))
            results_record.write("The fake test accuracy in lifted space is: %f, and the fake accuracy in original space is: %f.\n"%(np.mean(Fake_ErrorInLifted), np.mean(Fake_ErrorInOriginal)))
        results_record.write("The number of demo used for experiment is: %d.\n"%(num_demo))
        results_record.flush()
    results_record.close()

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def demo_playback(env_name, demo_paths, num_demo):
    e = GymEnv(env_name)
    e.reset()
    Training_data = []
    print("Begin loading demo data!")
    print(num_demo)
    sample_index = np.arange(num_demo)  
    for i in tqdm(sample_index):
        path = demo_paths[i]
        path_data = []
        e.set_env_state(path['init_state_dict'])
        actions = path['actions']
        observations = path['observations']  
        observations_visualize = path['observations_visualization']
        for tt in range(len(actions)):  
            tmp = dict()
            if tt == 0:
                tmp['init'] = path['init_state_dict']
            obs = observations[tt] 
            obs_visual = observations_visualize[tt]
            handpos = obs_visual[:28] 
            tmp['handpos'] = handpos
            tmp['handvel'] = obs_visual[30:58]
            tmp['objpos'] = obs[32:35] 
            tmp['objvel'] = obs_visual[58:59]
            tmp['handle_init'] = path['init_state_dict']['door_body_pos'] 
            tmp['observation'] = obs[35:38]
            tmp['action'] = actions[tt]
            if False:
                e.env.mj_render()  # render the visualizer
            path_data.append(tmp)
        Training_data.append(path_data)
    return Training_data

if __name__ == '__main__':
    main()

