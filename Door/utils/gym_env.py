"""
Wrapper around a gym env that provides convenience functions
"""

import gym
import numpy as np
from utils.rnn_network import RNNNetwork
import pickle
from tqdm import tqdm
import copy
from utils.coord_trans import ori_transform, ori_transform_inverse
from utils.quatmath import euler2quat

class EnvSpec(object):
    def __init__(self, obs_dim, act_dim, horizon):
        self.observation_dim = obs_dim
        self.action_dim = act_dim
        self.horizon = horizon


class GymEnv(object):
    def __init__(self, env, env_kwargs=None,
                 obs_mask=None, act_repeat=1, 
                 *args, **kwargs):
    
        # get the correct env behavior
        if type(env) == str:
            env = gym.make(env)  # generare the mojuco env
        elif isinstance(env, gym.Env):
            env = env
        elif callable(env):
            env = env(**env_kwargs)
        else:
            print("Unsupported environment format")
            raise AttributeError

        self.env = env
        self.env_id = env.spec.id
        self.act_repeat = act_repeat
        try:
            self._horizon = env.spec.max_episode_steps  # max_episode_steps is defnied in the __init__.py file (under )
        except AttributeError:
            self._horizon = env.spec._horizon
        assert self._horizon % act_repeat == 0
        self._horizon = self._horizon // self.act_repeat

        try:
            self._action_dim = self.env.env.action_dim
        except AttributeError:
            self._action_dim = self.env.action_space.shape[0]

        try:
            self._observation_dim = self.env.env.obs_dim
        except AttributeError:
            self._observation_dim = self.env.observation_space.shape[0]

        # Specs
        self.spec = EnvSpec(self._observation_dim, self._action_dim, self._horizon)

        # obs mask
        self.obs_mask = np.ones(self._observation_dim) if obs_mask is None else obs_mask

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def observation_dim(self):
        return self._observation_dim

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):  # each env has defined a action space
        return self.env.action_space

    @property
    def horizon(self):
        return self._horizon

    def reset(self, seed=None):
        try:
            self.env._elapsed_steps = 0
            return self.env.env.reset_model(seed=seed)
        except:
            if seed is not None:
                self.set_seed(seed)
            return self.env.reset()
    
    def reset4Koopman(self, seed=None, ori=None, init_pos=None, init_vel=None):
        try:
            self.env._elapsed_steps = 0
            return self.env.env.reset_model4Koopman(seed=seed, ori = ori, init_pos = init_pos, init_vel = init_vel)
        except:
            if seed is not None:
                self.set_seed(seed)
            return self.env.reset_model4Koopman(ori = ori, init_pos = init_pos, init_vel = init_vel)

    def KoopmanVisualize(self, seed=None, state_dict=None):
        try:
            self.env._elapsed_steps = 0
            return self.env.env.KoopmanVisualize(seed=seed, state_dict=state_dict)
        except:
            if seed is not None:
                self.set_seed(seed)
            return self.env.KoopmanVisualize(state_dict=state_dict)

    def reset_model(self, seed=None):
        # overloading for legacy code
        return self.reset(seed)

    def step(self, action):
        action = action.clip(self.action_space.low, self.action_space.high)
        # type(action_space) -> <class 'gym.spaces.box.Box'>
        # self.action_space.low -> numpy.ndarray(lowest boundary)
        # self.action_space.high -> numpy.ndarray(highest boundary)
        if self.act_repeat == 1: 
            obs, cum_reward, done, ifo = self.env.step(action)  # the system dynamics is defined in each separate env python file
            # if(ifo['goal_achieved']):
            #     print("done: ", ifo)    
            # Run one timestep of the environment’s dynamics.
        else:
            cum_reward = 0.0
            for i in range(self.act_repeat):
                obs, reward, done, ifo = self.env.step(action) # the actual operations can be found in the env files
                # seems done is always set to be False
                cum_reward += reward
                if done: break
        return self.obs_mask * obs, cum_reward, done, ifo

    def render(self):
        try:
            self.env.env.mujoco_render_frames = True
            self.env.env.mj_render()
        except:
            self.env.render()

    def set_seed(self, seed=123):
        try:
            self.env.seed(seed)
        except AttributeError:
            self.env._seed(seed)

    def get_obs(self):
        try:
            return self.obs_mask * self.env.env.get_obs()
        except:
            return self.obs_mask * self.env.env._get_obs()

    def get_env_infos(self):
        try:
            return self.env.env.get_env_infos()
        except:
            return {}

    # ===========================================
    # Trajectory optimization related
    # Envs should support these functions in case of trajopt

    def get_env_state(self):
        try:
            return self.env.env.get_env_state()
        except:
            raise NotImplementedError

    def set_env_state(self, state_dict):
        try:
            self.env.env.set_env_state(state_dict)
        except:
            raise NotImplementedError

    def real_env_step(self, bool_val):
        try:
            self.env.env.real_step = bool_val
        except:
            raise NotImplementedError

    def visualize_policy_on_demos(self, policy, demos, Visualize, horizon=1000):
        print("Testing the RL agent!")
        self.reset()
        init_state_dict = dict()
        demo_ori_error = np.zeros([horizon - 1, len(demos)])
        success_list_RL = []
        for k in tqdm(range(len(demos))):
            init_state_dict['qpos'] = demos[k]['init']['qpos']
            init_state_dict['qvel'] = demos[k]['init']['qvel']
            init_state_dict['door_body_pos'] = demos[k]['init']['door_body_pos']  # fixed for each piece of demo data
            self.set_env_state(init_state_dict)
            o = demos[k]['o']
            if True:  # RL agent is trained using RNN
                # generate the hidden states at time 0
                hidden_state = (np.zeros((1, 1, policy.model.rnn_cell.hidden_size)), np.zeros((1, 1, policy.model.rnn_cell.hidden_size)))
            for t in range(horizon - 1):  # this loop is for system evolution, open loop control, no feedback
                if True:
                    a = policy.get_action(o, hidden_state)
                    hidden_state = a[1]['hidden_state']
                else:
                    a = policy.get_action(o)
                a =a[1]['evaluation']
                o, *_ = self.step(a)
                if Visualize:
                    self.render()
                # calculated state values using Koopman rollouts (in simulator)      
                obj_obs = self.get_obs()
                # compute the errors
                demo_ori_error[t, k] = np.mean(np.abs(obj_obs[35:38]))  
                current_hinge_pos = obj_obs[28:29] # door opening angle
            if current_hinge_pos > 1.35:
                success_list_RL.append(1)
        print("Success rate (RL) = %f" % (len(success_list_RL) / len(demos)))
        return demo_ori_error

    def generate_unseen_data(self, number_sample):
        samples = []
        for ep in range(number_sample):
            o, desired_orien = self.reset(seed = None)
            episode_data = {}
            episode_data['init_state_dict'] = copy.deepcopy(self.get_env_state())
            episode_data['pen_desired_orien'] = desired_orien  # record the goal orientation angle
            episode_data['o'] = o
            handpos = o[:24]
            episode_data['handpos'] = handpos
            hand_vel = self.env.get_hand_vel()
            episode_data['handvel'] = hand_vel
            objpos = o[24:27]
            episode_data['objpos'] = objpos
            objvel = o[27:33] 
            episode_data['objvel'] = objvel
            episode_data['desired_ori'] = o[36:39]
            objorient = o[33:36]
            episode_data['objorient'] = ori_transform(objorient, episode_data['desired_ori']) 
            samples.append(episode_data)
        return samples

    def generate_unseen_data_relocate(self, number_sample):
        samples = []
        for ep in range(number_sample):
            o, desired_pos = self.reset(seed = None)
            o_visual = self.env.get_full_obs_visualization()
            episode_data = {}
            episode_data['init'] = copy.deepcopy(self.get_env_state())
            episode_data['desired_pos'] = desired_pos  
            episode_data['o'] = o
            handpos = o[:30]
            episode_data['handpos'] = handpos
            hand_vel = self.env.get_hand_vel()[:30]
            episode_data['handvel'] = hand_vel
            objpos = o[39:42]
            episode_data['objpos'] = objpos - episode_data['desired_pos']
            episode_data['objorient'] = o_visual[33:36]
            episode_data['objvel'] = self.env.get_hand_vel()[30:]
            samples.append(episode_data)
        return samples

    def generate_unseen_data_hammer(self, number_sample):
        samples = []
        for ep in range(number_sample):
            o, _ = self.reset(seed = None)
            hand_vel = self.env.get_hand_vel()
            episode_data = {}
            episode_data['init'] = copy.deepcopy(self.get_env_state())
            episode_data['o'] = o
            handpos = o[:26]
            episode_data['handpos'] = handpos
            episode_data['handvel'] = hand_vel[:26]
            objpos = o[49:52] + o[42:45] 
            episode_data['objpos'] = objpos # current tool position
            episode_data['objorient'] = o[39:42] 
            episode_data['objvel'] = o[27:33]
            episode_data['nail_goal'] = o[46:49]
            samples.append(episode_data)
        return samples

    def generate_unseen_data_door(self, number_sample):
        samples = []
        for ep in range(number_sample):
            o, desired_pos = self.reset(seed = ep)
            obs_visual = self.env.get_full_obs_visualization()
            hand_vel = self.env.get_hand_vel()
            episode_data = {}
            episode_data['init'] = copy.deepcopy(self.get_env_state())
            episode_data['o'] = o
            handpos = obs_visual[:28]
            episode_data['handpos'] = handpos
            episode_data['handvel'] = hand_vel[:28]
            objpos = o[32:35]
            episode_data['objpos'] = objpos # current tool position
            episode_data['objvel'] = obs_visual[58:59]
            episode_data['handle_init'] = episode_data['init']['door_body_pos']
            samples.append(episode_data)
        return samples