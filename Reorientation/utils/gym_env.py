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
        success_threshold = 10
        success_list_RL = []
        fall_list_RL = []
        success_rate = str()
        for k in tqdm(range(len(demos))):
            success_count_RL = np.zeros(horizon - 1)
            init_state_dict['qpos'] = np.append(demos[k]['handpos'], np.zeros(6))
            # For hand control test, we can set the initial pos of pen to be (0.15, 0, 0, 0, 0, 0), so that there is no contact.
            # init_state_dict['qpos'][num_handpos] = 0.15  
            init_state_dict['qvel'] = np.append(demos[k]['handvel'], np.zeros(6))
            init_state_dict['desired_orien'] = euler2quat(demos[k]['pen_desired_orien'])
            self.set_env_state(init_state_dict)
            o = demos[k]['o']
            if True:
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
                obj_vel = self.get_obs()[27:33] 
                orien_similarity_RL = np.dot(obj_obs[33:36], obj_obs[36:39])
                dist = np.linalg.norm(obj_obs[39:42])
                success_count_RL[t] = 1 if (orien_similarity_RL > 0.90) else 0
                # compute the errors
                demo_ori_error[t, k] = np.mean(np.abs(obj_obs[42:45]))  # obj_orien-desired_orien (goal error)
            if np.abs(obj_obs[39:42])[2] > 0.15:
                fall_list_RL.append(1)
            else:
                if sum(success_count_RL) > success_threshold and np.mean(np.abs(obj_vel)) < 1.:
                    success_list_RL.append(1)
        print("Success rate (RL) = %f" % (len(success_list_RL) / len(demos)))
        print("Throw out rate (RL) = %f" % (len(fall_list_RL) / len(demos)))
        success_rate += "Success rate (RL) = %f" % (len(success_list_RL) / len(demos))
        success_rate += "Throw out rate (RL) = %f" % (len(fall_list_RL) / len(demos))
        return demo_ori_error, success_rate

    def generate_unseen_data(self, number_sample):
        samples = []
        for ep in range(number_sample):
            o, desired_orien = self.reset(seed = ep)
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
