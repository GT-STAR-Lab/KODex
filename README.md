# On the Utility of Koopman Operator Theory in Learning Dexterous Manipulation Skills
This is a repository containing the code for the paper "On the Utility of Koopman Operator Theory in Learning Dexterous Manipulation Skills", which has been accepted as an <strong>oral presentation</strong> at CoRL 2023.

Project webpage: [KODex](https://sites.google.com/view/kodex-corl)

Paper link: [On the Utility of Koopman Operator Theory in Learning Dexterous Manipulation Skills](https://arxiv.org/abs/2303.13446)
## Environment Setup

Please refer to DAPG project to setup the Mujoco environment: [Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations](https://github.com/aravindr93/hand_dapg). Note that we are not aware if our codes support the latest mj_envs envinronment, so we recomment install v1.0.0 of mj_envs, which is the one used before.


## KODex 

The training and testing codes for each task are generated seperately, and differences mainly lie on the state design and task success criteia. The first step is to install the NGF dependency:
```
$ conda activate mjrl-env
$ pip install geometric-fabrics-torch/dist/geometric_fabrics_torch-0.0.0-py2.py3-none-any.whl --force
```
Please make sure that you switch to the conda environment where you installed DAPG dependencies. If you followed the instructions from DAPG project, it should be *mjrl-env*. In addtion, for Object Relocation and Tool Use tasks, the return of *get_obs()* functions under each env-py files need to be replaced with following:

1. In relocate_v0.py: `return np.concatenate([qp[:-6], palm_pos-obj_pos, palm_pos-target_pos, obj_pos-target_pos])` is replaced with `return np.concatenate([qp[:-6], palm_pos-obj_pos, palm_pos-target_pos, obj_pos-target_pos, obj_pos, palm_pos, target_pos])`  
2. In hammer_v0.py: `return np.concatenate([qp[:-6], qv[-6:], palm_pos, obj_pos, obj_rot, target_pos, np.array([nail_impact])])` is replaced with `return np.concatenate([qp[:-6], qv[-6:], palm_pos, obj_pos, obj_rot, target_pos, np.array([nail_impact]), goal_pos, tool_pos - target_pos])`

Additional, In KODex, we assume the initial object position is fixed. So, in relocate_v0.py, change the initial xy pos of the ball to be 0: 

1. line85: `self.model.body_pos[self.obj_bid,0] = 0`

2. line86: `self.model.body_pos[self.obj_bid,1] = 0`

Lastly, you have to add the following functions to each  env-py file:

1. In hammer_v0.py:
```python 
    def KoopmanVisualize(self, state_dict):
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        board_pos = state_dict['board_pos']
        self.set_state(qp, qv)
        self.model.body_pos[self.model.body_name2id('nail_board')] = board_pos
        self.sim.forward()

    def get_full_obs_visualization(self):  
        qp = self.data.qpos.ravel()
        qv = self.data.qvel.ravel()
        return np.concatenate([qp, qv])
```

2. In door_v0.py:
```python 
    def KoopmanVisualize(self, state_dict):
        # visualize the koopman trajectory
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        self.set_state(qp, qv)
        self.model.body_pos[self.door_bid] = state_dict['door_body_pos']
        self.sim.forward()

    def get_full_obs_visualization(self):  
        qp = self.data.qpos.ravel()
        qv = self.data.qvel.ravel()
        return np.concatenate([qp, qv])
```

3. In relocate_v0.py:
```python 
    def KoopmanVisualize(self, state_dict):
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        self.set_state(qp, qv)
        obj_pos = state_dict['obj_pos']
        target_pos = state_dict['target_pos']
        self.model.body_pos[self.obj_bid] = obj_pos
        self.model.site_pos[self.target_obj_sid] = target_pos
        self.sim.forward()

    def get_full_obs_visualization(self):  
        qp = self.data.qpos.ravel()
        qv = self.data.qvel.ravel()
        return np.concatenate([qp, qv])
```

4. In pen_v0.py:
```python 
    def KoopmanVisualize(self, state_dict):
        qp = self.init_qpos.copy()
        qp = state_dict['qpos']
        qv = self.init_qvel.copy()
        qv = state_dict['qvel']
        self.set_state(qp, qv)
        desired_orien = state_dict['desired_orien']
        self.model.body_quat[self.target_obj_bid] = euler2quat(desired_orien)
        self.sim.forward()

    def get_full_obs_visualization(self): 
        qp = self.data.qpos.ravel()
        qv = self.data.qvel.ravel()
        return np.concatenate([qp, qv])
```
As far as we know, after you make these changes, you should be able to run the following experiments. If you still get some unexpected errors, feel free to leave an issue or contact us via email!

### Door
To visulize each trained policy on the test set

**KODex**:
```
$ conda activate mjrl-env
$ cd Door/
$ MJPL python3 Koopman_training.py --env_name door-v0 --demo_file ./Data/Testing.pickle --num_demo 0 --koopmanoption Drafted --velocity False --save_matrix True --matrix_file ./Results/KODex/koopmanMatrix.npy --control True --error_type demo --visualize True --unseen_test False --rl_policy ./Results/Expert_policy/best_policy.pickle
```
**NN**:
```
$ conda activate mjrl-env
$ cd Door/
$ MJPL python3 BC_training.py --env_name door-v0 --demo_file ./Data/Testing.pickle --num_demo 0 --koopmanoption Drafted --velocity False --save_matrix True --matrix_file ./Results/NN/BC_agent.pt --control True --error_type demo --visualize True --unseen_test False --rl_policy ./Results/Expert_policy/best_policy.pickle
```
**LSTM**:
```
$ conda activate mjrl-env
$ cd Door/
$ MJPL python3 LSTM_training.py --env_name door-v0 --demo_file ./Data/Testing.pickle --num_demo 0 --koopmanoption Drafted --velocity False --save_matrix True --matrix_file ./Results/LSTM/LSTM_agent.pt --control True --error_type demo --visualize True --unseen_test False --rl_policy ./Results/Expert_policy/best_policy.pickle
```
**NDP**:
```
$ conda activate mjrl-env
$ cd Door/
$ MJPL python3 NDP_training.py --env_name door-v0 --demo_file ./Data/Testing.pickle --num_demo 0 --koopmanoption Drafted --velocity False --save_matrix True --matrix_file ./Results/NDP/NDP_agent.pt --control True --error_type demo --visualize True --unseen_test False --rl_policy ./Results/Expert_policy/best_policy.pickle
```
**NGF**:
```
$ conda activate mjrl-env
$ cd Door/
$ MJPL python3 NGF_training.py --env_name door-v0 --demo_file Data/Testing.pickle --num_demo 0 --koopmanoption Drafted --velocity False --save_matrix True --matrix_file Results/NGF/ --control True --error_type demo --visualize True --unseen_test False --rl_policy Results/Expert_policy/best_policy.pickle --seed 1 --first_demo 200
```

To train a new policy using 200 demonstrations

**KODex**:
```
$ conda activate mjrl-env
$ cd Door/
$ MJPL python3 Koopman_training.py --env_name door-v0 --demo_file ./Data/Demonstration.pickle --num_demo 200 --koopmanoption Drafted --velocity False --save_matrix True --matrix_file None --control True --error_type demo --visualize True --unseen_test False --rl_policy ./Results/Expert_policy/best_policy.pickle --folder_name Results/New_policy/
```

For other baselines, the ways to train new policies are simiar as KODex. Note that NGF is currently not available for training.

### Tool Use, Object Relocation, In-hand Reorientation
For these tasks, we also provide with the demonstration data under `/Data` folder of each task module. Therefore, you can use the similar commands as above to visualze the trained KODex policy or train a new KODex policy. For example, to visualize the KODex policy for Relocation task:

**Testing**:
```
$ conda activate mjrl-env
$ cd Relocation/
$ MJPL python3 Koopman_training.py --env_name relocate-v0 --demo_file ./Data/Relocate_task.pickle --num_demo 0 --koopmanoption Drafted --velocity False --save_matrix True --matrix_file ./Results/KODex/koopmanMatrix.npy --control True --error_type demo --visualize True --unseen_test False --rl_policy ./Results/Expert_policy/best_policy.pickle
```

And to train a new KODex policy for Relocation task:

**Training**:
```
$ conda activate mjrl-env
$ cd Relocation/
$ MJPL python3 Koopman_training.py --env_name relocate-v0 --demo_file ./Data/Relocate_task.pickle --num_demo 200 --koopmanoption Drafted --velocity False --save_matrix True --matrix_file None --control True --error_type demo --visualize True --unseen_test False --rl_policy ./Results/Expert_policy/best_policy.pickle --folder_name Results/New_policy/
```
**More Test instances**:

For each task, we also provide with extra 20,000 test instances. You could download them via [this Link](https://drive.google.com/file/d/12heE7bgf0NvU0TAmhgtRCNCNrpzjLEp6/view?usp=sharing), and then specify the demo_file location when running the commands.
## Bibtex
```
@inproceedings{han2023KODex,
  title={On the Utility of Koopman Operator Theory in Learning Dexterous Manipulation Skills},
  author={Han, Yunhai and Xie, Mandy and Zhao, Ye and Ravichandar, Harish},
  booktitle={Conference on Robot Learning},
  year={2023},
  organization={PMLR}
}
```