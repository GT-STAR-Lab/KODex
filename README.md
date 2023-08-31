# On the Utility of Koopman Operator Theory in Learning Dexterous Manipulation Skills

For the anynomous review purpose, we do not explicitly put a github link in the paper. Instead, we include the codes as part of supplementary materials. If the paper is finally accepted, we will open-source the codes and data. Since we are only allowed 100MB for the supplementary materials, we only add the demonstration and testing data for the Door task.

## Environment Setup

Please refer to DAPG project to setup the Mujoco environment: [Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations](https://github.com/aravindr93/hand_dapg).


## KODex 

The training and testing codes for each task are generated seperately, and differences mainly lie on the state design and task success criteia. The first step is to install the NGF dependency:
```
$ conda activate mjrl-env
$ pip install geometric-fabrics-torch/dist/geometric_fabrics_torch-0.0.0-py2.py3-none-any.whl --force
```
Please make sure that you switch to the conda environment where you installed DAPG dependencies. If you followed the instructions from DAPG project, it should be *mjrl-env*. In addtion, for Object Relocation and Tool Use tasks, the return of *get_obs()* functions under each env-py files need to be replaced with following:

1. In relocate_v0.py: change to `return np.concatenate([qp[:-6], palm_pos-obj_pos, palm_pos-target_pos, obj_pos-target_pos, obj_pos, palm_pos, target_pos])`  
2. In hammer_v0.py: change to `return np.concatenate([qp[:-6], qv[-6:], palm_pos, obj_pos, obj_rot, target_pos, np.array([nail_impact]), goal_pos, tool_pos - target_pos])`
```
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
$ MJPL python3 Koopman_training.py --env_name door-v0 --demo_file ./Data/Demonstratino.pickle --num_demo 200 --koopmanoption Drafted --velocity False --save_matrix True --matrix_file None --control True --error_type demo --visualize True --unseen_test False --rl_policy ./Results/Expert_policy/best_policy.pickle --folder_name Results/New_policy/
```

For other baselines, the ways to train new policies are simiar as KODex. Note that NGF is currently not available for training.

### Tool Use, Object Relocation, In-hand Reorientation
For these tasks, we only provide with the codes due to the size limitation.