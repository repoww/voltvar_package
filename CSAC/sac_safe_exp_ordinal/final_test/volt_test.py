import joblib
import numpy as np
import tensorflow as tf
from envs.distflow.voltvar_env import VoltVarEnv

import time



# lambda =1
#data_path = '../data/local/SACD-VoltVar4/2019-02-14-21-14-40-964588-PST/params.pkl'
data_path = '../data/local/SACD-VoltVar13-exp/2019-04-10-23-54-06-097717-PDT/params.pkl'

env = VoltVarEnv(mode='Test')

obs = env.reset()
rec_act = []
rec_reward = []
rec_cost = []
last_act = np.array([5,5,0])
swtich_num = 0

base = [5,5,0.5]

with tf.Session() as sess:
    data = joblib.load(data_path)


    policy = data['policy']

    policy._is_deterministic = True

    start = time.time()
    for i in range(168):
        action, _ = policy.get_action(obs)
        # if i <= 5:
        #     action = 120
        # else:
        #     action = 121
        #act = env.actions[action]
        act = [np.round(a*b+b) for a, b in zip(action,base)]
        swtich_num+=np.sum(np.abs(np.array(act)-np.array(last_act)))

        rec_act.append(act)
        rt_step= env.step(action)
        obs = rt_step[0]
        reward = rt_step[1]
        cost = rt_step[3]['cost']
        rec_reward.append(reward)
        rec_cost.append(cost)

        last_act = act

print(rec_act)
print(sum(rec_reward))
print(swtich_num)
print(sum(rec_cost))
