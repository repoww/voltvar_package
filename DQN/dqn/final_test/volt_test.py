import joblib
import numpy as np
import tensorflow as tf
from envs.distflow.voltvar_env import VoltVarEnv

import time



# lambda =1
#data_path = '../data/local/SACD-VoltVar4/2019-02-14-21-14-40-964588-PST/params.pkl'
# data_path = '../data/local/DQN-VoltVar13-exp/2019-04-09-15-33-08-411961-PDT/params.pkl'
data_path = '../data/local/DQN-VoltVar13-exp/2019-04-11-16-15-22-980700-PDT/params.pkl'
env = VoltVarEnv(mode='Test')

obs = env.reset()
rec_act = []
rec_reward = []
rec_cost = []
last_act = np.array([5,5,0])
swtich_num = 0


with tf.Session() as sess:
    data = joblib.load(data_path)


    qf = data['qf']

    #policy._is_deterministic = True

    start = time.time()
    for i in range(168):
        action = np.argmax(qf.eval(np.expand_dims(obs,axis=0)))
        # if i <= 5:
        #     action = 120
        # else:
        #     action = 121
        act = env.actions[action]
        swtich_num+=np.sum(np.abs(act-last_act))

        rec_act.append(act.tolist())
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
