from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.core.serializable import Serializable
from rllab.spaces import Box,Discrete
import numpy as np
import itertools
import envs.distflow.voltvar_env_utils123 as utils

class VoltVarEnv(Env,Serializable):
        def __init__(self, mode='Train'):
                self.load_train =np.loadtxt('dqn/envs/distflow/load_train.csv')
                self.load_test = np.loadtxt('dqn/envs/distflow/load_test.csv')
                self.load_len_train = self.load_train.shape[0]
                self.load_len_test = self.load_test.shape[0]
                self.state = []
                self.actions = np.array(list(itertools.product(range(11), range(11), range(2))))
                self.global_time = 0
                self.local_time = 0
                self.mode = mode
                self.week = 0
                #self.train_weeks = [i for i in range(26) if i != 5]
                super(Env, self).__init__()
                Serializable.quick_init(self, locals())

        @property
        def observation_space(self):
                return Box(low=-np.inf, high=np.inf, shape=(5,))

        @property
        def action_space(self):
                return Discrete(11*11*2)

        def reset(self):
                self.state = np.zeros(1 + 3 + 1)
                if self.mode == 'Train':
                        self.global_time = np.random.randint(42)*168
                        self.state[:1] = self.load_train[self.global_time]
                else:
                        self.global_time = self.week*168
                        self.state[:1] = self.load_test[self.global_time]
                self.local_time = self.global_time % 168
                # Our state:
                # current load
                # current tap positions
                # one-hot hour
                # one-hot day
                # current t
                self.state[1:4] = [5,5,0]
                self.state[4] = self.local_time
                return self.state

        def step(self, actid):
                action = self.actions[actid]
                load = self.state[0]
                Pij, Qij, Vi2, Lij, total_loss, iter_total, convergence_flag = utils.load_flow(action,load)
                if convergence_flag !=1:
                        print ('power flow not converge')
                Vi = np.sqrt(Vi2)
                # penalt for voltage constraint violation 0.01; total loss; tap change
                #reward = -0.01*np.sum(Vi<0.95)-0.01*np.sum(Vi>1.05)-total_loss-0.001*np.sum(np.abs(action-self.state[24:27]))
                #reward = - total_loss*10*100
                info = {}
                info['gain'] = - (total_loss*10*40+0.1*np.sum(np.abs(action-self.state[1:4])))  ## bus4: 10; bus13: 5
                info['cost'] = np.sum(Vi<0.95)+np.sum(Vi>1.05)
                reward = info['gain'] - info['cost']

                if (self.local_time+1) % 168 == 0:
                        done = True
                else:
                        done = False
                # update to next state
                self.global_time +=1

                if self.mode=='Train' and self.global_time >= self.load_len_train:
                        self.global_time = 0

                if self.mode=='Test' and self.global_time >= self.load_len_test:
                        self.global_time = 0

                self.local_time +=1
                self.state = np.zeros(1 + 3 + 1)
                if self.mode == 'Train':
                        self.state[:1] = self.load_train[self.global_time]
                else:
                        self.state[:1] = self.load_test[self.global_time]
                self.state[1:4] = action # current tap
                self.state[4] = self.local_time

                return Step(self.state, reward, done,**info)


if __name__ == "__main__":
    test = VoltVarEnv()
