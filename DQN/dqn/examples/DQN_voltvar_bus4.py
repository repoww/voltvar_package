import sys

sys.path.append(".")

from rllab.envs.normalized_env import normalize
#from rllab.misc.instrument import run_experiment_lite
#import lasagne.nonlinearities as NL

# Policy
# from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
# from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy

# Environment
from envs.distflow.voltvar_env import VoltVarEnv

# Policy optimization
from sac.algos.dqn import DQN
from sac.replay_buffers import SimpleReplayBuffer
from sac.misc.sampler import SimpleSampler
from sac.value_functions import NNQFunction2
from sac.misc.utils import timestamp
# from sac.policies.categorical_policy import CategoricalPolicy
from sac.policies import GreedyPolicy
# from sac.misc.instrument import run_sac_experiment
from rllab.misc import logger
import os
#ec2_mode = False
from sac.misc.utils import PROJECT_PATH
DEFAULT_LOG_DIR = PROJECT_PATH + "/data"


def run_experiment(*_):
    env = normalize(VoltVarEnv())

    pool = SimpleReplayBuffer(max_replay_buffer_size=1e6, env_spec=env.spec)

    sampler = SimpleSampler(
        max_path_length=168, min_pool_size=100, batch_size=256)

    base_kwargs = dict(
        sampler=sampler,
        epoch_length=1000,
        n_epochs=50,
        n_initial_exploration_steps=10000,
        n_train_repeat=1,
        # eval_render=False,
        eval_n_episodes=10,
        eval_deterministic=True
    )

    qf = NNQFunction2(
        env_spec=env.spec,
        hidden_layer_sizes=[64, 32],
        name='qf'
    )

    qftarg = NNQFunction2(
        env_spec=env.spec,
        hidden_layer_sizes=[64, 32],
        name='qf2'
    )

    policy = GreedyPolicy(
        env_spec=env.spec,
        qf=qf,
        epsilon=1.0
    )

    #initial_exploration_policy = UniformPolicy2(env_spec=env.spec)


    algo = DQN(
        base_kwargs=base_kwargs,
        env=env,
        policy = policy,
        #initial_exploration_policy=initial_exploration_policy,
        pool=pool,
        qf=qf,
        qftarg=qftarg,
        # plotter=plotter,
        lr=1e-3,
        discount=0.99,
        #tau=1e-4,
        target_update_interval=20,
        # reparameterize=False,
        save_full_state=False
    )

    algo.train()


if __name__ == "__main__":

    exp_prefix = 'DQN-VoltVar34-exp2'
    exp_name = timestamp()
    log_dir = os.path.join(
        DEFAULT_LOG_DIR,
        "local",
        exp_prefix.replace("_", "-"),
        exp_name)
    os.makedirs(log_dir,exist_ok=True)
    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode('last')
    tabular_log_file = os.path.join(log_dir,'progress.csv')
    text_log_file = os.path.join(log_dir,'debug.log')
    logger.add_text_output(text_log_file)
    logger.add_tabular_output(tabular_log_file)

    run_experiment()
    # run_sac_experiment(
    #     run_experiment,
    #     mode="local",
    #     exp_prefix='SACD-VoltVar4',
    #     n_parallel=4,
    #     seed=1,
    #     snapshot_mode="last",
    # )



