import sys

sys.path.append(".")

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
import lasagne.nonlinearities as NL

# Policy
# from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy

# Environment
from envs.distflow.voltvar_env34 import VoltVarEnv

# Policy optimization
from sac.algos.sacd import SACD
from sac.replay_buffers import SimpleReplayBuffer
from sac.misc.sampler import SimpleSampler
from sac.value_functions import NNQFunction, NNVFunction
from sac.misc.utils import timestamp
from sac.policies import UniformPolicy2, CategoricalPolicy
#from sac.misc.instrument import run_sac_experiment
from rllab.misc import logger
import os

from sac.misc.utils import PROJECT_PATH

DEFAULT_LOG_DIR = PROJECT_PATH + "/data"


def run_experiment(*_):
    env = normalize(VoltVarEnv())

    pool = SimpleReplayBuffer(max_replay_buffer_size=1e6, env_spec=env.spec)

    sampler = SimpleSampler(
        max_path_length=168, min_pool_size=100, batch_size=512)

    base_kwargs = dict(
        sampler=sampler,
        epoch_length=1000,
        n_epochs=50,
        n_initial_exploration_steps=10000,
        n_train_repeat=1,
        # eval_render=False,
        eval_n_episodes=10,#50,
        eval_deterministic=True
    )

    qf1 = NNQFunction(
        env_spec=env.spec,
        hidden_layer_sizes=[64, 32],
        name='qf1'
    )

    qf2 = NNQFunction(
        env_spec=env.spec,
        hidden_layer_sizes=[64, 32],
        name='qf2'
    )

    vf = NNVFunction(
        env_spec=env.spec,
        hidden_layer_sizes=[64, 32],
        name='vf'
    )

    qfc1 = NNQFunction(
        env_spec=env.spec,
        hidden_layer_sizes=[64, 32],
        name='qfc1'
    )

    qfc2 = NNQFunction(
        env_spec=env.spec,
        hidden_layer_sizes=[64, 32],
        name='qfc2'
    )

    vfc = NNVFunction(
        env_spec=env.spec,
        hidden_layer_sizes=[64, 32],
        name='vfc'
    )

    initial_exploration_policy = UniformPolicy2(env_spec=env.spec)

    # policy = GaussianPolicy(
    #     env_spec=env.spec,
    #     hidden_layer_sizes=[64, 32],
    #     reparameterize=True,
    #     reg=1e-3,
    # )
    policy = CategoricalPolicy(
        env_spec=env.spec,
        hidden_layer_sizes=[64, 32]
    )

    algo = SACD(
        base_kwargs=base_kwargs,
        env=env,
        policy=policy,
        initial_exploration_policy=initial_exploration_policy,
        pool=pool,
        qf1=qf1,
        qf2=qf2,
        qfc1=qfc1,
        qfc2=qfc2,
        vf=vf,
        vfc=vfc,
        # plotter=plotter,
        lr=1e-3,
        scale_reward=50,#2.5,  ## 50 bus 4; 10 bus34
        scale_rewardc=5,  # 2.5,  ## 50 bus 4; 10 bus34
        alpha=1,
        discount=0.99,
        tau=5e-4, #bus34 5e-4;bus123 2.5e-4,;
        target_update_interval=1,
        #reparameterize=True,
        save_full_state=False
    )

    algo.train()


if __name__ == "__main__":

    exp_prefix = 'SACD-VoltVar34-exp7'
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
    #     exp_prefix='SACD-VoltVar4-cont',
    #     n_parallel=1,
    #     #seed=1,
    #     snapshot_mode="last",
    # )



