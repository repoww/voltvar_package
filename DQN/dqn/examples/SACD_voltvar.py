import sys

sys.path.append(".")

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
import lasagne.nonlinearities as NL

# Policy
#from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.categorical_mlp_policy import  CategoricalMLPPolicy

# Environment
from envs.distflow.voltvar_env4 import VoltVarEnv

# Policy optimization
from sac.algos.sacd import SACD
from sac.replay_buffers import SimpleReplayBuffer
from sac.misc.sampler import SimpleSampler
from sac.value_functions import NNQFunction2, NNVFunction
from sac.misc.utils import timestamp
from sac.policies.categorical_policy import CategoricalPolicy
from sac.policies import UniformPolicy2
from sac.misc.instrument import run_sac_experiment

ec2_mode = False


def run_experiment(*_):
        
        env = normalize(VoltVarEnv())

        pool = SimpleReplayBuffer(max_replay_buffer_size=1e6, env_spec=env.spec)

        sampler = SimpleSampler(
            max_path_length=168, min_pool_size=100, batch_size=256)

        base_kwargs = dict(
            sampler=sampler,
            epoch_length=1000,
            n_epochs=1000,
            n_initial_exploration_steps = 10000,
            n_train_repeat=1,
            #eval_render=False,
            eval_n_episodes=50,
            eval_deterministic=False
        )

        qf1 = NNQFunction2(
            env_spec=env.spec,
            hidden_layer_sizes=[64, 32],
            name='qf1'
        )

        qf2 = NNQFunction2(
            env_spec=env.spec,
            hidden_layer_sizes=[64, 32],
            name='qf2'
        )

        vf = NNVFunction(
            env_spec=env.spec,
            hidden_layer_sizes=[64, 32]
        )

        initial_exploration_policy = UniformPolicy2(env_spec=env.spec)

        policy = CategoricalPolicy(
            env_spec=env.spec,
            hidden_layer_sizes=(64, 32),
        )

        algo = SACD(
            base_kwargs=base_kwargs,
            env=env,
            policy=policy,
            initial_exploration_policy=initial_exploration_policy,
            pool=pool,
            qf1=qf1,
            qf2=qf2,
            vf=vf,
            #plotter=plotter,
            lr=1e-3,
            scale_reward=10.0,
            discount=0.99,
            tau=1e-4,
            target_update_interval=1,
            #reparameterize=False,
            save_full_state=False
        )

        algo.train()


if __name__ == "__main__":

    run_experiment()
    # run_sac_experiment(
    #     run_experiment,
    #     mode="local",
    #     exp_prefix='SACD-VoltVar4',
    #     n_parallel=4,
    #     seed=1,
    #     snapshot_mode="last",
    # )



