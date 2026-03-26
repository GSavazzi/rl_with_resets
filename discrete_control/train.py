# coding=utf-8
# Copyright 2021 The Atari 100k Precipice Authors.
# ... (license header unchanged)

r"""Entry point for Atari 100k experiments.
# [CHG-1] tf.compat.v1.disable_v2_behavior() removed
# [CHG-2] 'sess' removed; seed=None default added
# [CHG-3] Removed unused import base_train
# [CHG-4] functools.partial replaced with closure
# [CHG-5] jax.interpreters.xla shim wrapped in try/except
# [CHG-6] create_agent handles environment=None — modern Dopamine Runner
#          passes environment=None at agent construction time; the real env
#          is created afterward. A temporary env is built from gin-configured
#          game_name to read num_actions.
"""

import os

try:
    import jax.core
    import jax.interpreters.xla as _jax_xla
    if not hasattr(_jax_xla, 'pytype_aval_mappings'):
        _jax_xla.pytype_aval_mappings = jax.core.pytype_aval_mappings
    del _jax_xla
except (ImportError, AttributeError):
    pass

from absl import app
from absl import flags
from absl import logging
from dopamine.discrete_domains import run_experiment
from dopamine.discrete_domains import atari_lib       # [CHG-6] needed for fallback env
import numpy as np
import tensorflow.compat.v2 as tf
from discrete_control import eval_run_experiment
from discrete_control.agents import rainbow_agent
import gin

FLAGS = flags.FLAGS
CONFIGS_DIR = './configs'
AGENTS = ['rainbow', 'der', 'dopamine_der', 'DrQ', 'OTRainbow', "SPR"]

flags.DEFINE_string('base_dir', './', 'Base directory for all required sub-directories.')
flags.DEFINE_multi_string('gin_files', [], 'List of paths to gin configuration files.')
flags.DEFINE_multi_string('gin_bindings', [], 'Gin bindings to override configuration values.')
flags.DEFINE_enum('agent', 'rainbow', AGENTS, 'Name of the agent.')
flags.DEFINE_integer('run_number', 1, 'Run number.')
flags.DEFINE_integer('agent_seed', None, 'If None, use the run_number')
flags.DEFINE_string('load_replay_dir', None,
    'Directory to load the initial replay buffer from a fixed dataset. '
    'If None, no transitions are loaded.')
flags.DEFINE_string('tag', None, 'Tag for this run')
flags.DEFINE_boolean('save_replay', False,
    'Whether to save agent\'s final replay buffer as a fixed dataset to '
    '${base_dir}/replay_logs.')
flags.DEFINE_integer('load_replay_number', None,
    'Index of the replay run to load the initial replay buffer from a fixed '
    'dataset. If None, uses the `run_number`.')
flags.DEFINE_boolean('max_episode_eval', True,
    'Whether to use `MaxEpisodeEvalRunner` or not.')
flags.DEFINE_boolean('wandb', False, 'Also log to wandb.')


def load_gin_configs(gin_files, gin_bindings):
    gin.parse_config_files_and_bindings(gin_files,
                                        bindings=gin_bindings,
                                        skip_unknown=False)


def init_wandb(base_dir, seed, tag=None, agent=None):
    os.environ['WANDB_MODE'] = 'offline'
    os.makedirs(base_dir, exist_ok=True)
    import wandb
    from gin import config
    clean_cfg = {k[1]: v for k, v in config._CONFIG.items()}
    clean_cfg["seed"] = seed
    clean_cfg["tag"] = tag
    clean_cfg["agent"] = agent
    wandb.init(config=clean_cfg, sync_tensorboard=True, dir=base_dir)


def create_load_replay_dir(xm_params):
    problem_name, run_number = '', ''
    for param, value in xm_params.items():
        if param.endswith('game_name'):
            problem_name = value
        elif param.endswith('run_number'):
            run_number = str(value)
    replay_dir = FLAGS.load_replay_dir
    if replay_dir:
        replay_number = str(FLAGS.load_replay_number) if FLAGS.load_replay_number else run_number
        replay_dir = os.path.join(replay_dir, problem_name, replay_number, 'replay_logs')
    return replay_dir


# [CHG-2] 'sess' removed; seed=None so gin never treats it as required.
# [CHG-6] environment=None: modern Dopamine Runner passes None at construction
#          time. Fall back to a temporary environment to read num_actions.
def create_agent(environment=None, seed=None, summary_writer=None):
    """Helper function for creating agent."""
    if environment is None:
        environment = atari_lib.create_atari_environment()
    return rainbow_agent.JaxSPRAgent(
        num_actions=environment.action_space.n,
        seed=seed,
        summary_writer=summary_writer)


def set_random_seed(seed):
    logging.info('Setting random seed: %d', seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def main(unused_argv):
    logging.set_verbosity(logging.INFO)
    # [CHG-1] tf.compat.v1.disable_v2_behavior() removed.

    base_dir = FLAGS.base_dir
    gin_files = FLAGS.gin_files
    gin_bindings = FLAGS.gin_bindings
    set_random_seed(FLAGS.run_number)
    run_experiment.load_gin_configs(gin_files, gin_bindings)
    if FLAGS.wandb:
        init_wandb(base_dir, FLAGS.run_number, FLAGS.tag, FLAGS.agent)

    # [CHG-4] Closure replaces functools.partial — avoids double seed injection.
    _seed = FLAGS.run_number if not FLAGS.agent_seed else FLAGS.agent_seed

    def create_agent_fn(environment=None, seed=None, summary_writer=None):
        return create_agent(environment, seed=_seed, summary_writer=summary_writer)

    if FLAGS.max_episode_eval:
        runner_fn = eval_run_experiment.DataEfficientAtariRunner
        logging.info('Using MaxEpisodeEvalRunner for evaluation.')
        runner = runner_fn(base_dir, create_agent_fn)
    else:
        runner = run_experiment.Runner(base_dir, create_agent_fn)
    runner.run_experiment()


if __name__ == '__main__':
    flags.mark_flag_as_required('base_dir')
    app.run(main)