import os
import random
import sys, importlib.abc, importlib.machinery

class _TFPJaxOpsPatcher(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    _TARGET = "tensorflow_probability.python.internal.backend.jax.ops"
    _OLD    = "jax.interpreters.xla.pytype_aval_mappings"
    _NEW    = "jax.core.pytype_aval_mappings"

    def find_spec(self, fullname, path, target=None):
        if fullname != self._TARGET:
            return None
        # Delegate to remaining finders, then hijack the loader
        for finder in sys.meta_path[1:]:
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                spec.loader = self
                self._origin = spec.origin
                return spec
        return None

    def create_module(self, spec):
        return None  # use default semantics

    def exec_module(self, module):
        import pathlib
        src = pathlib.Path(self._origin).read_text()
        patched = src.replace(self._OLD, self._NEW)
        exec(compile(patched, self._origin, "exec"), module.__dict__)
        print("✅ TFP JAX ops patched in-memory.")

# Register once at script startup — must come before `import tensorflow_probability`
sys.meta_path.insert(0, _TFPJaxOpsPatcher())


import jax
import flax
import numpy as np
import jax.numpy as jnp
import optax
import tqdm
from absl import app, flags
from ml_collections import config_flags

from continuous_control.agents import DrQLearner
from continuous_control.datasets import ReplayBuffer
from continuous_control.evaluation import evaluate
from continuous_control.utils import make_env


FLAGS = flags.FLAGS


flags.DEFINE_string('exp', '', 'Experiment description (not actually used).')
flags.DEFINE_string('env_name', 'quadruped-run', 'Environment name.')
flags.DEFINE_string('save_dir', './out/', 'Logging dir.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 512, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(5e5), 'Number of environment steps.')
flags.DEFINE_integer('start_training', int(1e3),
                     'Number of environment steps to start training.')
flags.DEFINE_integer(
    'action_repeat', None,
    'Action repeat, if None, uses 2 or PlaNet default values.')
flags.DEFINE_integer('reset_interval', 25000, 'Periodicity of resets.')
flags.DEFINE_boolean('resets', False,
                     'Periodically reset last actor / critic layers.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
config_flags.DEFINE_config_file(
    'config',
    'configs/drq.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


PLANET_ACTION_REPEAT = {
    'cartpole-swingup': 8,
    'reacher-easy': 4,
    'cheetah-run': 4,
    'finger-spin': 2,
    'ball_in_cup-catch': 4,
    'walker-walk': 2
}


def main(_):
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    if FLAGS.save_video:
        video_train_folder = os.path.join(FLAGS.save_dir, 'video', 'train')
        video_eval_folder = os.path.join(FLAGS.save_dir, 'video', 'eval')
    else:
        video_train_folder = None
        video_eval_folder = None

    if FLAGS.action_repeat is not None:
        action_repeat = FLAGS.action_repeat
    else:
        action_repeat = PLANET_ACTION_REPEAT.get(FLAGS.env_name, 2)

    all_kwargs = FLAGS.flag_values_dict()
    all_kwargs.update(all_kwargs.pop('config'))
    kwargs = dict(FLAGS.config)

    gray_scale = kwargs.pop('gray_scale')
    image_size = kwargs.pop('image_size')

    def make_pixel_env(seed, video_folder):
        return make_env(FLAGS.env_name,
                        seed,
                        video_folder,
                        action_repeat=action_repeat,
                        image_size=image_size,
                        frame_stack=3,
                        from_pixels=True,
                        gray_scale=gray_scale)

    env = make_pixel_env(FLAGS.seed, video_train_folder)
    eval_env = make_pixel_env(FLAGS.seed + 42, video_eval_folder)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    assert kwargs.pop('algo') == 'drq'
    replay_buffer_size = kwargs.pop('replay_buffer_size')

    obs_demo = env.observation_space.sample()
    action_demo = env.action_space.sample()
    agent = DrQLearner(FLAGS.seed,
                       obs_demo[np.newaxis],
                       action_demo[np.newaxis], **kwargs)

    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(env.observation_space, action_dim,
                                 replay_buffer_size or FLAGS.max_steps)

    eval_returns = []
    observation, _ = env.reset()                         # reset: obs, info
    done = False
    for i in tqdm.tqdm(range(1, FLAGS.max_steps // action_repeat + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)

        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # mask: 0 only on true termination, 1 on mid-episode or truncation
        mask = 0.0 if terminated else 1.0
        replay_buffer.insert(observation, action, reward, mask,
                             float(terminated or truncated),
                             next_observation)
        observation = next_observation

        if done:
            observation, _ = env.reset()
            done = False

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size)
            agent.update(batch)

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes)

            eval_returns.append(
                (info['total']['timesteps'], eval_stats['return']))
            np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])

        if FLAGS.resets and i % FLAGS.reset_interval == 0:
            # shared enc params: 388416
            # critic head(s) params: 366232
            # actor head params: 286882
            # so we reset roughly half of the agent (both layer and param wise)

            # save encoder parameters
            old_critic_enc = agent.critic.params['SharedEncoder']
            # target critic has its own copy of encoder
            old_target_critic_enc = agent.target_critic.params['SharedEncoder']
            # save encoder optimizer statistics
            old_critic_enc_opt = agent.critic.opt_state_enc

            # create new agent: note that the temperature is new as well
            agent = DrQLearner(FLAGS.seed + i,
                               env.observation_space.sample()[np.newaxis],
                               env.action_space.sample()[np.newaxis], **kwargs)

            # resetting critic: copy encoder parameters and optimizer statistics
            new_critic_params = {**agent.critic.params,
                                 'SharedEncoder': old_critic_enc}
            agent.critic = agent.critic.replace(params=new_critic_params,
                                                opt_state_enc=old_critic_enc_opt)

            # resetting actor: actor in DrQ uses critic's encoder
            new_actor_params = {**agent.actor.params,
                                'SharedEncoder': old_critic_enc}
            agent.actor = agent.actor.replace(params=new_actor_params)

            # resetting target critic
            new_target_critic_params = {**agent.target_critic.params,
                                        'SharedEncoder': old_target_critic_enc}
            agent.target_critic = agent.target_critic.replace(
                params=new_target_critic_params)


if __name__ == '__main__':
    app.run(main)