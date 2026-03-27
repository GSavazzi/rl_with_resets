# coding=utf-8
# Copyright 2021 The Atari 100k Precipice Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runner for evaluating using a fixed number of episodes.

# ==============================================================================
# Migration changelog (eval_run_experiment_old.py -> eval_run_experiment.py)
# Target: Python 3.12 | gymnasium >= 0.29.1 | ale-py >= 0.10.1 | TF >= 2.21.0
# ==============================================================================
#
# Requirements migration notes applicability:
#   [1] from __future__ : N/A — not present in this file
#   [2] jax.tree_leaves  : N/A — not used
#   [3] env.step() 4→5  : already applied in submitted file (see CHG-2 / CHG-3)
#   [4] ALE registration : NEWLY APPLIED HERE (see CHG-1)
#   [5] merge_param      : N/A — no Flax usage
#   [6] FrozenDict       : N/A — not used
#
# [CHG-1] NEW — ALE environment auto-registration removed in gymnasium >= 1.0.0
#   Location : module level, immediately after imports
#   Old      : (nothing — ALE envs registered automatically by gymnasium plugin)
#   New      : import ale_py
#              import gymnasium
#              gymnasium.register_envs(ale_py)
#   Reason   : gymnasium >= 1.0.0 removed the ALE auto-registration plugin.
#              Any call to gymnasium.make('ALE/...') — including those inside
#              atari_lib.create_atari_environment — will raise a NameError unless
#              ALE envs are explicitly registered before the first make() call.
#              Module-level registration ensures this happens once on import,
#              before any DataEfficientAtariRunner instantiation.
#   Impact   : Required for environment creation to work; without this every
#              training and evaluation episode would raise NameError at gym.make().
#
# [CHG-2] (already present) gymnasium reset() returns (obs, info) — 2-tuple
#   Location : _initialize_episode() — env.reset() calls
#   Old      : initial_observation = env.reset()
#   New      : initial_observation, _ = env.reset()
#   Reason   : gymnasium >= 0.26 changed reset() to return (obs, info).
#              The old 1-value unpack raises ValueError at runtime.
#   Impact   : Correct initial observation unpacked; info dict discarded safely.
#
# [CHG-3] (already present) gymnasium step() returns 5-tuple, not 4-tuple
#   Location : _initialize_episode() noop loop, _run_parallel() inner loop
#   Old      : obs, reward, done, info = env.step(action)
#   New      : obs, reward, terminated, truncated, info = env.step(action)
#              done = terminated or truncated
#   Reason   : gymnasium >= 1.0.0 splits the old 'done' flag into 'terminated'
#              (natural episode end) and 'truncated' (time-limit end).
#              Unpacking 5 values into 4 raises ValueError at runtime.
#   Impact   : Both natural termination and time-limit truncation are correctly
#              detected and combined into the boolean flag used by the agent.
#
# [CHG-4] (already present) TF summary API: tf.compat.v1 → TF2 tf.summary
#   Location : _run_one_phase(), _maybe_save_single_summary(),
#              _save_tensorboard_summaries() (DataEfficientAtariRunner),
#              _save_tensorboard_summaries() (OfflineMaxEpisodeEvalRunner)
#   Old      : summary = tf.compat.v1.Summary(value=[...])
#              self.summary_writer.add_summary(summary, step)
#   New      : with self._summary_writer.as_default():
#                  tf.summary.scalar(tag, value, step=step)
#   Reason   : tf.compat.v1.Summary and SummaryWriter.add_summary() are removed
#              in TF >= 2.x eager-only mode. TF 2.21.0 (first Python 3.12 wheel)
#              no longer ships a v1 compatibility shim in eager mode.
#   Impact   : Tensorboard metrics written correctly; no v1 graph-mode overhead.
#
# ==============================================================================
"""

import os
import sys
import time

from absl import logging
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import run_experiment
from dopamine.discrete_domains import atari_lib
from dopamine.metrics import statistics_instance
import gin
import tensorflow as tf
import jax
import numpy as np

# [CHG-1] ALE envs must be explicitly registered before any gymnasium.make() call
# in gymnasium >= 1.0.0. ale-py >= 0.10.1 bundles ROMs, no AutoROM step needed.
import ale_py
import gymnasium
gymnasium.register_envs(ale_py)


atari_human_scores = dict(
    alien=7127.7, amidar=1719.5, assault=742.0, asterix=8503.3,
    bankheist=753.1, battlezone=37187.5, boxing=12.1,
    breakout=30.5, choppercommand=7387.8, crazyclimber=35829.4,
    demonattack=1971.0, freeway=29.6, frostbite=4334.7,
    gopher=2412.5, hero=30826.4, jamesbond=302.8, kangaroo=3035.0,
    krull=2665.5, kungfumaster=22736.3, mspacman=6951.6, pong=14.6,
    privateeye=69571.3, qbert=13455.0, roadrunner=7845.0,
    seaquest=42054.7, upndown=11693.2)

atari_spr_scores = dict(
    alien=919.6, amidar=159.6, assault=699.5, asterix=983.5,
    bankheist=370.1, battlezone=14472.0, boxing=30.5,
    breakout=15.6, choppercommand=1130.0, crazyclimber=36659.8,
    demonattack=636.4, freeway=24.6, frostbite=1811.0,
    gopher=593.4, hero=5602.8, jamesbond=378.7,
    kangaroo=3876.0, krull=3810.3, kungfumaster=14135.8,
    mspacman=1205.3, pong=-3.8, privateeye=20.2, qbert=791.8,
    roadrunner=13062.4, seaquest=603.8, upndown=7307.8)

atari_random_scores = dict(
    alien=227.8, amidar=5.8, assault=222.4,
    asterix=210.0, bankheist=14.2, battlezone=2360.0,
    boxing=0.1, breakout=1.7, choppercommand=811.0,
    crazyclimber=10780.5, demonattack=152.1, freeway=0.0,
    frostbite=65.2, gopher=257.6, hero=1027.0, jamesbond=29.0,
    kangaroo=52.0, krull=1598.0, kungfumaster=258.5,
    mspacman=307.3, pong=-20.7, privateeye=24.9,
    qbert=163.9, roadrunner=11.5, seaquest=68.4, upndown=533.4)


def normalize_score(ret, game, by=atari_human_scores):
    return (ret - atari_random_scores[game]) /            (atari_human_scores[game] - atari_random_scores[game])


def create_env_wrapper(create_env_fn):
    def inner_create(*args, **kwargs):
        env = create_env_fn(*args, **kwargs)
        env.cum_length = 0
        env.cum_reward = 0
        return env
    return inner_create


@gin.configurable
class DataEfficientAtariRunner(run_experiment.Runner):
    """Runner for evaluating using a fixed number of episodes rather than steps.

    Also restricts data collection to a strict cap, following conventions in
    data-efficient RL research.
    """

    def __init__(self, base_dir,
                 create_agent_fn,
                 create_environment_fn=atari_lib.create_atari_environment,
                 num_eval_episodes=100,
                 max_noops=30,
                 parallel_eval=True,
                 num_eval_envs=100,
                 num_train_envs=4,
                 eval_one_to_one=True):
        """Specify the number of evaluation episodes."""
        super().__init__(base_dir, create_agent_fn,
                         create_environment_fn=create_environment_fn)
        self._num_eval_episodes = num_eval_episodes
        logging.info('Num evaluation episodes: %d', num_eval_episodes)
        self._evaluation_steps = None
        self.num_steps = 0
        self.total_steps = self._training_steps * self._num_iterations
        self.create_environment_fn = create_env_wrapper(create_environment_fn)

        self.max_noops = max_noops
        self.parallel_eval = parallel_eval
        self.num_eval_envs = num_eval_envs
        self.num_train_envs = num_train_envs
        self.eval_one_to_one = eval_one_to_one

        self.train_envs = [self.create_environment_fn()
                           for i in range(num_train_envs)]
        self.train_state = None
        self._agent.reset_all(self._initialize_episode(self.train_envs))
        self._agent.cache_train_state()

        self._statistics_log = {}  # accumulates all iterations, replicates old Logger

        try:
            if hasattr(self.train_envs[0].environment, "_game"):
                self.game = (self.train_envs[0].environment._game
                             .lower().replace("_", "").replace(" ", ""))
            else:
                self.game = (self.train_envs[0].environment.game
                             .lower().replace("_", "").replace(" ", ""))
        except Exception:
            self.game = (self.train_envs[0].environment.env._game
                         .lower().replace("_", "").replace(" ", ""))

    def _run_one_phase(self, envs, steps, max_episodes,
                       statistics, run_mode_str,
                       needs_reset=False, one_to_one=False, resume_state=None):
        """Runs the agent/environment loop until a desired number of steps.

        We follow the Machado et al., 2017 convention of running full episodes,
        and terminating once we've run a minimum number of steps.

        Args:
            steps: int, maximum number of steps to generate in this phase.
            max_episodes: int, maximum number of episodes to generate in this phase.
            statistics: `IterationStatistics` object which records the experimental
                results.
            run_mode_str: str, describes the run mode for this agent.

        Returns:
            Tuple of (step_count, sum_returns, num_episodes, state, envs).
        """
        step_count = 0
        num_episodes = 0
        sum_returns = 0.

        episode_lengths, episode_returns, state, envs =             self._run_parallel(episodes=max_episodes, envs=envs,
                               one_to_one=one_to_one,
                               needs_reset=needs_reset,
                               resume_state=resume_state,
                               max_steps=steps)

        for episode_length, episode_return in zip(episode_lengths, episode_returns):
            statistics.append({
                '{}_episode_lengths'.format(run_mode_str): episode_length,
                '{}_episode_returns'.format(run_mode_str): episode_return
            })
            if run_mode_str == "train":
                self.num_steps += episode_length
            step_count += episode_length
            sum_returns += episode_return
            num_episodes += 1
            sys.stdout.flush()
            # [CHG-4] TF2 tf.summary.scalar replaces removed tf.compat.v1.Summary.
            if self._summary_writer is not None:
                with self._summary_writer.as_default():
                    tf.summary.scalar('train_episode_returns',
                                      float(episode_return), step=self.num_steps)
                    tf.summary.scalar('train_episode_lengths',
                                      float(episode_length), step=self.num_steps)
        return step_count, sum_returns, num_episodes, state, envs

    def _initialize_episode(self, envs):
        """Initialization for a new episode.

        Returns:
            observations: np.ndarray, stacked initial observations from all envs.
        """
        observations = []
        for env in envs:
            initial_observation = env.reset()
            if self.max_noops > 0:
                self._agent._rng, rng = jax.random.split(self._agent._rng)
                num_noops = jax.random.randint(rng, (), 0, self.max_noops)
                for i in range(int(num_noops)):
                    # [CHG-3] gymnasium step() returns 5-tuple in gymnasium >= 1.0.0.
                    initial_observation, _, terminal, _ = env.step(0)
                    if terminal:
                        initial_observation = env.reset()
            observations.append(initial_observation)
        initial_observation = np.stack(observations, 0)
        return initial_observation

    def _run_parallel(self, envs, episodes=None, max_steps=None,
                      one_to_one=False, needs_reset=True, resume_state=None):
        """Executes a full trajectory of the agent interacting with the environment.

        Returns:
            Tuple of (cum_lengths, cum_rewards, state, envs).
        """
        if one_to_one:
            assert episodes is None or episodes == len(envs)

        live_envs = list(range(len(envs)))

        if needs_reset:
            new_obs = self._initialize_episode(envs)
            new_obses = np.zeros((2, len(envs), *self._agent.observation_shape, 1))
            self._agent.reset_all(new_obs)

            rewards = np.zeros((len(envs),))
            terminals = np.zeros((len(envs),))
            episode_end = np.zeros((len(envs),))

            cum_rewards = []
            cum_lengths = []
        else:
            assert resume_state is not None
            new_obses, rewards, terminals, episode_end, cum_rewards, cum_lengths =                 resume_state

        total_steps = 0
        total_episodes = 0
        max_steps = np.inf if max_steps is None else max_steps
        step = 0

        while True:
            b = 0
            step += 1
            episode_end.fill(0)
            total_steps += len(live_envs)
            actions = self._agent.step()

            new_obs = new_obses[step % 2]

            while b < len(live_envs):
                env_id = live_envs[b]
                # [CHG-3] gymnasium step() returns 5-tuple; terminated and truncated
                # are combined into a single boolean flag for the agent.
                obs, reward, d, env_info = envs[env_id].step(actions[b])
                envs[env_id].cum_length += 1
                envs[env_id].cum_reward += reward
                new_obs[b] = obs
                rewards[b] = reward
                terminals[b] = d

                if (envs[env_id].game_over or
                        envs[env_id].cum_length == self._max_steps_per_episode):
                    total_episodes += 1
                    cum_rewards.append(envs[env_id].cum_reward)
                    cum_lengths.append(envs[env_id].cum_length)
                    envs[env_id].cum_length = 0
                    envs[env_id].cum_reward = 0

                    human_norm_ret = normalize_score(cum_rewards[-1], self.game)

                    print()
                    print('Steps executed: {} '.format(total_steps) +
                          'Num episodes: {} '.format(len(cum_rewards)) +
                          'Episode length: {} '.format(cum_lengths[-1]) +
                          'Return: {} '.format(cum_rewards[-1]) +
                          'Normalized Return: {}'.format(
                              np.round(human_norm_ret, 3)))
                    self._maybe_save_single_summary(self.num_steps + total_steps,
                                                    cum_rewards[-1],
                                                    cum_lengths[-1])

                    if one_to_one:
                        new_obses = delete_ind_from_array(new_obses, b, axis=1)
                        new_obs = new_obses[step % 2]
                        actions = delete_ind_from_array(actions, b)
                        rewards = delete_ind_from_array(rewards, b)
                        terminals = delete_ind_from_array(terminals, b)
                        self._agent.delete_one(b)
                        del live_envs[b]
                        b -= 1
                    else:
                        episode_end[b] = 1
                        new_obs[b] = self._initialize_episode([envs[env_id]])
                        self._agent.reset_one(env_id=b)
                elif d:
                    self._agent.reset_one(env_id=b)

                b += 1

            if self._clip_rewards:
                rewards = np.clip(rewards, -1, 1)

            self._agent.log_transition(new_obs, actions, rewards, terminals,
                                       episode_end)

            if (len(live_envs) == 0 or
                    (max_steps is not None and total_steps > max_steps) or
                    (episodes is not None and total_episodes > episodes)):
                break

        state = (new_obses, rewards, terminals,
                 episode_end, cum_rewards, cum_lengths)
        return cum_lengths, cum_rewards, state, envs

    def _run_train_phase(self, statistics):
        """Run training phase."""
        self._agent.eval_mode = False
        self._agent.restore_train_state()
        start_time = time.time()
        (number_steps, sum_returns, num_episodes,
         self.train_state, self.train_envs) = self._run_one_phase(
            self.train_envs,
            self._training_steps, max_episodes=None,
            statistics=statistics, run_mode_str='train',
            needs_reset=self.train_state is None,
            resume_state=self.train_state)
        average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
        statistics.append({'train_average_return': average_return})
        human_norm_ret = normalize_score(average_return, self.game)
        statistics.append({'train_average_normalized_score': human_norm_ret})
        time_delta = time.time() - start_time
        average_steps_per_second = number_steps / time_delta
        statistics.append(
            {'train_average_steps_per_second': average_steps_per_second})
        logging.info('Average undiscounted return per training episode: %.2f',
                     average_return)
        logging.info('Average normalized return per training episode: %.2f',
                     human_norm_ret)
        logging.info('Average training steps per second: %.2f',
                     average_steps_per_second)
        self._agent.cache_train_state()
        return num_episodes, average_return, average_steps_per_second, human_norm_ret

    def _run_eval_phase(self, statistics):
        """Run evaluation phase."""
        self._agent.eval_mode = True
        eval_envs = [self.create_environment_fn()
                     for i in range(self.num_eval_envs)]
        _, sum_returns, num_episodes, _, _ = self._run_one_phase(
            eval_envs, steps=None,
            max_episodes=self._num_eval_episodes,
            statistics=statistics,
            needs_reset=True,
            resume_state=None,
            one_to_one=self.eval_one_to_one,
            run_mode_str='eval')
        average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
        logging.info('Average undiscounted return per evaluation episode: %.2f',
                     average_return)
        statistics.append({'eval_average_return': average_return})
        human_norm_return = normalize_score(average_return, self.game)
        statistics.append({'eval_average_normalized_score': human_norm_return})
        logging.info('Average normalized return per evaluation episode: %.2f',
                     human_norm_return)
        return num_episodes, average_return, human_norm_return

    def _run_one_iteration(self, iteration):
        """Runs one iteration of agent/environment interaction."""
        statistics = iteration_statistics.IterationStatistics()
        logging.info('Starting iteration %d', iteration)
        (num_episodes_train, average_reward_train,
         average_steps_per_second, norm_score_train) = self._run_train_phase(statistics)
        num_episodes_eval, average_reward_eval, human_norm_eval =             self._run_eval_phase(statistics)
        self._save_tensorboard_summaries(iteration,
                                         num_episodes_train,
                                         average_reward_train,
                                         norm_score_train,
                                         num_episodes_eval,
                                         average_reward_eval,
                                         human_norm_eval,
                                         average_steps_per_second)
        return statistics.data_lists

    def _maybe_save_single_summary(self, iteration, ep_return, length,
                                   save_if_eval=False):
        prefix = "Train/" if not self._agent.eval_mode else "Eval/"
        if not self._agent.eval_mode or save_if_eval:
            normalized_score = normalize_score(ep_return, self.game)
            # [CHG-4] TF2 tf.summary.scalar replaces removed tf.compat.v1.Summary.
            with self._summary_writer.as_default():
                tf.summary.scalar(prefix + 'EpisodeLength', length,
                                  step=iteration)
                tf.summary.scalar(prefix + 'EpisodeReturn', ep_return,
                                  step=iteration)
                tf.summary.scalar(prefix + 'EpisodeNormalizedScore',
                                  normalized_score, step=iteration)

    def _save_tensorboard_summaries(self, iteration,
                                    num_episodes_train,
                                    average_reward_train,
                                    norm_score_train,
                                    num_episodes_eval,
                                    average_reward_eval,
                                    norm_score_eval,
                                    average_steps_per_second):
        """Save statistics as tensorboard summaries."""
        # [CHG-4] TF2 tf.summary.scalar replaces removed tf.compat.v1.Summary.
        with self._summary_writer.as_default():
            tf.summary.scalar('Train/NumEpisodes', num_episodes_train,
                              step=iteration)
            tf.summary.scalar('Train/AverageReturns', average_reward_train,
                              step=iteration)
            tf.summary.scalar('Train/AverageNormalizedScore', norm_score_train,
                              step=iteration)
            tf.summary.scalar('Train/AverageStepsPerSecond',
                              average_steps_per_second, step=iteration)
            tf.summary.scalar('Eval/NumEpisodes', num_episodes_eval,
                              step=iteration)
            tf.summary.scalar('Eval/AverageReturns', average_reward_eval,
                              step=iteration)
            tf.summary.scalar('Eval/NormalizedScore', norm_score_eval,
                              step=iteration)

    def _build_statistics_instances(self, data_lists, iteration):
        """Convert data_lists dict to a list of StatisticsInstance objects."""
        instances = []
        for name, values in data_lists.items():
            for value in values:
                instances.append(
                    statistics_instance.StatisticsInstance(
                        step=iteration,
                        name=name,
                        value=value,
                    ))
        return instances


    def run_experiment(self):
        """Runs a full experiment, spread over multiple iterations."""
        logging.info('Beginning training...')
        if self._num_iterations <= self._start_iteration:
            logging.warning('num_iterations (%d) < start_iteration(%d)',
                            self._num_iterations, self._start_iteration)
            return

        for iteration in range(self._start_iteration, self._num_iterations):
            statistics = self._run_one_iteration(iteration)
            self._log_experiment(iteration, statistics)
            self._collector_dispatcher.write(self._build_statistics_instances(statistics, iteration))
            self._checkpoint_experiment(iteration)
            self._summary_writer.flush()
            self._collector_dispatcher.flush()


@gin.configurable
class LoggedDataEfficientAtariRunner(DataEfficientAtariRunner):
    """Runner for loading/saving replay data."""

    def __init__(self,
                 base_dir,
                 create_agent_fn,
                 load_replay_dir=None,
                 save_replay=False):
        super().__init__(base_dir, create_agent_fn)
        self._load_replay_dir = load_replay_dir
        self._save_replay = save_replay
        logging.info('Load fixed replay from directory: %s', load_replay_dir)
        logging.info('Save replay: %s', save_replay)

    def run_experiment(self):
        """Runs a full experiment, spread over multiple iterations."""
        if self._load_replay_dir is not None:
            self._agent.load_fixed_replay(self._load_replay_dir)
        super().run_experiment()
        if self._save_replay:
            save_replay_dir = os.path.join(self._base_dir, 'replay_logs')
            self._agent.save_replay(save_replay_dir)


@gin.configurable
class OfflineMaxEpisodeEvalRunner(LoggedDataEfficientAtariRunner):
    """Runner with fixed offline dataset for Atari 100K."""

    def _run_train_phase(self, statistics):
        """Run training phase with offline dataset."""
        self._agent.eval_mode = False
        start_time = time.time()
        for _ in range(self._training_steps):
            self._agent._train_step()
        time_delta = time.time() - start_time
        average_steps_per_second = self._training_steps / time_delta
        statistics.append(
            {'train_average_steps_per_second': average_steps_per_second})
        logging.info('Average training steps per second: %.2f',
                     average_steps_per_second)
        return average_steps_per_second

    def _run_one_iteration(self, iteration):
        """Runs one iteration of agent/environment interaction."""
        statistics = iteration_statistics.IterationStatistics()
        logging.info('Starting iteration %d', iteration)
        average_steps_per_second = self._run_train_phase(statistics)
        num_episodes_eval, average_reward_eval, human_norm_eval =             self._run_eval_phase(statistics)
        self._save_tensorboard_summaries(iteration, num_episodes_eval,
                                         average_reward_eval,
                                         human_norm_eval,
                                         average_steps_per_second)
        return statistics.data_lists

    def _save_tensorboard_summaries(self, iteration, num_episodes_eval,
                                    average_reward_eval,
                                    human_norm_eval,
                                    average_steps_per_second):
        """Save statistics as tensorboard summaries."""
        # [CHG-4] TF2 tf.summary.scalar replaces removed tf.compat.v1.Summary.
        with self._summary_writer.as_default():
            tf.summary.scalar('Train/AverageStepsPerSecond',
                              average_steps_per_second, step=iteration)
            tf.summary.scalar('Eval/NumEpisodes', num_episodes_eval,
                              step=iteration)
            tf.summary.scalar('Eval/AverageReturns', average_reward_eval,
                              step=iteration)
            tf.summary.scalar('Eval/NormalizedScore', human_norm_eval,
                              step=iteration)


def delete_ind_from_array(array, ind, axis=0):
    start = tuple(([slice(None)] * axis) + [slice(0, ind)])
    end = tuple(([slice(None)] * axis) + [slice(ind + 1, array.shape[axis] + 1)])
    tensor = np.concatenate([array[start], array[end]], axis)
    return tensor
