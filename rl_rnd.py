# RND paper: https://arxiv.org/pdf/1810.12894.pdf
# PPO paper: https://arxiv.org/pdf/1707.06347.pdf
# PPO algorithm: https://spinningup.openai.com/en/latest/_images/math/0a399dc49e3b45664a7edaf485ab5c23a7282f43.svg

import os
import argparse
import re
import time
from multiprocessing.pool import ThreadPool as Pool
import traceback
import gzip
from shutil import rmtree
import math
import random
from datetime import datetime
from PIL import Image

import numpy as np

np.set_printoptions(precision=4)
try:
    import cupy as cp
except Exception as e:
    pass
import imageio

import gym
try:
    import gym_ple
except Exception as e:
    pass

import chainer

chainer.global_config.dtype = np.float64
from chainer import dataset
import chainer.functions as F
from chainer import training
from chainer.training import extensions, Trainer
import chainer.links as L

try:
    from dnc import DNC
except Exception as e:
    pass

ACTIONS = {
    'PixelCopter-v0': [0, 1],
}
ID = "rl_rnd"


def log(id, message):
    print(str(datetime.now()) + " [" + str(id) + "] " + str(message))


class Dataset(dataset.DatasetMixin):
    def __init__(self, rollouts_dir='', is_rnd=False):
        self.rollouts_dir = rollouts_dir
        self.is_rnd = is_rnd

    def reset(self):
        rollouts = os.listdir(self.rollouts_dir)
        rollouts_counts = {}
        for rollout in rollouts:
            count_file = os.path.join(self.rollouts_dir, rollout, "count")
            if os.path.exists(count_file):
                with open(count_file, 'r') as count_file:
                    count = int(count_file.read())
                    rollouts_counts[rollout] = count - 1
        rollouts = list(rollouts_counts.keys())
        random.shuffle(rollouts)
        longest_rollout = sorted(rollouts, key=lambda x: -rollouts_counts[x])[0]
        rollouts.remove(longest_rollout)
        rollouts = [longest_rollout] + rollouts
        self.rollouts = rollouts
        self.num_rollouts = len(rollouts)

    def __len__(self):
        return self.num_rollouts

    def get_example(self, i):
        if not self.is_rnd:
            frames_file = os.path.join(self.rollouts_dir, self.rollouts[i], "stacked_frames.npy.gz")
        else:
            frames_file = os.path.join(self.rollouts_dir, self.rollouts[i], "stacked_frames_rnd.npy.gz")
        with gzip.GzipFile(frames_file, "r") as file:
            frames = np.load(file)
        misc_file = os.path.join(self.rollouts_dir, self.rollouts[i], "misc.npz")
        npz = np.load(misc_file)
        actions = npz['actions'].astype(np.float64)
        actions_pre_softmax = npz['actions_pre_softmax'].astype(np.float64)
        rewards = npz['rewards_target'].astype(np.float64).reshape(-1, 1)
        returns_extrinsic = npz['returns_extrinsic'].astype(np.float64).reshape(-1, 1)
        returns_intrinsic = npz['returns_intrinsic'].astype(np.float64).reshape(-1, 1)
        advantages = npz['advantages_combined'].astype(np.float64).reshape(-1, 1)
        npz.close()
        frames = frames[:-1]
        return frames, actions, actions_pre_softmax, rewards, returns_extrinsic, returns_intrinsic, advantages


class Model(chainer.Chain):
    def __init__(self, z_dim, rnn_hidden_dim, rnn_hidden_layers, final_hidden_dim, final_hidden_layers, final_dim,
                 dnc, epsilon_ppo, beta_entropy, is_value_net=False, is_rnd_net=False):
        self.z_dim = z_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_hidden_layers = rnn_hidden_layers
        self.final_hidden_dim = final_hidden_dim
        self.final_hidden_layers = final_hidden_layers
        self.final_dim = final_dim
        self.dnc = dnc
        self.epsilon_ppo = epsilon_ppo
        self.beta_entropy = beta_entropy
        self.is_value_net = is_value_net
        self.is_rnd_net = is_rnd_net

        init_dict = {}

        init_dict["encoder_conv0"] = L.Convolution2D(None, 32, 8, 4)
        init_dict["encoder_conv1"] = L.Convolution2D(None, 64, 4, 2)
        init_dict["encoder_conv2"] = L.Convolution2D(None, 64, 3, 1)
        init_dict["encoder_z"] = L.Linear(None, z_dim)

        if dnc:
            N, W, R, K = dnc.split(",")
            # X, Y, N, W, R, rnn_hidden_dim, K, gpu=-1, gpu_for_nn_only=False
            init_dict["rnn_layer"] = DNC(z_dim,
                                         final_dim,
                                         int(N),
                                         int(W),
                                         int(R),
                                         rnn_hidden_dim,
                                         int(K))
        else:
            for i in range(rnn_hidden_layers):
                init_dict["rnn_layer_" + str(i)] = L.LSTM(None, rnn_hidden_dim)

        for i in range(final_hidden_layers):
            init_dict["final_layer_" + str(i)] = L.Linear(None, final_hidden_dim)
        init_dict["final_output_layer"] = L.Linear(None, final_dim)

        super(Model, self).__init__(**init_dict)

    def encode(self, frames):
        if len(frames.shape) == 3:
            frames = F.expand_dims(frames, 0)
        h = F.leaky_relu(self.encoder_conv0(frames))
        h = F.leaky_relu(self.encoder_conv1(h))
        h = F.leaky_relu(self.encoder_conv2(h))
        h = F.reshape(h, (h.shape[0], -1))
        z = self.encoder_z(h)
        return z

    def model(self, z_t):
        rnn_hidden_layers = self.rnn_hidden_layers
        if len(z_t.shape) == 1:
            z_t = F.expand_dims(z_t, 0)
        h = z_t
        for i in range(rnn_hidden_layers):
            h = self["rnn_layer_" + str(i)](h)
        return h

    def final(self, h_t):
        final_hidden_layers = self.final_hidden_layers
        if len(h_t.shape) == 1:
            h_t = F.expand_dims(h_t, 0)
        h = h_t
        for i in range(final_hidden_layers):
            h = F.leaky_relu(self["final_layer_" + str(i)](h))
        output = self.final_output_layer(h)
        return output

    def get_loss_func(self):
        def lf(frames, actions, actions_pre_softmax_old, rewards, returns_extrinsic, returns_intrinsic, advantages,
               reset=True):
            if len(actions.shape) == 1:
                actions = F.expand_dims(actions, 0)
            if len(actions_pre_softmax_old.shape) == 1:
                actions_pre_softmax_old = F.expand_dims(actions_pre_softmax_old, 0)
            if len(rewards.shape) == 1:
                rewards = F.expand_dims(rewards, 0)
            if len(returns_extrinsic.shape) == 1:
                returns_extrinsic = F.expand_dims(returns_extrinsic, 0)
            if len(returns_intrinsic.shape) == 1:
                returns_intrinsic = F.expand_dims(returns_intrinsic, 0)
            if len(advantages.shape) == 1:
                advantages = F.expand_dims(advantages, 0)

            if reset:
                self.reset_state()

            loss = 0

            z_t = self.encode(frames)

            h_t = self.model(z_t)

            if z_t.shape[0] != h_t.shape[0]:
                h_t = h_t[:z_t.shape[0]]

            final_pred = self.final(h_t)

            if self.is_value_net:
                loss += F.sum((final_pred[:, 0] - returns_extrinsic) ** 2)
                loss += F.sum((final_pred[:, 1] - returns_intrinsic) ** 2)
            elif self.is_rnd_net:
                loss += F.sum((final_pred - rewards) ** 2)
            else:
                actions_pre_softmax_new = final_pred

                # Numerical stability:
                action_prob_old_clipped = F.clip(F.softmax(actions_pre_softmax_old, axis=1), 0 + 1e-5, 1 - 1e-5)
                action_prob_new_clipped = F.clip(F.softmax(actions_pre_softmax_new, axis=1), 0 + 1e-5, 1 - 1e-5)

                action_prob_old_sum = F.sum(action_prob_old_clipped * actions, axis=1)
                action_prob_new_sum = F.sum(action_prob_new_clipped * actions, axis=1)
                r_theta = action_prob_new_sum / action_prob_old_sum
                advantages_clipped = F.where(advantages > 0., (1 + self.epsilon_ppo) * advantages,
                                             (1 - self.epsilon_ppo) * advantages)
                # Maximize (gradient ascent) according to
                # https://spinningup.openai.com/en/latest/_images/math/0a399dc49e3b45664a7edaf485ab5c23a7282f43.svg.
                # To maximize, minimizing -cost, therefore the minus at the front:
                # https://stackoverflow.com/questions/38235648/is-there-an-easy-way-to-implement-a-optimizer-maximize-function-in-tensorflow
                loss_ppo = -F.sum(F.minimum(r_theta * advantages, advantages_clipped))

                # # Simply the policy gradient loss
                # actions_nll = -F.sum(actions * F.log(action_prob_clipped), axis=1)
                # loss_pg = F.sum(actions_nll * advantages.reshape(-1))

                # Entropy formula has a negative at the front, and if added to loss, we'd be minimizing entropy.
                # Therefore we take the negative of that (positive) to maximize entropy.
                loss_entropy = self.beta_entropy * -F.sum(
                    -F.sum(action_prob_new_clipped * F.log(action_prob_new_clipped), axis=1))

                loss += loss_ppo + loss_entropy

            return loss

        return lf

    def reset_state(self):
        rnn_hidden_layers = self.rnn_hidden_layers
        for i in range(rnn_hidden_layers):
            self["rnn_layer_" + str(i)].reset_state()

    def get_h(self):
        if self.dnc:
            return self.rnn_layer.get_h()
        else:
            rnn_hidden_layers = self.rnn_hidden_layers
            return self["rnn_layer_" + str(rnn_hidden_layers - 1)].h

    def get_c(self):
        if self.dnc:
            return self.rnn_layer.get_c()
        else:
            rnn_hidden_layers = self.rnn_hidden_layers
            return self["rnn_layer_" + str(rnn_hidden_layers - 1)].c

    def get_dnc_read_head(self):
        if self.dnc:
            return self.rnn_layer.get_read_head()
        else:
            return None

    def get_dnc_memory(self):
        if self.dnc:
            return self.rnn_layer.get_memory()
        else:
            return None


class TBPTTUpdater(training.updaters.StandardUpdater):
    def __init__(self, train_iter, device, optimizer, loss_func, sequence_length, minibatches,
                 rnn_single_record_training):
        self.sequence_length = sequence_length
        self.minibatches = minibatches
        self.rnn_single_record_training = rnn_single_record_training
        self.num_rollouts_processed = 0
        self.num_records_processed = 0
        self.previous_loss = 0
        self.target_iterations = 0
        super(TBPTTUpdater, self).__init__(
            train_iter, optimizer, device=device,
            loss_func=loss_func)

    def update_core(self):
        loss_for_reporting = 0.

        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        batch = train_iter.__next__()
        frames, actions, actions_pre_softmax, rewards, returns_extrinsic, returns_intrinsic, advantages = \
            self.converter(batch, self.device)
        frames = frames[0]
        actions = actions[0]
        actions_pre_softmax = actions_pre_softmax[0]
        rewards = rewards[0]
        returns_extrinsic = returns_extrinsic[0]
        returns_intrinsic = returns_intrinsic[0]
        advantages = advantages[0]
        loss = self.previous_loss

        def backprop(loss):
            loss /= self.num_records_processed  # Take the mean
            optimizer.target.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
            self.num_rollouts_processed = 0
            self.num_records_processed = 0
            self.previous_loss = 0

        for i in range(math.ceil(actions.shape[0] / self.sequence_length)):
            start_idx = i * self.sequence_length
            end_idx = (i + 1) * self.sequence_length
            this_frames = frames[start_idx:end_idx]
            this_actions = actions[start_idx:end_idx]
            this_actions_pre_softmax = actions_pre_softmax[start_idx:end_idx]
            this_rewards = rewards[start_idx:end_idx]
            this_returns_extrinsic = returns_extrinsic[start_idx:end_idx]
            this_returns_intrinsic = returns_intrinsic[start_idx:end_idx]
            this_advantages = advantages[start_idx:end_idx]
            if self.rnn_single_record_training:
                for j in range(this_actions.shape[0]):
                    loss += self.loss_func(this_frames[j],
                                           this_actions[j],
                                           this_actions_pre_softmax[j],
                                           this_rewards[j],
                                           this_returns_extrinsic[j],
                                           this_returns_intrinsic[j],
                                           this_advantages[j],
                                           True if i == 0 and j == 0 else False)
            else:
                loss += self.loss_func(this_frames,
                                       this_actions,
                                       this_actions_pre_softmax,
                                       this_rewards,
                                       this_returns_extrinsic,
                                       this_returns_intrinsic,
                                       this_advantages,
                                       True if i == 0 else False)
            self.num_records_processed += this_actions.shape[0]
            self.num_rollouts_processed += 1
            loss_for_reporting = loss / self.num_records_processed
            if self.num_rollouts_processed < self.minibatches:
                self.previous_loss = loss
            else:
                backprop(loss)
                loss = 0

        if self.iteration == (self.target_iterations - 1) and type(loss) != int:
            backprop(loss)

        chainer.report({'loss': loss_for_reporting})


def get_returns(rewards, gamma):
    returns = np.zeros_like(rewards).astype(np.float64)
    for i in range(len(rewards)):
        returns[i] = sum(
            (gamma ** i) * reward for i, reward in enumerate(rewards[i + 1:]))
    return returns


# https://arxiv.org/pdf/1506.02438.pdf
def get_advantages(rewards, values, gamma, lambda_gae):
    # print("Rewards:", rewards.flatten())
    # print("Values:", values.flatten())
    # print("Full length:", len(rewards))
    length = len(rewards) - 1
    # print("0-index length:", length)
    advantages = np.zeros_like(rewards).astype(np.float64)
    delta_ts = [0.] * (length + 1)
    for t in reversed(range(length + 1)):
        # print()
        if t == length:
            delta_ts[t] = 0.
            # print("t="+str(t)+": delta_ts["+str(t)+"] = 0.")
        else:
            delta_ts[t] = rewards[t] + (gamma * values[t + 1]) - values[t]
            # print("t="+str(t)+": delta_ts["+str(t)+"] = rewards["+str(t)+"="+str(rewards[t])+"]
            # + (gamma="+str(gamma)+" * values["+str(t)+"+1="+str(t+1)+"]="+str(values[t+1])+")
            # - values["+str(t)+"]="+str(values[t]))
        this_advantage = delta_ts[t]
        # print("this_advantage = delta_ts["+str(t)+"] = "+str(delta_ts[t]))
        advantage_lookahead = length - t
        # print("Looping 0 to min of ", length-t, advantage_lookahead)
        for i in range(min(length - t, advantage_lookahead)):
            this_advantage += ((gamma * lambda_gae) ** (i + 1)) * delta_ts[i + t + 1]
            # print("    i="+str(i)+": this_advantage +=
            # (("+str(gamma)+"*"+str(args.lambda_gae)+")**("+str(i)+"+1="+str(i+1)+"))
            # * delta_ts["+str(i)+"+"+str(t)+"+1="+str(i+t+1)+"]="+str(delta_ts[i+t+1]))
        advantages[t] = this_advantage
        # print("advantages[t="+str(t)+"] = this_advantage="+str(this_advantage))
    # print("Advantages raw:", advantages.flatten())
    # advantages = (advantages - advantages.mean()) / (advantages.std())
    # print("Advantages normalized:", advantages.flatten())
    return advantages


def rollout(rollout_args):
    iteration, trial, args, model_value, model_policy, model_rnd_target, model_rnd_predictor, rnd_parameters, \
    output_dir, epsilon_greedy = rollout_args
    try:
        np.random.seed()  # The same starting seed gets passed in multiprocessing, need to reset it for each process

        log(ID, ">> Starting iteration #" + str(iteration) + ", trial #" + str(trial))
        raw_frames = []
        stacked_frames = []
        stacked_frames_rnd = []
        actions = []
        actions_pre_softmax = []
        rewards_extrinsic = []
        rewards_intrinsic = []
        rewards_target = []
        values_extrinsic = []
        values_intrinsic = []
        start_time = time.time()

        model_value = model_value.copy()
        model_policy = model_policy.copy()
        model_rnd_target = model_rnd_target.copy()
        model_rnd_predictor = model_rnd_predictor.copy()
        if args.rnn_single_record_training:
            model_value.reset_state()
            model_policy.reset_state()
            model_rnd_target.reset_state()
            model_rnd_predictor.reset_state()

        env = gym.make(args.game)

        observation = np.array(Image.fromarray(env.reset()).resize((args.frame_resize, args.frame_resize)))
        # observation = np.random.randint(0, 256, observation.shape) #Just to test what happens with random inputs
        observation = np.dot(observation[..., :3], [0.299, 0.587, 0.114]).astype(np.float64)  # convert to grayscale
        raw_frames.append(observation)
        stacked_observations = np.zeros((1, args.stacked_frames, args.frame_resize, args.frame_resize)).astype(
            np.float64)
        stacked_observations[0, -1] = observation / 255.
        stacked_frames.append(stacked_observations)
        stacked_observations_rnd = np.zeros((1, args.stacked_frames_rnd, args.frame_resize, args.frame_resize)).astype(
            np.float64)
        if iteration == 0:
            stacked_observations_rnd[0, -1] = observation
        else:
            stacked_observations_rnd[0, -1] = np.clip((observation - rnd_parameters["observations_mean"]) / (
                rnd_parameters["observations_std"] + 1e-5), -args.rnd_obs_norm_clip, args.rnd_obs_norm_clip)
        stacked_frames_rnd.append(stacked_observations_rnd)

        if args.gpu >= 0:
            xp = cp
        else:
            xp = np

        cumulative_rewards_extrinsic = 0
        t = 0
        a_t_prev = None
        a_t_step_prev = None

        while True:
            if args.rnn_single_record_training:
                stacked_observations_ndarray = xp.asarray(stacked_observations)
                stacked_observations_ndarray_rnd = xp.asarray(stacked_observations_rnd)

                z_t_value = model_value.encode(stacked_observations_ndarray).data[0]
                z_t_policy = model_policy.encode(stacked_observations_ndarray).data[0]
                z_t_rnd_target = model_rnd_target.encode(stacked_observations_ndarray_rnd).data[0]
                z_t_rnd_predictor = model_rnd_predictor.encode(stacked_observations_ndarray_rnd).data[0]

                h_t_value = model_value.model(z_t_value)
                h_t_policy = model_policy.model(z_t_policy)
                h_t_rnd_target = model_rnd_target.model(z_t_rnd_target)
                h_t_rnd_predictor = model_rnd_predictor.model(z_t_rnd_predictor)

                value = model_value.final(h_t_value).data[0]
                if args.gpu >= 0:
                    value = cp.asnumpy(value)
                value_extrinsic = value[0]
                value_intrinsic = value[1]
                action_pre_softmax = model_policy.final(h_t_policy).data[0]
                if args.gpu >= 0:
                    action_pre_softmax = cp.asnumpy(action_pre_softmax)
                reward_target = model_rnd_target.final(h_t_rnd_target).data[0]
                if args.gpu >= 0:
                    reward_target = cp.asnumpy(reward_target)
                reward_predictor = model_rnd_predictor.final(h_t_rnd_predictor).data[0]
                if args.gpu >= 0:
                    reward_predictor = cp.asnumpy(reward_predictor)
            else:
                model_value.reset_state()
                model_policy.reset_state()
                model_rnd_target.reset_state()
                model_rnd_predictor.reset_state()

                stacked_observations_ndarray = xp.array(stacked_frames)
                stacked_observations_ndarray = stacked_observations_ndarray.reshape(
                    stacked_observations_ndarray.shape[0], stacked_observations_ndarray.shape[2],
                    stacked_observations_ndarray.shape[3], stacked_observations_ndarray.shape[4])
                stacked_observations_ndarray_rnd = xp.array(stacked_frames_rnd)
                stacked_observations_ndarray_rnd = stacked_observations_ndarray_rnd.reshape(
                    stacked_observations_ndarray_rnd.shape[0], stacked_observations_ndarray_rnd.shape[2],
                    stacked_observations_ndarray_rnd.shape[3], stacked_observations_ndarray_rnd.shape[4])

                for i in range(math.ceil(stacked_observations_ndarray.shape[0] / args.sequence_length)):
                    start_idx = i * args.sequence_length
                    end_idx = (i + 1) * args.sequence_length
                    this_observations_ndarray = stacked_observations_ndarray[start_idx:end_idx]
                    this_observations_rnd_ndarray = stacked_observations_ndarray_rnd[start_idx:end_idx]

                    z_t_value = model_value.encode(this_observations_ndarray)
                    z_t_policy = model_policy.encode(this_observations_ndarray)
                    z_t_rnd_target = model_rnd_target.encode(this_observations_rnd_ndarray)
                    z_t_rnd_predictor = model_rnd_predictor.encode(this_observations_rnd_ndarray)

                    h_t_value = model_value.model(z_t_value)
                    h_t_policy = model_policy.model(z_t_policy)
                    h_t_rnd_target = model_rnd_target.model(z_t_rnd_target)
                    h_t_rnd_predictor = model_rnd_predictor.model(z_t_rnd_predictor)

                    value = model_value.final(h_t_value).data[-1]
                    if args.gpu >= 0:
                        value = cp.asnumpy(value)
                    value_extrinsic = value[0]
                    value_intrinsic = value[1]
                    action_pre_softmax = model_policy.final(h_t_policy).data[-1]
                    if args.gpu >= 0:
                        action_pre_softmax = cp.asnumpy(action_pre_softmax)
                    reward_target = model_rnd_target.final(h_t_rnd_target).data[-1]
                    if args.gpu >= 0:
                        reward_target = cp.asnumpy(reward_target)
                    reward_predictor = model_rnd_predictor.final(h_t_rnd_predictor).data[-1]
                    if args.gpu >= 0:
                        reward_predictor = cp.asnumpy(reward_predictor)

            actions_pre_softmax.append(action_pre_softmax)
            action_probabilities = F.softmax(F.expand_dims(action_pre_softmax, 0)).data[0]

            reward_intrinsic = (reward_target - reward_predictor) ** 2

            if np.random.uniform(0, 1) < args.sticky_action_probability and a_t_prev is not None:
                a_t = a_t_prev
                a_t_step = a_t_step_prev
            else:
                if np.random.random() > epsilon_greedy:
                    discrete_a_t = np.random.choice(range(args.action_dim), p=action_probabilities)
                else:
                    discrete_a_t = np.random.randint(0, args.action_dim)
                a_t = np.array([0.] * args.action_dim).astype(np.float64)
                a_t[discrete_a_t] = 1.
                a_t_step = ACTIONS[args.game][discrete_a_t]
            a_t_prev = a_t
            a_t_step_prev = a_t_step
            try:
                # print("[ ", end="")
                # for ap in action_probabilities:
                #     print("{:.2f} ".format(ap), end="")
                # print("]", a_t, a_t_step, t)
                observation, reward_extrinsic, done, _ = env.step(a_t_step)
                # observation = np.random.randint(0, 256, observation.shape)
            except:
                break

            observation = np.array(Image.fromarray(observation).resize((args.frame_resize, args.frame_resize)))
            observation = np.dot(observation[..., :3], [0.299, 0.587, 0.114]).astype(np.float64)  # convert to grayscale
            raw_frames.append(observation)
            stacked_observations = np.concatenate(
                (stacked_observations, np.expand_dims(np.expand_dims(observation / 255., 0), 0)), axis=1)
            stacked_observations = stacked_observations[:, 1:1 + args.stacked_frames]
            stacked_frames.append(stacked_observations)
            if iteration == 0:
                obs_normalized_rnd = observation
            else:
                obs_normalized_rnd = np.clip((observation - rnd_parameters["observations_mean"]) / (
                    rnd_parameters["observations_std"] + 1e-5), -args.rnd_obs_norm_clip, args.rnd_obs_norm_clip)
            stacked_observations_rnd = np.concatenate(
                (stacked_observations_rnd, np.expand_dims(np.expand_dims(obs_normalized_rnd, 0), 0)), axis=1)
            stacked_observations_rnd = stacked_observations_rnd[:, 1:1 + args.stacked_frames_rnd]
            stacked_frames_rnd.append(stacked_observations_rnd)

            cumulative_rewards_extrinsic += reward_extrinsic

            reward_extrinsic = np.array([reward_extrinsic]).astype(np.float64)

            actions.append(a_t)
            rewards_extrinsic.append(reward_extrinsic)
            rewards_intrinsic.append(reward_intrinsic)
            rewards_target.append(reward_target)
            values_extrinsic.append(value_extrinsic)
            values_intrinsic.append(value_intrinsic)

            t += 1

            if done:
                break

        env.close()

        raw_frames = np.asarray(raw_frames)
        stacked_frames = np.asarray(stacked_frames)
        stacked_frames = stacked_frames.reshape(stacked_frames.shape[0], stacked_frames.shape[2],
                                                stacked_frames.shape[3], stacked_frames.shape[4])
        stacked_frames_rnd = np.asarray(stacked_frames_rnd)
        stacked_frames_rnd = stacked_frames_rnd.reshape(stacked_frames_rnd.shape[0], stacked_frames_rnd.shape[2],
                                                        stacked_frames_rnd.shape[3], stacked_frames_rnd.shape[4])
        actions = np.asarray(actions)
        actions_pre_softmax = np.asarray(actions_pre_softmax).astype(np.float64)
        rewards_extrinsic = np.asarray(rewards_extrinsic).astype(np.float64)
        rewards_extrinsic = args.extrinsic_coefficient * np.clip(rewards_extrinsic, -args.extrinsic_reward_clip,
                                                                 args.extrinsic_reward_clip)
        rewards_intrinsic = np.asarray(rewards_intrinsic)
        rewards_target = np.asarray(rewards_target)
        values_extrinsic = np.asarray(values_extrinsic).astype(np.float64)
        values_intrinsic = np.asarray(values_intrinsic).astype(np.float64)

        this_output_dir = os.path.join(output_dir, "rollouts", "{}".format(time.time()))
        os.makedirs(this_output_dir, exist_ok=True)

        imageio.mimsave(os.path.join(this_output_dir, 'rollout.gif'), raw_frames.astype(np.uint8), fps=20)

        with gzip.GzipFile(os.path.join(this_output_dir, "raw_frames.npy.gz"), "w") as file:
            np.save(file, raw_frames)
        with gzip.GzipFile(os.path.join(this_output_dir, "stacked_frames.npy.gz"), "w") as file:
            np.save(file, stacked_frames)
        with gzip.GzipFile(os.path.join(this_output_dir, "stacked_frames_rnd.npy.gz"), "w") as file:
            np.save(file, stacked_frames_rnd)
        count = raw_frames.shape[0]
        with open(os.path.join(this_output_dir, "count"), "w") as file:
            print("{}".format(count), file=file)
        with open(os.path.join(this_output_dir, "cumulative_rewards_extrinsic"), "w") as file:
            print("{}".format(cumulative_rewards_extrinsic), file=file)

        np.savez_compressed(os.path.join(this_output_dir, "misc.npz"),
                            actions=actions,
                            actions_pre_softmax=actions_pre_softmax,
                            rewards_extrinsic=rewards_extrinsic,
                            rewards_intrinsic=rewards_intrinsic,
                            rewards_target=rewards_target,
                            values_extrinsic=values_extrinsic,
                            values_intrinsic=values_intrinsic)

        log(ID,
            ">> Finished iteration #{}, trial #{} in {} timesteps in {:.2f}s with extrinsic cumulative reward {:.2f}"
            .format(iteration, trial, t, (time.time() - start_time),
                    cumulative_rewards_extrinsic))

        returns_intrinsic = get_returns(rewards_intrinsic, args.gamma_intrinsic)
        return this_output_dir, cumulative_rewards_extrinsic, raw_frames.mean(axis=0), raw_frames.std(axis=0), \
               returns_intrinsic.mean(), returns_intrinsic.std(), t

    except Exception:
        print(traceback.format_exc())
        return 0.


def compute_advatages(worker_args):
    args, this_output_dir, rnd_parameters = worker_args
    npz = np.load(os.path.join(this_output_dir, "misc.npz"))
    actions = npz["actions"]
    actions_pre_softmax = npz["actions_pre_softmax"]
    rewards_extrinsic = npz["rewards_extrinsic"]
    rewards_intrinsic = npz["rewards_intrinsic"]
    rewards_target = npz["rewards_target"]
    values_extrinsic = npz["values_extrinsic"]
    values_intrinsic = npz["values_intrinsic"]
    npz.close()

    rewards_intrinsic = args.intrinsic_coefficient * rewards_intrinsic / rnd_parameters["returns_intrinsic_std"]

    returns_extrinsic = get_returns(rewards_extrinsic, args.gamma_extrinsic)
    returns_intrinsic = get_returns(rewards_intrinsic, args.gamma_intrinsic)
    advantages_extrinsic = get_advantages(rewards_extrinsic, values_extrinsic, args.gamma_extrinsic, args.lambda_gae)
    advantages_intrinsic = get_advantages(rewards_intrinsic, values_intrinsic, args.gamma_intrinsic, args.lambda_gae)
    advantages_combined = advantages_extrinsic + advantages_intrinsic

    np.savez_compressed(os.path.join(this_output_dir, "misc.npz"),
                        actions=actions,
                        actions_pre_softmax=actions_pre_softmax,
                        rewards_extrinsic=rewards_extrinsic,
                        rewards_intrinsic=rewards_intrinsic,
                        rewards_target=rewards_target,
                        returns_extrinsic=returns_extrinsic,
                        returns_intrinsic=returns_intrinsic,
                        advantages_combined=advantages_combined,
                        values_extrinsic=values_extrinsic,
                        values_intrinsic=values_intrinsic)


def main():
    parser = argparse.ArgumentParser(description=ID)
    parser.add_argument('--data_dir', '-d', default="/data/" + ID, help='The base data/output directory')
    parser.add_argument('--game', default='PixelCopter-v0', help='Game to use')
    parser.add_argument('--experiment_name', default='experiment_1', help='To isolate its files from others')
    parser.add_argument('--model', action='store_true',
                        help='Resume using .model files that are saved when training completes')
    parser.add_argument('--no_resume', action='store_true', help='Don''t auto resume from the latest snapshot')
    parser.add_argument('--z_dim', '-z', default=32, type=int, help='Dimension of encoded vector')
    parser.add_argument('--rnn_hidden_dim', default=256, type=int, help='RNN hidden units')
    parser.add_argument('--rnn_hidden_layers', default=1, type=int, help='RNN hidden layers')
    parser.add_argument('--final_hidden_dim', default=0, type=int,
                        help='Units for additional linear layers before final output')
    parser.add_argument('--final_hidden_layers', default=0, type=int,
                        help='Additional linear layers before final output')
    parser.add_argument('--num_trials', default=128, type=int,
                        help='Trials per iteration of training. Referred to as "Rollout Length" in the paper')
    parser.add_argument('--target_cumulative_rewards_extrinsic', default=100, type=int,
                        help='Target cumulative extrinsic reward over all trials in an interation. '
                             'Training ends when this is achieved')
    parser.add_argument('--frame_resize', default=84, type=int, help='h x w resize of each observation frame')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--num_threads', default=10, type=int,
                        help='# threads for running rollouts in parallel. Best to use 1 only for GPU')
    parser.add_argument('--dnc', default="", help="Differentiable Neural Computer. N,W,R,K, e.g. 256,64,4,0")
    parser.add_argument('--sequence_length', type=int, default=64,
                        help='This amount of input records are stacked together for a forward pass')
    parser.add_argument('--minibatches', type=int, default=4, help='Backprop performed over this many rollouts')
    parser.add_argument('--gamma_extrinsic', default=0.999, type=float, help='Discount factor for extrinsic rewards')
    parser.add_argument('--gamma_intrinsic', default=0.99, type=float, help='Discount factor for intrinsic rewards')
    parser.add_argument('--lambda_gae', default=0.95, type=float, help='GAE parameter')
    parser.add_argument('--gradient_clip', default=0.0, type=float,
                        help='Gradients clipped scaled to this L2 norm threshold')
    parser.add_argument('--epochs_per_iteration', type=int, default=4, help='Number of optimization epochs')
    parser.add_argument('--epsilon_greedy', default=0.0, type=float, help='epsilon-greedy for exploration')
    parser.add_argument('--rng_seed', default=31337, type=int, help='')
    parser.add_argument('--rnn_single_record_training', action='store_true', help="Required for DNC")
    parser.add_argument('--beta_entropy', type=float, default=0.001, help="Entropy coeficient")
    parser.add_argument('--epsilon_ppo', type=float, default=0.1, help="PPO loss clip range, to +/- of this value")
    parser.add_argument('--model_value_lr', type=float, default=0.0001, help="Learning rate for value network")
    parser.add_argument('--model_policy_lr', type=float, default=0.0001, help="Learning rate for policy network")
    parser.add_argument('--model_rnd_predictor_lr', type=float, default=0.0001,
                        help="Learning rate for RND predictor network")
    parser.add_argument('--disable_progress_bar', action='store_true',
                        help="Disable Chainer's progress bar when optimizing")
    parser.add_argument('--extrinsic_coefficient', type=float, default=2.0, help="As in the RND paper")
    parser.add_argument('--intrinsic_coefficient', type=float, default=1.0, help="As in the RND paper")
    parser.add_argument('--initial_normalization_num_trials', type=int, default=8,
                        help="Collect observations over this many trials for initialization of RND normalization "
                             "pramaters")
    parser.add_argument('--portion_experience_train_rnd_predictor', type=float, default=0.25,
                        help="As in the RND paper")
    parser.add_argument('--stacked_frames', type=int, default=4,
                        help="# of observations stacked together for the value/policy")
    parser.add_argument('--stacked_frames_rnd', type=int, default=1,
                        help="# of observations stacked together for the RND predictor network")
    parser.add_argument('--extrinsic_reward_clip', type=float, default=1.0,
                        help="Will be clipped from - to +, as in the RND paper")
    parser.add_argument('--rnd_obs_norm_clip', type=float, default=5.0,
                        help="Will be clipped from - to +, as in the RND paper")
    parser.add_argument('--sticky_action_probability', type=float, default=0.25,
                        help="Repeat the previous action with this probability")
    parser.add_argument('--keep_past_x_snapshots', type=int, default=10,
                        help="Delete snapshots older than this many iterations to free up disk space")
    args = parser.parse_args()
    log(ID, "args =\n " + str(vars(args)).replace(",", ",\n "))

    if args.game not in ACTIONS:
        log(ID, "Error: No actions configured for " + args.game + ". Please add them to the ACTIONS constant.")
        exit()

    np.random.seed(args.rng_seed)
    random.seed(args.rng_seed)

    output_dir = os.path.join(args.data_dir, args.game, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    action_dim = len(ACTIONS[args.game])
    args.action_dim = action_dim

    auto_resume_file = None
    iteration = 0
    if not args.model and not args.no_resume:
        files = os.listdir(output_dir)
        for file in files:
            if re.match(r'^snapshot_iter_', file):
                this_iteration = int(re.search(r'\d+', file).group())
                if this_iteration > iteration:
                    iteration = this_iteration
        if iteration > 0:
            auto_resume_file = os.path.join(output_dir, "snapshot_iter_{}".format(iteration))

    trainer_value_file = None
    trainer_policy_file = None
    trainer_rnd_predictor_file = None
    model_value_file = None
    model_policy_file = None
    model_rnd_target_file = None
    model_rnd_predictor_file = None
    rnd_parameters_file = None
    if args.model:
        model_value_file = os.path.join(output_dir, "model_value.model")
        model_policy_file = os.path.join(output_dir, "model_policy.model")
        model_rnd_target_file = os.path.join(output_dir, "model_rnd_target.model")
        model_rnd_predictor_file = os.path.join(output_dir, "model_rnd_predictor.model")
        rnd_parameters_file = os.path.join(output_dir, "rnd_parameters.npz")
        log(ID, "Loading saved models from disk")
    elif auto_resume_file:
        log(ID, "Auto resuming from last snapshot: " + auto_resume_file)
        trainer_value_file = auto_resume_file + "_value.trainer"
        trainer_policy_file = auto_resume_file + "_policy.trainer"
        trainer_rnd_predictor_file = auto_resume_file + "_rnd_predictor.trainer"
        model_value_file = auto_resume_file + "_value.model"
        model_policy_file = auto_resume_file + "_policy.model"
        model_rnd_target_file = os.path.join(output_dir, "model_rnd_target.model")
        model_rnd_predictor_file = auto_resume_file + "_rnd_predictor.model"
        rnd_parameters_file = auto_resume_file + "_rnd_parameters.npz"

    model_value = Model(z_dim=args.z_dim,
                        rnn_hidden_dim=args.rnn_hidden_dim,
                        rnn_hidden_layers=args.rnn_hidden_layers,
                        final_hidden_dim=args.final_hidden_dim,
                        final_hidden_layers=args.final_hidden_layers,
                        final_dim=2,
                        dnc=args.dnc,
                        epsilon_ppo=args.epsilon_ppo,
                        beta_entropy=args.beta_entropy,
                        is_value_net=True,
                        is_rnd_net=False)
    model_policy = Model(z_dim=args.z_dim,
                         rnn_hidden_dim=args.rnn_hidden_dim,
                         rnn_hidden_layers=args.rnn_hidden_layers,
                         final_hidden_dim=args.final_hidden_dim,
                         final_hidden_layers=args.final_hidden_layers,
                         final_dim=args.action_dim,
                         dnc=args.dnc,
                         epsilon_ppo=args.epsilon_ppo,
                         beta_entropy=args.beta_entropy,
                         is_value_net=False,
                         is_rnd_net=False)
    model_rnd_target = Model(z_dim=args.z_dim,
                             rnn_hidden_dim=args.rnn_hidden_dim,
                             rnn_hidden_layers=args.rnn_hidden_layers,
                             final_hidden_dim=args.final_hidden_dim,
                             final_hidden_layers=args.final_hidden_layers,
                             final_dim=1,
                             dnc=args.dnc,
                             epsilon_ppo=args.epsilon_ppo,
                             beta_entropy=args.beta_entropy,
                             is_value_net=False,
                             is_rnd_net=True)
    model_rnd_predictor = Model(z_dim=args.z_dim,
                                rnn_hidden_dim=args.rnn_hidden_dim,
                                rnn_hidden_layers=args.rnn_hidden_layers,
                                final_hidden_dim=args.final_hidden_dim,
                                final_hidden_layers=args.final_hidden_layers,
                                final_dim=1,
                                dnc=args.dnc,
                                epsilon_ppo=args.epsilon_ppo,
                                beta_entropy=args.beta_entropy,
                                is_value_net=False,
                                is_rnd_net=True)
    rnd_parameters = {}

    if model_value_file:
        chainer.serializers.load_npz(model_value_file, model_value)
        chainer.serializers.load_npz(model_policy_file, model_policy)
        chainer.serializers.load_npz(model_rnd_target_file, model_rnd_target)
        chainer.serializers.load_npz(model_rnd_predictor_file, model_rnd_predictor)
        rnd_parameters_npz = np.load(rnd_parameters_file)
        rnd_parameters["count"] = rnd_parameters_npz["count"]
        rnd_parameters["returns_intrinsic_mean"] = rnd_parameters_npz["returns_intrinsic_mean"]
        rnd_parameters["returns_intrinsic_std"] = rnd_parameters_npz["returns_intrinsic_std"]
        rnd_parameters["observations_mean"] = rnd_parameters_npz["observations_mean"]
        rnd_parameters["observations_std"] = rnd_parameters_npz["observations_std"]
        rnd_parameters_npz.close()
    else:
        model_rnd_target_file = os.path.join(output_dir, "model_rnd_target.model")
        log(ID, "> Saving model_rnd_target to: " + model_rnd_target_file)
        chainer.serializers.save_npz(model_rnd_target_file, model_rnd_target)
        rnd_parameters["count"] = 0
        rnd_parameters["returns_intrinsic_mean"] = None
        rnd_parameters["returns_intrinsic_std"] = None
        rnd_parameters["observations_mean"] = None
        rnd_parameters["observations_std"] = None

    optimizer_value = chainer.optimizers.Adam(alpha=args.model_value_lr)
    optimizer_value.setup(model_value)
    optimizer_policy = chainer.optimizers.Adam(alpha=args.model_policy_lr)
    optimizer_policy.setup(model_policy)
    optimizer_rnd_predictor = chainer.optimizers.Adam(alpha=args.model_rnd_predictor_lr)
    optimizer_rnd_predictor.setup(model_rnd_predictor)

    if args.gradient_clip > 0.:
        optimizer_value.add_hook(chainer.optimizer_hooks.GradientClipping(args.gradient_clip))
        optimizer_policy.add_hook(chainer.optimizer_hooks.GradientClipping(args.gradient_clip))
        optimizer_rnd_predictor.add_hook(chainer.optimizer_hooks.GradientClipping(args.gradient_clip))
    rollouts_dir = os.path.join(output_dir, "rollouts")
    data = Dataset(rollouts_dir=rollouts_dir, is_rnd=False)
    data_rnd = Dataset(rollouts_dir=rollouts_dir, is_rnd=True)
    train_iter_value = chainer.iterators.SerialIterator(data, 1, shuffle=False)
    train_iter_policy = chainer.iterators.SerialIterator(data, 1, shuffle=False)
    train_iter_rnd_predictor = chainer.iterators.SerialIterator(data_rnd, 1, shuffle=False)
    updater_value = TBPTTUpdater(train_iter_value, args.gpu, optimizer_value, model_value.get_loss_func(),
                                 args.sequence_length, args.minibatches, args.rnn_single_record_training)
    updater_policy = TBPTTUpdater(train_iter_policy, args.gpu, optimizer_policy, model_policy.get_loss_func(),
                                  args.sequence_length, args.minibatches, args.rnn_single_record_training)
    updater_rnd_predictor = TBPTTUpdater(train_iter_rnd_predictor, args.gpu, optimizer_rnd_predictor,
                                         model_rnd_predictor.get_loss_func(), args.sequence_length, args.minibatches,
                                         args.rnn_single_record_training)

    if args.gpu >= 0:
        model_value.to_gpu(args.gpu)
        model_policy.to_gpu(args.gpu)
        model_rnd_target.to_gpu(args.gpu)
        model_rnd_predictor.to_gpu(args.gpu)

    log(ID, "Starting")

    ###***###***###*** MAIN ITERATION LOOP ***###***###***###
    while True:
        iteration += 1

        if iteration > args.keep_past_x_snapshots:
            delete_snapshots = iteration - args.keep_past_x_snapshots
            files = os.listdir(output_dir)
            for file in files:
                if re.match(r'^snapshot_iter_', file):
                    this_iteration = int(re.search(r'\d+', file).group())
                    if this_iteration < delete_snapshots:
                        delete_file = os.path.join(output_dir, file)
                        log(ID, "Deleting old snapshot file: " + delete_file)
                        os.remove(delete_file)

        ##***###***###*** DATA COLLECTION ***###***###***###
        if iteration == 1:
            worker_arg_tuples = []
            for trial in range(1, args.initial_normalization_num_trials + 1):
                worker_arg_tuples.append((0, trial, args, model_value, model_policy, model_rnd_target,
                                          model_rnd_predictor, rnd_parameters, output_dir, 1.))
            pool = Pool(args.num_threads)
            log(ID, "> Performing " + str(args.initial_normalization_num_trials)
                + " rollouts at iteration 0, epsilon_greedy: {:.2f}, for observation normalization parameters"
                .format(1.))
            outputs = pool.map(rollout, worker_arg_tuples)
            pool.close()
            pool.join()
            total_observations_mean = None
            total_observations_std = None
            total_timesteps = 0
            #http://blog.adeel.io/2019/02/05/calculating-running-estimate-of-mean-and-standard-deviation-in-python/
            for output in outputs:
                rollout_dir, cumulative_rewards_extrinsic, observations_mean, observations_std, \
                returns_intrinsic_mean, returns_intrinsic_std, timesteps = output
                if total_observations_mean is None and total_observations_std is None:
                    total_observations_mean = observations_mean
                    total_observations_std = observations_std
                else:
                    new_observations_mean = (total_observations_mean * total_timesteps + observations_mean *
                                             timesteps) / (total_timesteps + timesteps)
                    new_observations_std = \
                        np.sqrt(((total_timesteps - 1) * (total_observations_std ** 2) + (timesteps - 1)
                                 * (observations_std ** 2) + total_timesteps *
                                 ((new_observations_mean - total_observations_mean) ** 2) + timesteps *
                                 ((new_observations_mean - observations_mean) ** 2)) /
                                (total_timesteps + timesteps - 1))
                    total_observations_mean = new_observations_mean
                    total_observations_std = new_observations_std
                total_timesteps += timesteps
            rnd_parameters["observations_mean"] = total_observations_mean
            rnd_parameters["observations_std"] = total_observations_std
            rnd_parameters["count"] = total_timesteps
            try:
                rmtree(os.path.join(output_dir, "rollouts"))
            except Exception as e:
                None
            log(ID, "> Collected initial observation normalization parameters")

        try:
            rmtree(os.path.join(output_dir, "rollouts"))
        except Exception as e:
            None
        worker_arg_tuples = []
        for trial in range(1, args.num_trials + 1):
            worker_arg_tuples.append((iteration, trial, args, model_value, model_policy, model_rnd_target,
                                      model_rnd_predictor, rnd_parameters, output_dir, args.epsilon_greedy))
        pool = Pool(args.num_threads)
        log(ID, "> Starting collecting " + str(args.num_trials) + " rollouts for iteration " + str(
            iteration) + ", epsilon_greedy: {:.2f}".format(args.epsilon_greedy))
        outputs = pool.map(rollout, worker_arg_tuples)
        pool.close()
        pool.join()
        new_rollout_dirs = []
        total_cumulative_rewards_extrinsic = []
        total_observations_mean = rnd_parameters["observations_mean"]
        total_observations_std = rnd_parameters["observations_std"]
        total_returns_intrinsic_mean = rnd_parameters["returns_intrinsic_mean"]
        total_returns_intrinsic_std = rnd_parameters["returns_intrinsic_std"]
        total_timesteps = rnd_parameters["count"]
        for output in outputs:
            rollout_dir, cumulative_rewards_extrinsic, observations_mean, observations_std, returns_intrinsic_mean,\
            returns_intrinsic_std, timesteps = output
            new_rollout_dirs.append(rollout_dir)
            total_cumulative_rewards_extrinsic.append(cumulative_rewards_extrinsic)
            new_observations_mean = \
                (total_observations_mean * total_timesteps + observations_mean * timesteps) / \
                (total_timesteps + timesteps)
            new_observations_std = \
                np.sqrt(((total_timesteps - 1) * (total_observations_std ** 2) + (timesteps - 1)
                    * (observations_std ** 2) + total_timesteps *
                         ((new_observations_mean - total_observations_mean) ** 2) + timesteps
                         * ((new_observations_mean - observations_mean) ** 2))
                        / (total_timesteps + timesteps - 1))
            total_observations_mean = new_observations_mean
            total_observations_std = new_observations_std
            if total_returns_intrinsic_mean is None and total_returns_intrinsic_std is None:
                total_returns_intrinsic_mean = returns_intrinsic_mean
                total_returns_intrinsic_std = returns_intrinsic_std
            else:
                new_returns_intrinsic_mean = \
                    (total_returns_intrinsic_mean * total_timesteps + returns_intrinsic_mean * timesteps) / \
                    (total_timesteps + timesteps)
                new_returns_intrinsic_std = \
                    np.sqrt(((total_timesteps - 1) * (total_returns_intrinsic_std ** 2) + (timesteps - 1) *
                             (returns_intrinsic_std ** 2) + total_timesteps * ((new_returns_intrinsic_mean
                                                                                - total_returns_intrinsic_mean) ** 2)
                             + timesteps * ((new_returns_intrinsic_mean - returns_intrinsic_mean) ** 2)) /
                            (total_timesteps + timesteps - 1))
                total_returns_intrinsic_mean = new_returns_intrinsic_mean
                total_returns_intrinsic_std = new_returns_intrinsic_std
            total_timesteps += timesteps
        rnd_parameters["observations_mean"] = total_observations_mean
        rnd_parameters["observations_std"] = total_observations_std
        rnd_parameters["returns_intrinsic_mean"] = total_returns_intrinsic_mean
        rnd_parameters["returns_intrinsic_std"] = total_returns_intrinsic_std
        rnd_parameters["count"] = total_timesteps
        avg_cumulative_rewards_extrinsic = np.mean(total_cumulative_rewards_extrinsic)
        log(ID, "> Done collecting " + str(args.num_trials) + " rollouts for iteration " + str(
            iteration) + ", epsilon_greedy: {:.2f}, avg cum rwd: {:.2f}, std: {:.2f}, min: {:.2f}, max: {:.2f}"
            .format(args.epsilon_greedy, avg_cumulative_rewards_extrinsic,
                    np.std(total_cumulative_rewards_extrinsic), np.min(total_cumulative_rewards_extrinsic),
                    np.max(total_cumulative_rewards_extrinsic)))
        if avg_cumulative_rewards_extrinsic >= args.target_cumulative_rewards_extrinsic:
            log(ID, "Target cumulative reward reached!")
            model_value_file = os.path.join(output_dir, "model_value.model")
            model_policy_file = os.path.join(output_dir, "model_policy.model")
            log(ID, "Saving final models to: " + model_value_file + ", " + model_policy_file)
            chainer.serializers.save_npz(model_value_file, model_value)
            chainer.serializers.save_npz(model_policy_file, model_policy)
            break
        worker_arg_tuples = []
        for rollout_dir in new_rollout_dirs:
            worker_arg_tuples.append((args, rollout_dir, rnd_parameters))
        pool = Pool(args.num_threads)
        log(ID, "> Computing cumulative rewards and advantages for collected rollouts")
        pool.map(compute_advatages, worker_arg_tuples)
        pool.close()
        pool.join()

        ###***###***###*** MODEL TRAIN ***###***###***###
        data.reset()
        data_rnd.reset()
        data_rnd.num_rollouts = int(args.num_trials * args.portion_experience_train_rnd_predictor)

        updater_value.target_iterations = iteration * args.epochs_per_iteration * len(data)
        trainer_value = Trainer(updater_value, (iteration * args.epochs_per_iteration, 'epoch'), out=output_dir)
        trainer_value.extend(extensions.LogReport(trigger=(1 if args.gpu < 0 else 50, 'iteration')))
        trainer_value.extend(extensions.PrintReport(['epoch', 'iteration', 'loss', 'elapsed_time']))
        if not args.disable_progress_bar:
            trainer_value.extend(extensions.ProgressBar(update_interval=1 if args.gpu < 0 else 10))
        if not args.model and trainer_value_file is not None:
            chainer.serializers.load_npz(trainer_value_file, trainer_value)
        log(ID, "Running trainer_value at iteration " + str(iteration))
        trainer_value.run()
        trainer_value_file = os.path.join(output_dir, "snapshot_iter_{}_value.trainer".format(iteration))
        log(ID, "> Saving trainer_value snapshot to: " + trainer_value_file)
        chainer.serializers.save_npz(trainer_value_file, trainer_value)

        updater_policy.target_iterations = iteration * args.epochs_per_iteration * len(data)
        trainer_policy = Trainer(updater_policy, (iteration * args.epochs_per_iteration, 'epoch'), out=output_dir)
        trainer_policy.extend(extensions.LogReport(trigger=(1 if args.gpu < 0 else 50, 'iteration')))
        trainer_policy.extend(extensions.PrintReport(['epoch', 'iteration', 'loss', 'elapsed_time']))
        if not args.disable_progress_bar:
            trainer_policy.extend(extensions.ProgressBar(update_interval=1 if args.gpu < 0 else 10))
        if not args.model and trainer_policy_file is not None:
            chainer.serializers.load_npz(trainer_policy_file, trainer_policy)
        log(ID, "Running trainer_policy at iteration " + str(iteration))
        trainer_policy.run()
        trainer_policy_file = os.path.join(output_dir, "snapshot_iter_{}_policy.trainer".format(iteration))
        log(ID, "> Saving trainer_policy snapshot to: " + trainer_policy_file)
        chainer.serializers.save_npz(trainer_policy_file, trainer_policy)

        updater_rnd_predictor.target_iterations = iteration * args.epochs_per_iteration * len(data_rnd)
        trainer_rnd_predictor = Trainer(updater_rnd_predictor, (iteration * args.epochs_per_iteration, 'epoch'),
                                        out=output_dir)
        trainer_rnd_predictor.extend(extensions.LogReport(trigger=(1 if args.gpu < 0 else 50, 'iteration')))
        trainer_rnd_predictor.extend(extensions.PrintReport(['epoch', 'iteration', 'loss', 'elapsed_time']))
        if not args.disable_progress_bar:
            trainer_rnd_predictor.extend(extensions.ProgressBar(update_interval=1 if args.gpu < 0 else 10))
        if not args.model and trainer_rnd_predictor_file is not None:
            chainer.serializers.load_npz(trainer_rnd_predictor_file, trainer_rnd_predictor)
        log(ID, "Running trainer_rnd_predictor at iteration " + str(iteration))
        trainer_rnd_predictor.run()
        trainer_rnd_predictor_file = os.path.join(output_dir,
                                                  "snapshot_iter_{}_rnd_predictor.trainer".format(iteration))
        log(ID, "> Saving trainer_rnd_predictor snapshot to: " + trainer_rnd_predictor_file)
        chainer.serializers.save_npz(trainer_rnd_predictor_file, trainer_rnd_predictor)

        model_value_file = os.path.join(output_dir, "snapshot_iter_{}_value.model".format(iteration))
        log(ID, "> Saving model_value snapshot to: " + model_value_file)
        chainer.serializers.save_npz(model_value_file, model_value)
        model_policy_file = os.path.join(output_dir, "snapshot_iter_{}_policy.model".format(iteration))
        log(ID, "> Saving model_policy snapshot to: " + model_policy_file)
        chainer.serializers.save_npz(model_policy_file, model_policy)
        model_rnd_predictor_file = os.path.join(output_dir,
                                                "snapshot_iter_{}_rnd_predictor.model".format(iteration))
        log(ID, "> Saving model_rnd_predictor snapshot to: " + model_rnd_predictor_file)
        chainer.serializers.save_npz(model_rnd_predictor_file, model_rnd_predictor)
        rnd_parameters_file = os.path.join(output_dir, "snapshot_iter_{}_rnd_parameters.npz".format(iteration))
        log(ID, "> Saving rnd_parameters snapshot to: " + rnd_parameters_file)
        np.savez_compressed(rnd_parameters_file,
                            count=rnd_parameters["count"],
                            returns_intrinsic_mean=rnd_parameters["returns_intrinsic_mean"],
                            returns_intrinsic_std=rnd_parameters["returns_intrinsic_std"],
                            observations_mean=rnd_parameters["observations_mean"],
                            observations_std=rnd_parameters["observations_std"])

    log(ID, "Done")


if __name__ == '__main__':
    main()
