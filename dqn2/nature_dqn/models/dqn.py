import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
from nature_dqn.configs.config import config
from nature_dqn.models.replay_buffer import ReplayBuffer


class DQN(object):
    def __init__(self, env):
        self.env = env
        self.build()

    def make_placeholder(self):
        state_shape = list(self.env.observation_space.shape)
        # img_height, img_width, nchannels = state_shape[1], state_shape[2], state_shape[0]
        img_height, img_width, nchannels = state_shape[0], state_shape[1], state_shape[2]
        self.s = tf.placeholder(tf.uint8, [None, img_height, img_width, state_shape[2]*config.state_history])
        self.a = tf.placeholder(tf.int32, [None])
        self.r = tf.placeholder(tf.float32, [None])
        self.sp = tf.placeholder(tf.uint8, [None, img_height, img_width, state_shape[2]*config.state_history])
        self.done_mask = tf.placeholder(tf.bool, [None])
        self.lr = tf.placeholder(tf.float32)
        # print(self.sp.shape)

    def calculate_q_value(self, state, scope, reuse):
        num_actions = self.env.action_space.n
        with tf.variable_scope(scope, reuse=reuse):
            l1 = layers.conv2d(state, num_outputs=32, kernel_size=8, stride=4, padding='VALID', activation_fn=tf.nn.relu)
            l2 = layers.conv2d(l1, num_outputs=64, kernel_size=4, stride=2, padding='VALID', activation_fn=tf.nn.relu)
            l3 = layers.conv2d(l2, num_outputs=64, kernel_size=3, stride=1, padding='VALID', activation_fn=tf.nn.relu)
            l3 = layers.flatten(l3)
            l4 = layers.fully_connected(inputs=l3, num_outputs=512, activation_fn=tf.nn.relu)
            out = layers.fully_connected(inputs=l4, num_outputs=num_actions, activation_fn=None)
        return out

    def calculate_loss(self, q, target_q):
        num_actions = self.env.action_space.n
        q_tmp = self.r + config.gamma * tf.reduce_max(target_q, axis=1)
        q_samp = tf.where(self.done_mask, self.r, q_tmp)
        a_tmp = tf.one_hot(self.a, num_actions)
        q_new = tf.reduce_sum(q * a_tmp, axis=1)
        self.loss = tf.reduce_mean(tf.squared_difference(q_samp, q_new))

    def update_target_q_param(self, q_scope, target_q_scope):
        q_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=q_scope)
        target_q_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_q_scope)
        self.update_target_op = tf.group(*[tf.assign(target_q_var[i], q_var[i]) for i in range(len(q_var))])

    def optimize_param(self, scope):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        scope_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        grads_and_vars = optimizer.compute_gradients(self.loss, scope_variable)
        if config.grad_clip:
            clipped_grads_and_vars = [(tf.clip_by_norm(item[0], config.clip_val), item[1]) for item in
                                      grads_and_vars]
        self.train_op = optimizer.apply_gradients(clipped_grads_and_vars)
        self.grad_norm = tf.global_norm([item[0] for item in grads_and_vars])

    def process_state(self, state):
        state = tf.cast(state, tf.float32)
        state /= config.high
        return state

    def build(self):
        self.make_placeholder()

        s = self.process_state(self.s)
        # print(s.shape)
        self.q = self.calculate_q_value(s, scope='q', reuse=False)

        sp = self.process_state(self.sp)
        self.target_q = self.calculate_q_value(sp, scope='target_q', reuse=False)

        self.update_target_q_param('q', 'target_q')

        self.calculate_loss(self.q, self.target_q)

        self.optimize_param('q')

    def initialize(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.update_target_op)

    def get_best_action(self, s):
        action_value = self.sess.run(self.q, feed_dict={self.s: s})[0]
        return np.argmax(action_value), action_value

    def update_step(self, t, replay_buffer, lr):
        s_batch, a_batch, r_batch, done_mask_batch, sp_batch = replay_buffer.sample(
            config.batch_size)

        fd = {
            # inputs
            self.s: s_batch,
            self.a: a_batch,
            self.r: r_batch,
            self.sp: sp_batch,
            self.done_mask: done_mask_batch,
            self.lr: lr,
        }

        loss_eval, grad_norm_eval, _ = self.sess.run([self.loss, self.grad_norm, self.train_op], feed_dict=fd)
        return loss_eval, grad_norm_eval

    def train(self, exp_schedule, lr_schedule):
        """
        Performs training of Q

        Args:
            exp_schedule: Exploration instance s.t.
                exp_schedule.get_action(best_action) returns an action
            lr_schedule: Schedule for learning rate
        """

        # initialize replay buffer and variables
        # replay_buffer = ReplayBuffer(config.buffer_size, config.state_history)
        replay_buffer = ReplayBuffer(config.buffer_size, [84, 84, 1], config.state_history)
        t = 0
        episode = 1
        # interact with environment
        while t < config.nsteps_train:
            total_reward = 0
            state = self.env.reset()
            while True:
                t += 1
                if config.render_train: self.env.render()
                # replay memory stuff
                # idx = replay_buffer.store_frame(state)
                idx = replay_buffer.store(state)
                # q_input = replay_buffer.encode_recent_observation()
                q_input = replay_buffer.encode_recent_obs()
                # chose action according to current Q and exploration
                best_action, q_values = self.get_best_action(q_input)
                action = exp_schedule.get_action(best_action)
                exp_schedule.update(t)

                # perform action in env
                new_state, reward, done, info = self.env.step(action)

                # store the transition
                # replay_buffer.store_effect(idx, action, reward, done)
                replay_buffer.store_effect(action, reward, done)
                state = new_state

                # perform a training step
                loss_eval, grad_eval = self.train_step(t, replay_buffer, lr_schedule.epsilon)
                lr_schedule.update(t)
                if t%1000==0:
                    print('time stpe = ', t, ' ,epsilon = ', exp_schedule.epsilon, ' ,lr = ', lr_schedule.epsilon
                          , ', loss = ', loss_eval, ' .........')
                # count reward
                total_reward += reward
                if done or t >= config.nsteps_train:
                    break
            print('episode ', episode, ' loss = ', loss_eval, ', reward = ', total_reward)
            episode += 1

    def train_step(self, t, replay_buffer, lr):
        loss_eval, grad_eval = 0, 0

        # perform training step
        if t > config.learning_start and t % config.learning_freq == 0:
            loss_eval, grad_eval = self.update_step(t, replay_buffer, lr)

        # occasionaly update target network with q network
        if t % config.target_update_freq == 0:
            self.sess.run(self.update_target_op)

        return loss_eval, grad_eval

    def run(self, exp_schedule, lr_schedule):
        self.initialize()

        self.train(exp_schedule, lr_schedule)

