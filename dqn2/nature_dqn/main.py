from nature_dqn.environments.preprocess import greyscale
from nature_dqn.environments.atari_wrapper import deepmind_wrapping
from nature_dqn.schedulers.linear_scheduler import LinearScheduler
from nature_dqn.schedulers.linear_scheduler import LinearExploration
from nature_dqn.models.dqn import DQN
from nature_dqn.configs.config import config
import gym


if __name__ == '__main__':
    # creating environment and wrapping environment
    env = gym.make('PongNoFrameskip-v4')
    env = deepmind_wrapping(env, greyscale, 4)

    # learning rate scheduling and epsilon scheduling
    lr_schedule = LinearScheduler(config.lr_begin, config.lr_end, config.lr_nsteps)
    exp_schedule = LinearExploration(env, config.eps_begin, config.eps_end, config.eps_nsteps)

    # build model
    model = DQN(env)

    # run model
    model.run(exp_schedule, lr_schedule)

