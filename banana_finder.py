import argh
import torch
import numpy as np

from argh import arg
from collections import deque
from unityagents import UnityEnvironment
from dqn_agent import Agent


def dqn(agent, env, brain_name, n_episodes=2500, max_t=1000, eps_start=1.0,
        eps_end=0.01, eps_decay=0.999, train=True):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon,
                           for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode)
                           for decreasing epsilon
    """
    # list containing scores from each episode
    scores = []
    # set length of buffer to last 100 scores
    scores_window = deque(maxlen=100)
    eps = eps_start if train else 0.0
    for i_episode in range(1, n_episodes+1):
        # reset the environment
        env_info = env.reset(train_mode=train)[brain_name]
        # get the current state
        state = env_info.vector_observations[0]
        score = 0
        for _ in range(max_t):
            # select an action
            action = agent.act(state, eps)
            # send the action to the envirinment
            env_info = env.step(action)[brain_name]
            # get the next state
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            if train:
                agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        # save the most recent score
        scores_window.append(score)
        scores.append(score)
        # decrease epsilon
        eps = max(eps_end, eps_decay*eps)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(
            i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_window)))
        if train and np.mean(scores_window) >= 13.0:
            print('\nEnv solved in {:d} episodes!\tAverage Score: {:.2f}'
                  .format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_dqn.pth')
            break
    return scores


@arg('--chkp', help='Checkpoint file. If present, agen will run in eval mode.')
def runner(chkp=None):
    '''
    This function loads the environment and the agent. By default runs in
    training mode, but if a checkpoint file is passed it runs in eval mode.

    Params
    ======
        chkp (None|file): file containing a checkpoint saved during training.
    '''
    env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)
    agent = Agent(state_size=state_size, action_size=action_size, seed=0)

    if chkp:
        checkpoint = torch.load(chkp)
        agent.qnetwork_local.load_state_dict(checkpoint)
        agent.qnetwork_target.load_state_dict(checkpoint)

        dqn(agent, env, brain_name, n_episodes=100, train=False)

    dqn(agent, env, brain_name, train=True)


if __name__ == '__main__':
    argh.dispatch_command(runner)
