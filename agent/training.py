from itertools import cycle, count

from env.minigrid.wrappers import find
from misc.typevars import Option, Transition

def get_seeds(settings):
    if 'seed' in settings:
        if isinstance(settings['seed'], int):
            start = settings['seed']
        else:
            return cycle(settings['seed'])
    else:
        start = 0
    return count(start, 7)


def train(agent, env, settings, testfn=None, vizfn=None, savefn=None):
    seeds = get_seeds(settings)
    ts = 0
    test_after_episode = False
    viz_after_episode = False

    for ep in settings['N_EPISODES']:
        env.seed(next(seeds))
        state = env.reset()
        goal_point = find(state, 'Goal')
        option = Option(goal_point, depth=0)
        agent.reset(env, option, random_seed=3)
        done = False

        while not done:
            action = agent.act(state, option)
            state, reward, done, _ = env.step(action)
            agent.view(Transition(state, action, reward))

            ts += 1

            if settings['TEST_FREQ'] is not None and ts % settings['TEST_FREQ']:
                test_after_episode = True
            if settings['VIZ_FREQ'] is not None and  ts % settings['VIZ_FREQ']:
                viz_after_episode = True


        if test_after_episode:
            testfn(agent, ep, ts)
        if viz_after_episode:
            vizfn(agent, ep, ts)

    if savefn is not None:
        savefn(agent)

def test(agent, ep, ts):
    ...

def visualize(agent, ep, ts):
    ...