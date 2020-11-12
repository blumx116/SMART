from itertools import cycle, count
import tempfile
from typing import Iterable, Dict, Any, List
import os

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from agent import SMARTAgent
from env.minigrid.types import Point
from env.minigrid.wrappers import find, onehot2directedpoint
from misc.typevars import Option, Transition

def _get_seeds_(settings):
    if 'seed' in settings:
        if isinstance(settings['seed'], int):
            start = settings['seed']
        else:
            return cycle(settings['seed'])
    else:
        start = 0
    return count(start, 7)

def _render_options_(rgb_array, options):
    #tiles of size 32 x 32
    for option in options:
        tile_ur: np.ndarray = option.value.astype(np.int32) * 32
        tile_ur = tile_ur[::-1]
        for y in range(4, 28):
            for x in range(4, 28):
                new_x, new_y = tile_ur + np.asarray([x, y])

                colors = [[255, 0, 0], [255, 102, 102], [255, 128, 0], [255, 178, 102], [255, 255, 0], [255, 255, 102]]
                rgb_array[new_x, new_y, :] = np.asarray(colors[option.depth])
    return rgb_array

def _get_option_tree_(agent):
    result = []
    option_node = agent.current_option_node
    prev_option = None
    while option_node is not None:
        if prev_option is None or option_node.left == prev_option:
            result.append(option_node.value)
        prev_option = option_node
        option_node = option_node.parent
    return result


def train(agent, env, settings, testfn=None, vizfn=None, savefn=None):
    seeds = _get_seeds_(settings)
    ts = 0
    test_after_episode = False
    viz_after_episode = False

    for ep in range(settings['N_EPISODES']):
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

            if settings['TEST_FREQ'] is not None and ts % settings['TEST_FREQ'] == 0:
                test_after_episode = True
            if settings['VIZ_FREQ'] is not None and ts % settings['VIZ_FREQ'] == 0:
                viz_after_episode = True

            agent.optimize()

        if test_after_episode:
            testfn(agent, ep, ts)
            test_after_episode = False
        if viz_after_episode:
            vizfn(agent, ep, ts)
            viz_after_episode = False

    if savefn is not None:
        savefn(agent)

def make_simple_minigrid_test(
        env,
        writer: SummaryWriter,
        seeds: Iterable[int] = None):
    if seeds is None:
        seeds = [None] * 5
    seeds = list(seeds)

    def test(agent: SMARTAgent, ep, ts):
        rewards = [0] * len(seeds)
        for i, seed in enumerate(seeds):
            env.seed(seed)
            state = env.reset()

            goal_point = find(state, 'Goal')
            option = Option(goal_point, depth=0)
            agent.reset(env, option, random_seed=3)

            done = False

            while not done:
                action = agent.act(state)
                state, reward, done, info = env.step(action)

                rewards[seed] += reward

        for i, seed in enumerate(seeds):
            writer.add_scalar(f"Test Reward: {seed}", rewards[i], global_step=ts)

    return test

def make_mesh_grid(
        possibilities: List[Option[Point]],
        probabilities: List[float]) -> (np.ndarray, ) * 3:
    xy = np.vstack([p.value for p in possibilities])
    # np.ndarray[int8] : [n_possibilities, 2]
    maxes = np.max(xy, axis=0)
    # np.ndarray[int8] : [2,]
    mins = np.min(xy, axis=0)
    # np.ndarray[int8]
    zz = np.zeros(shape=1+maxes-mins, dtype=np.float)
    for option, proba in zip(possibilities, probabilities):
        pos: np.ndarray = option.value - mins
        zz[pos] = proba
    xs = np.arange(mins[0], maxes[0]+1, step=1)
    ys = np.arange(mins[1], maxes[1]+1, step=1)
    xx, yy = np.meshgrid(xs, ys)
    return xx, yy, zz

def render_mesh(
        xx: np.ndarray,
        yy: np.ndarray,
        zz: np.ndarray) -> plt.Figure:
    fig: Figure = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.view_init(30, 135)
    ax1.plot_surface(xx, yy, zz, alpha=0.3, cmap='magma')
    ax2.plot_surface(xx, yy, zz, alpha=0.3, cmap='magma')
    return fig


def visualize_decision(
        agent: SMARTAgent,
        state,
        writer: SummaryWriter,
        tag: str,
        ep: int=None,
        ts: int=None) -> None:
    prev_option = agent._prev_option_()
    parent_option = agent.current_option_node.value
    possibilities = agent.generator.generate(
        state, prev_option, parent_option)
    probabilities = agent.evaluator._selection_probabilities_(
        state, possibilities, prev_option, parent_option)
    xx, yy, zz = make_mesh_grid(possibilities, probabilities)
    fig = render_mesh(xx, yy, zz)
    writer.add_figure(tag, fig, global_step=ts)

def make_visualize(env,
                   writer: SummaryWriter, seeds=None):
    if seeds is None:
        seeds = [None] * 5
    seeds = list(seeds)
    def visualize(
            agent: SMARTAgent,
            ep: int,
            ts: int):
        images = []
        for seed in seeds:
            if seed is not None:
                env.seed(seed)
            state = env.reset()
            goal_point = find(state, 'Goal')
            option = Option(goal_point, depth=0)
            agent.reset(env, option, random_seed=3)

            visualize_decision(agent, state, writer, f'likelihoods: {seed}', ep, ts)

            images.append(env.render('rgb_array'))
            done = False
            while not done:
                action = agent.act(state, option)
                state, reward, done, _ = env.step(action)
                options = _get_option_tree_(agent)
                print(f"@{onehot2directedpoint(state)} : {reward} => {options}")
                rendered = _render_options_(env.render('rgb_array'), options)
                images.append(rendered)
        gif = np.stack(images, 0)
        # np.ndarray [t, imx, imy, 3]
        gif_tensor: torch.Tensor = torch.from_numpy(gif).type(torch.uint8).unsqueeze(0)
        # torch.Tensor[uint8] [1, t, imx, imy, 3]
        gif_tensor = gif_tensor.permute(0, 1, 4, 2, 3)
        writer.add_video('sample trajectory', gif_tensor, global_step=ts)
    return visualize

def summarize(agent, env,
              settings: Dict[str, Any],
              seeds: List[int],
              writer: SummaryWriter):
    rewards = [0] * len(seeds)
    for i, seed in enumerate(seeds):
        if seed is not None:
            env.seed(seed)
        state = env.reset()
        goal_point = find(state, 'Goal')
        option = Option(goal_point, depth=0)
        agent.reset(env, option, random_seed=3)

        done = False
        while not done:
            action = agent.act(state, option)
            state, reward, done, _ = env.step(action)

            rewards[i] += reward

    writer.add_hparams(
        {key: value for (key, value) in settings.items() if key not in ['device']},
        {'average reward': np.mean(rewards),
         'min reward': np.min(rewards),
         'max reward': np.max(rewards)})
