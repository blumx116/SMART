from itertools import cycle, count
import tempfile
import os

import matplotlib.pyplot as plt
import moviepy.editor as mpy
import numpy as np
import tensorflow as tf
from tensorflow import summary

from env.minigrid.wrappers import find, onehot2directedpoint
from misc.typevars import Option, Transition


def _encode_gif_bytes_(im_thwc, fps=4):
  with tempfile.NamedTemporaryFile() as f: fname = f.name + '.gif'
  clip = mpy.ImageSequenceClip(list(im_thwc), fps=fps)
  clip.write_gif(fname, verbose=False, progress_bar=False)

  with open(fname, 'rb') as f: enc_gif = f.read()
  os.remove(fname)

  return enc_gif

def gif_summary(im_thwc, fps=4):
  """
  Given a 4D numpy tensor of images (TxHxWxC), encode a gif into a Summary protobuf.
  NOTE: Tensor must be in the range [0, 255] as opposed to the usual small float values.
  """
  # create a tensorflow image summary protobuf:
  thwc = im_thwc.shape
  im_summ = tf.compat.v1.Summary.Image()
  im_summ.height = thwc[1]
  im_summ.width = thwc[2]
  im_summ.colorspace = 3 # fix to 3 for RGB
  im_summ.encoded_image_string = _encode_gif_bytes_(im_thwc, fps)

  # create a serialized summary obj:
  summ = tf.compat.v1.Summary()
  summ.value.add(image=im_summ)
  return summ.SerializeToString()

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
            test_after_episode = False
        if viz_after_episode:
            vizfn(agent, ep, ts)
            viz_after_episode = False

    if savefn is not None:
        savefn(agent)

def test(agent, ep, ts):
    ...

def make_visualize(env, writer, seeds=None):
    if seeds is None:
        seeds = [None] * 5
    seeds = list(seeds)
    def visualize(agent, ep, ts):
        images = []
        for seed in seeds:
            if seed is not None:
                env.seed(seed)
            state = env.reset()
            goal_point = find(state, 'Goal')
            option = Option(goal_point, depth=0)
            agent.reset(env, option, random_seed=3)

            images.append([plt.imshow(env.render('rgb_array'), animated=True)])
            done = False
            while not done:
                action = agent.act(state, option)
                state, reward, done, _ = env.step(action)
                options = _get_option_tree_(agent)
                print(f"@{onehot2directedpoint(state)} : {reward} => {options}")
                rendered = _render_options_(env.render('rgb_array'), options)
                images.append(rendered)
        gif = np.stack(images, 0)
        gif = gif_summary(gif, fps=24)
        with writer.as_default():
            summary.experimental.write_raw_pb(gif, step=ts, name='wow gifs')
    return visualize
