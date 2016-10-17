# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.

import os
import time

import numpy as np
import scipy.misc

import math
from argparse import ArgumentParser

# default arguments
CONTENT_WEIGHT = 5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 1e2
LEARNING_RATE = 1e1
STYLE_SCALE = 1.0
ITERATIONS = 1000
OPTIMIZER = 'lbfgs'
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content',
            dest='content', help='content image',
            metavar='CONTENT', required=True)
    parser.add_argument('--styles',
            dest='styles',
            nargs='+', help='one or more style images',
            metavar='STYLE', required=True)
    parser.add_argument('--output',
            dest='output', help='output path',
            metavar='OUTPUT', required=True)
    parser.add_argument('--optimizer',
            dest='optimizer', help='lbfgs or adam (default %(default)s)',
            metavar='OPTIMIZER', default=OPTIMIZER)
    parser.add_argument('--iterations', type=int,
            dest='iterations', help='iterations (default %(default)s)',
            metavar='ITERATIONS', default=ITERATIONS)
    parser.add_argument('--print-iterations', type=int,
            dest='print_iterations', help='statistics printing frequency',
            metavar='PRINT_ITERATIONS')
    parser.add_argument('--checkpoint-output',
            dest='checkpoint_output', help='checkpoint output format, e.g. output%%s.jpg',
            metavar='OUTPUT')
    parser.add_argument('--checkpoint-iterations', type=int,
            dest='checkpoint_iterations', help='checkpoint frequency',
            metavar='CHECKPOINT_ITERATIONS')
    parser.add_argument('--width', type=int,
            dest='width', help='output width',
            metavar='WIDTH')
    parser.add_argument('--style-scales', type=float,
            dest='style_scales',
            nargs='+', help='one or more style scales',
            metavar='STYLE_SCALE')
    parser.add_argument('--network',
            dest='network', help='path to network parameters (default %(default)s)',
            metavar='VGG_PATH', default=VGG_PATH)
    parser.add_argument('--content-weight', type=float,
            dest='content_weight', help='content weight (default %(default)s)',
            metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    parser.add_argument('--style-weight', type=float,
            dest='style_weight', help='style weight (default %(default)s)',
            metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)
    parser.add_argument('--style-blend-weights', type=float,
            dest='style_blend_weights', help='style blending weights',
            nargs='+', metavar='STYLE_BLEND_WEIGHT')
    parser.add_argument('--tv-weight', type=float,
            dest='tv_weight', help='total variation regularization weight (default %(default)s)',
            metavar='TV_WEIGHT', default=TV_WEIGHT)
    parser.add_argument('--learning-rate', type=float,
            dest='learning_rate', help='learning rate (default %(default)s)',
            metavar='LEARNING_RATE', default=LEARNING_RATE)
    parser.add_argument('--initial',
            dest='initial', help='initial image',
            metavar='INITIAL')
    parser.add_argument('--savestate-iterations', type=int,
            dest='savestate_iterations', help='savestate frequency',
            metavar='savestate_iterations')
    parser.add_argument('--savestate-path',
            dest='savestate_path', help='Saves current internal state here. overwritten each save',
            metavar='SAVESTATE_PATH')
    parser.add_argument('--savestate-restore-file',
            dest='savestate_restore_file', help='loads saved checkpoint state.',
            metavar='CHECKPOINT_RESTORE_FILE')

    return parser


def main():
    start_time = last_save = time.time()
    parser = build_parser()
    options = parser.parse_args()

    if not os.path.isfile(options.network):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % options.network)

    content_image = imread(os.path.abspath(options.content))
    style_images = [imread(style) for style in options.styles]
    optimizer = options.optimizer

    width = options.width
    if width is not None:
        new_shape = (int(math.floor(float(content_image.shape[0]) /
                content_image.shape[1] * width)), width)
        content_image = scipy.misc.imresize(content_image, new_shape)
    target_shape = content_image.shape
    for i in range(len(style_images)):
        style_scale = STYLE_SCALE
        if options.style_scales is not None:
            style_scale = options.style_scales[i]
        style_images[i] = scipy.misc.imresize(style_images[i], style_scale *
                target_shape[1] / style_images[i].shape[1])

    style_blend_weights = options.style_blend_weights
    if style_blend_weights is None:
        # default is equal weights
        style_blend_weights = [1.0/len(style_images) for _ in style_images]
    else:
        total_blend_weight = sum(style_blend_weights)
        style_blend_weights = [weight/total_blend_weight
                               for weight in style_blend_weights]

    initial = options.initial
    if initial is not None:
        initial = scipy.misc.imresize(imread(initial), content_image.shape[:2])

    if options.checkpoint_output and "%s" not in options.checkpoint_output:
        parser.error("To save intermediate images, the checkpoint output "
                     "parameter must contain `%s` (e.g. `foo%s.jpg`)")

    print('checkpoint_iterations=%d' % options.checkpoint_iterations)

    savestate_path = options.savestate_path
    if savestate_path:
        if (not os.path.isdir(os.path.dirname(os.path.abspath(options.savestate_path)))):
            parser.error('Path not found for --savestate-path %s' % options.savestate_path)
        else:
            if not options.savestate_iterations:
                parser.error('--savestate-path requires you set --savestate-iterations to a number')
            savestate_path = os.path.abspath(options.savestate_path)

    savestate_restore_file = None
    if options.savestate_restore_file:
        if (not os.path.isfile(os.path.abspath(options.savestate_restore_file))):
            parser.error('Path not found for savestate_restore_file %s' % options.savestate_restore_file)
        else:
            savestate_restore_file = os.path.abspath(options.savestate_restore_file)

    from stylize import stylize

    for iteration, image in stylize(
        network=options.network,
        initial=initial,
        content=content_image,
        styles=style_images,
        optimizer=optimizer,
        iterations=options.iterations,
        content_weight=options.content_weight,
        style_weight=options.style_weight,
        style_blend_weights=style_blend_weights,
        tv_weight=options.tv_weight,
        learning_rate=options.learning_rate,
        print_iterations=options.print_iterations,
        checkpoint_iterations=options.checkpoint_iterations,
        savestate_iterations=options.savestate_iterations,
        savestate_path=savestate_path,
        savestate_restore_file=savestate_restore_file
    ):
        output_file = None
        if iteration is not None:
            if options.checkpoint_output:
                output_file = options.checkpoint_output % ('%04d' % iteration)
        else:
            output_file = os.path.abspath(options.output)
        if output_file:
            imsave(output_file, image)
            print('Image saved as: `%s`' % output_file)
            m, s = divmod((time.time() - last_save), 60)
            h, m = divmod(m, 60)
            print('Step Duration %02dm%02ds' % (m, s))

            m, s = divmod((time.time() - start_time), 60)
            h, m = divmod(m, 60)
            print('Total Duration %02dh%02dm%02ds' % (h,m,s))
            last_save = time.time()


def imread(path):
    return scipy.misc.imread(path).astype(np.float)


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)


if __name__ == '__main__':
    main()
