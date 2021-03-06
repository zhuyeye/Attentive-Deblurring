import os
import argparse
import tensorflow as tf
import models.models as model


def parse_args():
    parser = argparse.ArgumentParser(description='deblur arguments')
    parser.add_argument('--phase', type=str, default='train', help='determine whether train or test')
    parser.add_argument('--datalist', type=str, default='./datalist_gopro.txt', help='training datalist')
    parser.add_argument('--model', type=str, default='RDAN', help='model name of checkpoint folder')
    parser.add_argument('--steps', help='steps of pretrained model', type=int, default=675000)
    parser.add_argument('--batch_size', help='training batch size', type=int, default=16)
    parser.add_argument('--epoch', help='training epoch number', type=int, default=4000)
    parser.add_argument('--lr', type=float, default=1e-4, dest='learning_rate', help='initial learning rate')
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0', help='use gpu or cpu')
    parser.add_argument('--height',
                        type=int,
                        default=720,
                        help='height for the tensorflow placeholder, should be multiples of 16')
    parser.add_argument('--width',
                        type=int,
                        default=1280,
                        help='width for the tensorflow placeholder, should be multiple of 16 for 3 scales')
    parser.add_argument('--input_path', type=str, default='./testing_set/test', help='input path for testing images')
    parser.add_argument('--output_path', type=str, default='./output/', help='output path for testing images')
    args = parser.parse_args()
    return args


def main(_):
    args = parse_args()
    tf.reset_default_graph()
    # set gpu/cpu mode
    if int(args.gpu_id) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    print(tf.test.is_gpu_available())
    # set up deblur models
    deblur = model.DEBLUR(args)
    if args.phase == 'test':
        if args.model == 'defocus':
            deblur.defocus(args.height, args.width, args.input_path, args.output_path, args.steps)
        else:
            deblur.test(args.height, args.width, args.input_path, args.output_path, args.steps)
    elif args.phase == 'train':
        deblur.train()
    else:
        print('phase should be set to either test or train')


if __name__ == '__main__':
    tf.app.run()
