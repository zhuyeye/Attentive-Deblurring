from __future__ import print_function
import os
import time
import random
import datetime
import scipy.misc
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from util.util import *
import skimage


class DEBLUR(object):
    def __init__(self, args):
        self.args = args
        self.n_levels = 3
        self.scale = 0.5
        self.chns = 3

        self.crop_size = 256
        self.data_list = open(args.datalist, 'rt').read().splitlines()
        self.data_list = list(map(lambda x: x.split(' '), self.data_list))
        random.shuffle(self.data_list)
        self.train_dir = os.path.join('./checkpoints', args.model)
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.data_size = (len(self.data_list)) // self.batch_size
        self.max_steps = int(self.epoch * self.data_size)
        self.learning_rate = args.learning_rate

    def diff2(self, img, dim):
        img_g = tf.image.rgb_to_grayscale(img)
        #        n,h,w,c=img_g.get_shape().as_list()
        if dim == 0:
            filter = tf.constant([1.0, -1.0], shape=[1, 2, 1, 1])
        else:
            filter = tf.constant([1.0, -1.0], shape=[2, 1, 1, 1])
        return tf.nn.convolution(img_g, filter, 'VALID')

    def input_producer(self, batch_size=10):
        def flip(img_in, img_gt):
            flag = tf.random.uniform([], minval=0, maxval=2, dtype=tf.dtypes.int32)
            img_in = tf.cond(tf.equal(flag, tf.constant(1)), lambda: tf.image.flip_left_right(img_in), lambda: img_in)
            img_gt = tf.cond(tf.equal(flag, tf.constant(1)), lambda: tf.image.flip_left_right(img_gt), lambda: img_gt)
            flag = tf.random.uniform([], minval=0, maxval=2, dtype=tf.dtypes.int32)
            img_in = tf.cond(tf.equal(flag, tf.constant(1)), lambda: tf.image.flip_up_down(img_in), lambda: img_in)
            img_gt = tf.cond(tf.equal(flag, tf.constant(1)), lambda: tf.image.flip_up_down(img_gt), lambda: img_gt)
            return img_in, img_gt

        def read_data():
            img_a = tf.image.decode_image(tf.read_file(tf.string_join(['./training_set/train/', self.data_queue[0]])),
                                          channels=3)
            img_b = tf.image.decode_image(tf.read_file(tf.string_join(['./training_set/train/', self.data_queue[1]])),
                                          channels=3)
            img_a, img_b = preprocessing([img_a, img_b])
            img_a, img_b = flip(img_a, img_b)
            return img_a, img_b

        def preprocessing(imgs):
            imgs = [tf.cast(img, tf.float32) / 255.0 for img in imgs]
            img_crop = tf.random_crop(tf.stack(imgs, axis=0), [2, self.crop_size, self.crop_size, self.chns])
            return tf.unstack(img_crop, axis=0)

        with tf.variable_scope('input'):
            List_all = tf.convert_to_tensor(self.data_list, dtype=tf.string)

            gt_list = List_all[:, 0]
            in_list = List_all[:, 1]

            self.data_queue = tf.train.slice_input_producer([in_list, gt_list], capacity=20)
            image_in, image_gt = read_data()
            batch_in, batch_gt = tf.train.batch([image_in, image_gt], batch_size=batch_size, num_threads=8, capacity=20)

        return batch_in, batch_gt

    def generator(self, inputs, reuse=False, scope='g_net'):
        n, h, w, c = inputs.get_shape().as_list()

        x_unwrap = []
        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                activation_fn=tf.nn.relu,
                                padding='SAME',
                                normalizer_fn=None,
                                weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                biases_initializer=tf.constant_initializer(0.0)):

                inp_pred = inputs
                for i in range(self.n_levels):
                    scale = self.scale**(self.n_levels - i - 1)
                    hi = int(round(h * scale))
                    wi = int(round(w * scale))
                    inp_blur = tf.image.resize_images(inputs, [hi, wi], method=0)
                    inp_pred = tf.stop_gradient(tf.image.resize_images(inp_pred, [hi, wi], method=0))
                    inp_all = tf.concat([inp_blur, inp_pred], axis=3, name='inp')
                    # encoder
                    conv1_1 = slim.conv2d(inp_all, 32, [5, 5], scope='enc1_1')
                    conv1_2 = RDAB(conv1_1, 32, 5, scope='enc1_2')
                    conv1_3 = RDAB(conv1_2, 32, 5, scope='enc1_3')
                    conv1_4 = RDAB(conv1_3, 32, 5, scope='enc1_4')
                    conv2_1 = slim.conv2d(conv1_4, 64, [5, 5], stride=2, scope='enc2_1')
                    conv2_2 = RDAB(conv2_1, 64, 5, scope='enc2_2')
                    conv2_3 = RDAB(conv2_2, 64, 5, scope='enc2_3')
                    conv2_4 = RDAB(conv2_3, 64, 5, scope='enc2_4')
                    conv3_1 = slim.conv2d(conv2_4, 128, [5, 5], stride=2, scope='enc3_1')
                    conv3_2 = RDAB(conv3_1, 128, 5, scope='enc3_2')
                    conv3_3 = RDAB(conv3_2, 128, 5, scope='enc3_3')
                    conv3_4 = RDAB(conv3_3, 128, 5, scope='enc3_4')

                    # decoder
                    deconv3_3 = RDAB(conv3_4, 128, 5, scope='dec3_3')
                    deconv3_2 = RDAB(deconv3_3, 128, 5, scope='dec3_2')
                    deconv3_1 = RDAB(deconv3_2, 128, 5, scope='dec3_1')
                    deconv2_4 = slim.conv2d_transpose(deconv3_1, 64, [4, 4], stride=2, scope='dec2_4')
                    cat2 = deconv2_4 + conv2_4
                    deconv2_3 = RDAB(cat2, 64, 5, scope='dec2_3')
                    deconv2_2 = RDAB(deconv2_3, 64, 5, scope='dec2_2')
                    deconv2_1 = RDAB(deconv2_2, 64, 5, scope='dec2_1')
                    deconv1_4 = slim.conv2d_transpose(deconv2_1, 32, [4, 4], stride=2, scope='dec1_4')
                    cat1 = deconv1_4 + conv1_4
                    deconv1_3 = RDAB(cat1, 32, 5, scope='dec1_3')
                    deconv1_2 = RDAB(deconv1_3, 32, 5, scope='dec1_2')
                    deconv1_1 = RDAB(deconv1_2, 32, 5, scope='dec1_1')
                    inp_pred = slim.conv2d(deconv1_1, self.chns, [5, 5], activation_fn=None, scope='dec1_0')

                    if i >= 0:
                        x_unwrap.append(inp_pred)
                    if i == 0:
                        tf.get_variable_scope().reuse_variables()

            return x_unwrap

    def build_model(self):
        img_in, img_gt = self.input_producer(self.batch_size)
        tf.summary.image('img_in', im2uint8(img_in))
        tf.summary.image('img_gt', im2uint8(img_gt))
        print('img_in, img_gt', img_in.get_shape(), img_gt.get_shape())

        # generator
        [x_unwrap, _] = self.generator(img_in, reuse=False, scope='g_net')
        # calculate multi-scale loss
        self.loss_total = 0
        self.psnr = 0
        self.ssim = 0
        for i in range(self.n_levels):
            _, hi, wi, _ = x_unwrap[i].get_shape().as_list()
            gt_i = tf.image.resize_images(img_gt, [hi, wi], method=0)
            img_i = tf.image.resize_images(img_in, [hi, wi], method=0)
            loss = tf.reduce_mean((gt_i - x_unwrap[i])**2)
            self.loss_total += loss
            tf.summary.image('out_' + str(i), im2uint8(x_unwrap[i]))
            tf.summary.scalar('loss_' + str(i), loss)
        loss_p = tf.reduce_mean((self.diff2(gt_i, 0) - self.diff2(x_unwrap[2], 0))**2) + tf.reduce_mean(
            (self.diff2(gt_i, 1) - self.diff2(x_unwrap[2], 1))**2)
        tf.summary.scalar('loss_p', 2.5 * loss_p)
        self.loss_total += loss_p * 2.5
        for i in range(self.batch_size):
            self.ssim = self.ssim + tf.image.ssim(
                img_gt[i, :, :, :], x_unwrap[self.n_levels - 1][i, :, :, :], max_val=1.0)
            self.psnr = self.psnr + tf.image.psnr(
                img_gt[i, :, :, :], x_unwrap[self.n_levels - 1][i, :, :, :], max_val=1.0)
        self.psnr = self.psnr / self.batch_size
        self.ssim = self.ssim / self.batch_size
        # losses
        tf.summary.scalar('loss_total', self.loss_total)

        # training vars
        all_vars = tf.trainable_variables()
        self.all_vars = all_vars
        self.g_vars = [var for var in all_vars if 'g_net' in var.name]
        self.lstm_vars = [var for var in all_vars if 'LSTM' in var.name]
        for var in all_vars:
            print(var.name)

    def train(self):
        def get_optimizer(loss, global_step=None, var_list=None, is_gradient_clip=False):
            train_op = tf.train.AdamOptimizer(self.lr)
            if is_gradient_clip:
                grads_and_vars = train_op.compute_gradients(loss, var_list=var_list)
                unchanged_gvs = [(grad, var) for grad, var in grads_and_vars if not 'LSTM' in var.name]
                rnn_grad = [grad for grad, var in grads_and_vars if 'LSTM' in var.name]
                rnn_var = [var for grad, var in grads_and_vars if 'LSTM' in var.name]
                capped_grad, _ = tf.clip_by_global_norm(rnn_grad, clip_norm=3)
                capped_gvs = list(zip(capped_grad, rnn_var))
                train_op = train_op.apply_gradients(grads_and_vars=capped_gvs + unchanged_gvs, global_step=global_step)
            else:
                train_op = train_op.minimize(loss, global_step, var_list)
            return train_op

        global_step = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False)
        self.global_step = global_step

        # build model
        self.build_model()
        # learning rate decay
        self.lr = tf.train.polynomial_decay(self.learning_rate,
                                            global_step,
                                            self.max_steps,
                                            end_learning_rate=0.0,
                                            power=0.3)
        tf.summary.scalar('learning_rate', self.lr)

        # training operators
        train_gnet = get_optimizer(self.loss_total, global_step, self.all_vars)

        # session and thread
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        #        sess=tf.Session()
        self.sess = sess
        sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # training summary
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph, flush_secs=30)
        t_psnr = 0
        t_ssim = 0
        cnt = 0
        for step in range(sess.run(global_step), self.max_steps + 1):

            start_time = time.time()

            # update G network
            _, loss_total_val, psnr, ssim = sess.run([train_gnet, self.loss_total, self.psnr, self.ssim])
            t_psnr = (t_psnr + psnr)
            t_ssim += ssim
            cnt = cnt + 1
            duration = time.time() - start_time
            # print loss_value
            assert not np.isnan(loss_total_val), 'Model diverged with loss = NaN'

            if step % 5 == 0:
                num_examples_per_step = self.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = (%.5f) psnr/ssim = (%.5f[%.5f]/%.5f)(%.1f data/s; %.3f s/bch)')
                print(format_str % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), step, loss_total_val, psnr,
                                    t_psnr / cnt, t_ssim / cnt, examples_per_sec, sec_per_batch))

            if step % 20 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or step == self.max_steps:
                checkpoint_path = os.path.join(self.train_dir, 'checkpoints')
                self.save(sess, checkpoint_path, step)
                t_psnr = 0
                t_ssim = 0
                cnt = 0

    def save(self, sess, checkpoint_dir, step):
        model_name = "deblur.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, sess, checkpoint_dir, step=None):
        print(" [*] Reading checkpoints...")
        model_name = "deblur.model"
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if step is not None:
            ckpt_name = model_name + '-' + str(step)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading intermediate checkpoints... Success")
            return str(step)
        elif ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            ckpt_iter = ckpt_name.split('-')[1]
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading updated checkpoints... Success")
            return ckpt_iter
        else:
            print(" [*] Reading checkpoints... ERROR")
            return False

    def test(self, height, width, input_path, output_path, steps):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        imgsName = []
        gtsName = []
        name = []
        list = os.listdir(input_path)
        for i in range(0, len(list)):
            path = os.path.join(input_path, list[i])
            tempimg = os.listdir(os.path.join(path, 'blur'))
            if not os.path.exists(os.path.join(output_path, list[i])):
                os.makedirs(os.path.join(output_path, list[i]))
            for j in range(0, len(tempimg)):
                imgsName.append(os.path.join(os.path.join(path, 'blur'), tempimg[j]))
                gtsName.append(os.path.join(os.path.join(path, 'sharp'), tempimg[j]))
                name.append(os.path.join(output_path, list[i], tempimg[j]))
        H, W = height, width
        inp_chns = 3
        self.batch_size = 1
        inputs = tf.placeholder(shape=[self.batch_size, H, W, inp_chns], dtype=tf.float32)
        print(inputs.get_shape())
        outputs = self.generator(inputs, reuse=False)

        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        self.saver = tf.train.Saver()
        self.load(sess, self.train_dir + '/checkpoints/', step=steps)

        PSNR = 0
        cnt = 0
        for i in range(0, len(imgsName)):
            blur = scipy.misc.imread(imgsName[i])
            sharp = scipy.misc.imread(gtsName[i])
            h, w, c = blur.shape
            # make sure the width is larger than the height
            rot = False
            if h > w:
                blur = np.transpose(blur, [1, 0, 2])
                rot = True
            h = int(blur.shape[0])
            w = int(blur.shape[1])
            resize = False
            if h > H or w > W:
                scale = min(1.0 * H / h, 1.0 * W / w)
                new_h = int(h * scale)
                new_w = int(w * scale)
                blur = scipy.misc.imresize(blur, [new_h, new_w], 'bicubic')
                resize = True
                blurPad = np.pad(blur, ((0, H - new_h), (0, W - new_w), (0, 0)), 'edge')
            else:
                blurPad = np.pad(blur, ((0, H - h), (0, W - w), (0, 0)), 'edge')
            blurPad = np.expand_dims(blurPad, 0)

            deblur = sess.run(outputs, feed_dict={inputs: blurPad / 255.0})
            res = deblur[-1]
            res = im2uint8(res[0, :, :, :])
            # crop the image into original size
            if resize:
                res = res[:new_h, :new_w, :]
                res = scipy.misc.imresize(res, [h, w], 'bicubic')
            else:
                res = res[:h, :w, :]
            if rot:
                res = np.transpose(res, [1, 0, 2])
            scipy.misc.imsave(name[i], res)
            psnr = skimage.measure.compare_psnr(sharp, res)
            PSNR = PSNR + psnr
            cnt = cnt + 1
            print('Saving results: %s ... psnr=%.5f/%.5f' % (name[i], psnr, PSNR / cnt))
        PSNR = PSNR / len(imgsName)
        print('test psnr= %.5f' % (PSNR))

    def defocus(self, height, width, input_path, output_path, steps):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        imgsName = []
        gtsName = []
        name = []
        blur_path = os.path.join(input_path, 'source')
        gt_path = os.path.join(input_path, 'target')
        blurs = os.listdir(blur_path)
        gts = os.listdir(gt_path)
        blurs.sort()
        gts.sort()
        for i in range(0, len(blurs)):
            imgsName.append(os.path.join(blur_path, blurs[i]))
            gtsName.append(os.path.join(gt_path, gts[i]))
            name.append(os.path.join(output_path, gts[i]))
        H, W = height, width
        inp_chns = 3
        self.batch_size = 1
        inputs = tf.placeholder(shape=[self.batch_size, H, W, inp_chns], dtype=tf.float32)
        print(inputs.get_shape())
        outputs = self.generator(inputs, reuse=False)

        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        self.saver = tf.train.Saver()
        self.load(sess, self.train_dir + '/checkpoints/', step=steps)

        PSNR = 0
        cnt = 0
        for i in range(0, len(imgsName)):
            blur = scipy.misc.imread(imgsName[i])
            sharp = scipy.misc.imread(gtsName[i])
            h, w, c = blur.shape
            # make sure the width is larger than the height
            rot = False
            if h > w:
                blur = np.transpose(blur, [1, 0, 2])
                rot = True
            h = int(blur.shape[0])
            w = int(blur.shape[1])
            resize = False
            if h > H or w > W:
                scale = min(1.0 * H / h, 1.0 * W / w)
                new_h = int(h * scale)
                new_w = int(w * scale)
                blur = scipy.misc.imresize(blur, [new_h, new_w], 'bicubic')
                resize = True
                blurPad = np.pad(blur, ((0, H - new_h), (0, W - new_w), (0, 0)), 'edge')
            else:
                blurPad = np.pad(blur, ((0, H - h), (0, W - w), (0, 0)), 'edge')
            blurPad = np.expand_dims(blurPad, 0)
            s = time.time()
            deblur = sess.run(outputs, feed_dict={inputs: blurPad / 255.0})
            e = time.time()
            res = deblur[-1]
            res = im2uint8(res[0, :, :, :])
            # crop the image into original size
            if resize:
                res = res[:new_h, :new_w, :]
                res = scipy.misc.imresize(res, [h, w], 'bicubic')
            else:
                res = res[:h, :w, :]
            if rot:
                res = np.transpose(res, [1, 0, 2])
            scipy.misc.imsave(name[i], res)
            psnr = skimage.measure.compare_psnr(sharp, res)
            PSNR = PSNR + psnr
            cnt = cnt + 1
            print('Saving results: %s ... time=%.5f...psnr=%.5f/%.5f' % (name[i], e - s, psnr, PSNR / cnt))
        PSNR = PSNR / len(imgsName)
        print('test psnr= %.5f' % (PSNR))
