

import tensorflow as tf
import numpy as np
# from util import bilinear_upsample_weights

"""Define a base class, containing some useful layer functions"""



class Network(object):
    def __init__(self, inputs):
        self.inputs = []
        self.layers = {}
        self.outputs = {}

    """Extract parameters from ckpt file to npy file"""

    def extract(self, data_path, session, saver):
        raise NotImplementedError('Must be subclassed.')

    """Load pre-trained model from numpy data_dict"""

    def load(self, data_dict, session, ignore_missing=True):
        for key in data_dict:
            with tf.variable_scope(key, reuse=True):
                for subkey in data_dict[key]:
                    try:
                        var = tf.get_variable(subkey)
                        session.run(var.assign(data_dict[key][subkey]))
                        print("Assign pretrain model " + subkey + " to " + key)
                    except ValueError:
                        print("Ignore " + key)
                        if not ignore_missing:
                            raise
            if key == 'conv5_down':
                with tf.variable_scope('conv5_3_down', reuse=True):
                    for subkey in data_dict[key]:
                        try:
                            var = tf.get_variable(subkey)
                            session.run(var.assign(data_dict[key][subkey]))
                            print("Assign pretrain model " + subkey + " to " + 'conv5_3_down')
                        except ValueError:
                            print("Ignore " + key)
                            if not ignore_missing:
                                raise
            if key == 'conv5_down':
                with tf.variable_scope('conv5_3_2_down', reuse=True):
                    for subkey in data_dict[key]:
                        try:
                            var = tf.get_variable(subkey)
                            session.run(var.assign(data_dict[key][subkey]))
                            print("Assign pretrain model " + subkey + " to " + 'conv5_3_2_down')
                        except ValueError:
                            print("Ignore " + key)
                            if not ignore_missing:
                                raise
            if key == 'deconv':
                with tf.variable_scope('conv5_3_down_deconv', reuse=True):
                    for subkey in data_dict[key]:
                        try:
                            var = tf.get_variable(subkey)
                            session.run(var.assign(data_dict[key][subkey]))
                            print("Assign pretrain model " + subkey + " to " + 'conv5_3_down_deconv')
                        except ValueError:
                            print("Ignore " + key)
                            if not ignore_missing:
                                raise
                with tf.variable_scope('conv5_3_2_down_deconv', reuse=True):
                    for subkey in data_dict[key]:
                        try:
                            var = tf.get_variable(subkey)
                            session.run(var.assign(data_dict[key][subkey]))
                            print("Assign pretrain model " + subkey + " to " + 'conv5_3_2_down_deconv')
                        except ValueError:
                            print("Ignore " + key)
                            if not ignore_missing:
                                raise
            if key == 'conv3_1' or key == 'conv3_2' or key == 'conv3_3' or key == 'conv4_1' or key == 'conv4_2' or key == 'conv4_3' or key == 'conv5_1' or key == 'conv5_2' or key == 'conv5_3':
                with tf.variable_scope(key + '_2', reuse=True):
                    for subkey in data_dict[key]:
                        try:
                            var = tf.get_variable(subkey)
                            session.run(var.assign(data_dict[key][subkey]))
                            print("Assign pretrain model " + subkey + " to " + key + '_2')
                        except ValueError:
                            print("Ignore" + key)
                            if not ignore_missing:
                                raise

    """Get outputs given key names"""

    def get_output(self, key):
        if key not in self.outputs:
            raise KeyError
        return self.outputs[key]

    """Get parameters given key names"""

    def get_param(self, key):
        if key not in self.layers:
            raise KeyError
        return self.layers[key]['weights'], self.layers[key]['biases']

    """Add conv part of vgg16"""

    def add_conv(self, inputs, num_classes):

        # Conv1
        with tf.variable_scope('conv1_1') as scope:
            w_conv1_1 = tf.get_variable('weights', [3, 3, 3, 64], trainable=False,
                                        initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
            b_conv1_1 = tf.get_variable('biases', [64], trainable=False,
                                        initializer=tf.constant_initializer(0))
            z_conv1_1 = tf.nn.conv2d(inputs, w_conv1_1, strides=[1, 1, 1, 1],
                                     padding='SAME') + b_conv1_1
            a_conv1_1 = tf.nn.relu(z_conv1_1)

        with tf.variable_scope('conv1_2') as scope:
            w_conv1_2 = tf.get_variable('weights', [3, 3, 64, 64], trainable=False,
                                        initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
            b_conv1_2 = tf.get_variable('biases', [64], trainable=False,
                                        initializer=tf.constant_initializer(0))
            z_conv1_2 = tf.nn.conv2d(a_conv1_1, w_conv1_2, strides=[1, 1, 1, 1],
                                     padding='SAME') + b_conv1_2
            a_conv1_2 = tf.nn.relu(z_conv1_2)

        pool1 = tf.nn.max_pool(a_conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')

        # Conv2
        with tf.variable_scope('conv2_1') as scope:
            w_conv2_1 = tf.get_variable('weights', [3, 3, 64, 128], trainable=False,
                                        initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
            b_conv2_1 = tf.get_variable('biases', [128], trainable=False,
                                        initializer=tf.constant_initializer(0))
            z_conv2_1 = tf.nn.conv2d(pool1, w_conv2_1, strides=[1, 1, 1, 1],
                                     padding='SAME') + b_conv2_1
            a_conv2_1 = tf.nn.relu(z_conv2_1)

        with tf.variable_scope('conv2_2') as scope:
            w_conv2_2 = tf.get_variable('weights', [3, 3, 128, 128], trainable=False,
                                        initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
            b_conv2_2 = tf.get_variable('biases', [128], trainable=False,
                                        initializer=tf.constant_initializer(0))
            z_conv2_2 = tf.nn.conv2d(a_conv2_1, w_conv2_2, strides=[1, 1, 1, 1],
                                     padding='SAME') + b_conv2_2
            a_conv2_2 = tf.nn.relu(z_conv2_2)

        pool2 = tf.nn.max_pool(a_conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool2')

        # Conv3
        with tf.variable_scope('conv3_1') as scope:
            w_conv3_1 = tf.get_variable('weights', [3, 3, 128, 256], trainable=False,
                                        initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
            b_conv3_1 = tf.get_variable('biases', [256], trainable=False,
                                        initializer=tf.constant_initializer(0))
            z_conv3_1 = tf.nn.conv2d(pool2, w_conv3_1, strides=[1, 1, 1, 1],
                                     padding='SAME') + b_conv3_1
            a_conv3_1 = tf.nn.relu(z_conv3_1)

        with tf.variable_scope('conv3_2') as scope:
            w_conv3_2 = tf.get_variable('weights', [3, 3, 256, 256], trainable=False,
                                        initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
            b_conv3_2 = tf.get_variable('biases', [256], trainable=False,
                                        initializer=tf.constant_initializer(0))
            z_conv3_2 = tf.nn.conv2d(a_conv3_1, w_conv3_2, strides=[1, 1, 1, 1],
                                     padding='SAME') + b_conv3_2
            a_conv3_2 = tf.nn.relu(z_conv3_2)

        with tf.variable_scope('conv3_3') as scope:
            w_conv3_3 = tf.get_variable('weights', [3, 3, 256, 256], trainable=False,
                                        initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
            b_conv3_3 = tf.get_variable('biases', [256], trainable=False,
                                        initializer=tf.constant_initializer(0))
            z_conv3_3 = tf.nn.conv2d(a_conv3_2, w_conv3_3, strides=[1, 1, 1, 1],
                                     padding='SAME') + b_conv3_3
            a_conv3_3 = tf.nn.relu(z_conv3_3)

        pool3 = tf.nn.max_pool(a_conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool3')

        # Conv4
        with tf.variable_scope('conv4_1') as scope:
            w_conv4_1 = tf.get_variable('weights', [3, 3, 256, 512], trainable=False,
                                        initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
            b_conv4_1 = tf.get_variable('biases', [512], trainable=False,
                                        initializer=tf.constant_initializer(0))
            z_conv4_1 = tf.nn.conv2d(pool3, w_conv4_1, strides=[1, 1, 1, 1],
                                     padding='SAME') + b_conv4_1
            a_conv4_1 = tf.nn.relu(z_conv4_1)

        with tf.variable_scope('conv4_2') as scope:
            w_conv4_2 = tf.get_variable('weights', [3, 3, 512, 512], trainable=False,
                                        initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
            b_conv4_2 = tf.get_variable('biases', [512], trainable=False,
                                        initializer=tf.constant_initializer(0))
            z_conv4_2 = tf.nn.conv2d(a_conv4_1, w_conv4_2, strides=[1, 1, 1, 1],
                                     padding='SAME') + b_conv4_2
            a_conv4_2 = tf.nn.relu(z_conv4_2)

        with tf.variable_scope('conv4_3') as scope:
            w_conv4_3 = tf.get_variable('weights', [3, 3, 512, 512], trainable=False,
                                        initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
            b_conv4_3 = tf.get_variable('biases', [512], trainable=False,
                                        initializer=tf.constant_initializer(0))
            z_conv4_3 = tf.nn.conv2d(a_conv4_2, w_conv4_3, strides=[1, 1, 1, 1],
                                     padding='SAME') + b_conv4_3
            a_conv4_3 = tf.nn.relu(z_conv4_3)

        pool4 = tf.nn.max_pool(a_conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool4')

        # Conv5
        with tf.variable_scope('conv5_1') as scope:
            w_conv5_1 = tf.get_variable('weights', [3, 3, 512, 512], trainable=False,
                                        initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
            b_conv5_1 = tf.get_variable('biases', [512], trainable=False,
                                        initializer=tf.constant_initializer(0))
            z_conv5_1 = tf.nn.conv2d(pool4, w_conv5_1, strides=[1, 1, 1, 1],
                                     padding='SAME') + b_conv5_1
            a_conv5_1 = tf.nn.relu(z_conv5_1)

        with tf.variable_scope('conv5_2') as scope:
            w_conv5_2 = tf.get_variable('weights', [3, 3, 512, 512], trainable=False,
                                        initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
            b_conv5_2 = tf.get_variable('biases', [512], trainable=False,
                                        initializer=tf.constant_initializer(0))
            z_conv5_2 = tf.nn.conv2d(a_conv5_1, w_conv5_2, strides=[1, 1, 1, 1],
                                     padding='SAME') + b_conv5_2
            a_conv5_2 = tf.nn.relu(z_conv5_2)

        with tf.variable_scope('conv5_3') as scope:
            w_conv5_3 = tf.get_variable('weights', [3, 3, 512, 512], trainable=False,
                                        initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
            b_conv5_3 = tf.get_variable('biases', [512], trainable=False,
                                        initializer=tf.constant_initializer(0))
            z_conv5_3 = tf.nn.conv2d(a_conv5_2, w_conv5_3, strides=[1, 1, 1, 1],
                                     padding='SAME') + b_conv5_3
            a_conv5_3 = tf.nn.relu(z_conv5_3)


        with tf.variable_scope('conv5_3_down') as scope:
            w_conv5_3_down = tf.get_variable('weights', [1, 1, 512, 32], trainable=False,
                                             initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
            b_conv5_3_down = tf.get_variable('biases', [32], trainable=False,
                                             initializer=tf.constant_initializer(0))
            z_conv5_3_down = tf.nn.conv2d(a_conv5_3, w_conv5_3_down, strides=[1, 1, 1, 1],
                                          padding='SAME') + b_conv5_3_down
            a_conv5_3_down = tf.nn.relu(z_conv5_3_down)


        ########################################################################3


        with tf.variable_scope('conv5_1_va') as scope:
            w_conv5_1_va = tf.get_variable('weights', [1, 1, 512, 512],
                                             initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
            b_conv5_1_va = tf.get_variable('biases', [512],
                                             initializer=tf.constant_initializer(0))
            z_conv5_1_va = tf.nn.conv2d(pool4, w_conv5_1_va, strides=[1, 1, 1, 1],
                                          padding='SAME') + b_conv5_1_va
            a_conv5_1_va = tf.nn.relu(z_conv5_1_va)
            a_conv5_1_va_out = a_conv5_1 + a_conv5_1_va

        with tf.variable_scope('conv5_2_va') as scope:
            w_conv5_2_va = tf.get_variable('weights', [1, 1, 512, 512],
                                             initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
            b_conv5_2_va = tf.get_variable('biases', [512],
                                             initializer=tf.constant_initializer(0))
            z_conv5_2_va = tf.nn.conv2d(a_conv5_1_va_out, w_conv5_2_va, strides=[1, 1, 1, 1],
                                          padding='SAME') + b_conv5_2_va
            a_conv5_2_va = tf.nn.relu(z_conv5_2_va)
            a_conv5_2_va_out = a_conv5_2 + a_conv5_2_va


        with tf.variable_scope('conv5_3_va') as scope:
            w_conv5_3_va = tf.get_variable('weights', [1, 1, 512, 512],
                                             initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
            b_conv5_3_va = tf.get_variable('biases', [512],
                                             initializer=tf.constant_initializer(0))
            z_conv5_3_va = tf.nn.conv2d(a_conv5_2_va_out, w_conv5_3_va, strides=[1, 1, 1, 1],
                                          padding='SAME') + b_conv5_3_va
            a_conv5_3_va = tf.nn.relu(z_conv5_3_va)
            a_conv5_3_va_out = a_conv5_3 + a_conv5_3_va

        with tf.variable_scope('conv5_3_down_va') as scope:
            w_conv5_3_down_va = tf.get_variable('weights', [1, 1, 512, 32],
                                             initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
            b_conv5_3_down_va = tf.get_variable('biases', [32],
                                             initializer=tf.constant_initializer(0))
            z_conv5_3_down_va = tf.nn.conv2d(a_conv5_3_va_out, w_conv5_3_down_va, strides=[1, 1, 1, 1],
                                          padding='SAME') + b_conv5_3_down_va
            a_conv5_3_down_va = tf.nn.relu(z_conv5_3_down_va)
            a_conv5_3_down_va_out = a_conv5_3_down + a_conv5_3_down_va



        with tf.variable_scope('conv5_3_down_deconv') as scope:
            w_conv5_3_down_deconv = tf.get_variable('weights', [32, 32, 32, 32],
                                                    initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
            b_conv5_3_down_deconv = tf.get_variable('biases', [1],
                                                    initializer=tf.constant_initializer(0))
            z_conv5_3_down_deconv = tf.nn.conv2d_transpose(a_conv5_3_down_va_out, w_conv5_3_down_deconv,
                                                           [self.batch_num, self.max_size[0], self.max_size[1], 32],
                                                           strides=[1, 16, 16, 1], padding='SAME',
                                                           name='z') + b_conv5_3_down_deconv
            a_conv5_3_down_deconv = tf.nn.relu(z_conv5_3_down_deconv)

        with tf.variable_scope('conv5_3_down_final') as scope:
            w_conv5_3_down_final = tf.get_variable('weights', [1, 1, 32, 1],
                                             initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
            b_conv5_3_down_final = tf.get_variable('biases', [1],
                                             initializer=tf.constant_initializer(0))
            z_conv5_3_down_final = tf.nn.conv2d(a_conv5_3_down_deconv, w_conv5_3_down_final, strides=[1, 1, 1, 1],
                                          padding='SAME') + b_conv5_3_down_final
            a_conv5_3_down_final = tf.nn.sigmoid(z_conv5_3_down_final)


        key1 = 'conv5_1_va_'
        key2 = 'conv5_2_va_'
        key3 = 'conv5_3_va_'
        key4 = 'conv5_3_down_va_'
        key5 = 'conv5_3_down_deconv_'
        key6 = 'conv5_3_down_final_'

        for i in range(2, 33):

            with tf.variable_scope(key1 + str(i)) as scope:
                exec("w_conv5_1_va_%s=tf.get_variable('weights', [1, 1, 512, 512],initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))"%i)
                exec("b_conv5_1_va_%s=tf.get_variable('biases', [512],initializer=tf.constant_initializer(0))"%i)
                exec("z_conv5_1_va_%s=tf.nn.conv2d(pool4, w_conv5_1_va_%s, strides=[1, 1, 1, 1],padding='SAME') + b_conv5_1_va_%s"%(i,i,i))
                exec("a_conv5_1_va_%s=tf.nn.relu(z_conv5_1_va_%s)"%(i,i))
                exec("a_conv5_1_va_out_%s = a_conv5_1 + a_conv5_1_va_%s"%(i,i))

            with tf.variable_scope(key2 + str(i)) as scope:
                exec("w_conv5_2_va_%s=tf.get_variable('weights', [1, 1, 512, 512],initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))"%i)
                exec("b_conv5_2_va_%s=tf.get_variable('biases', [512],initializer=tf.constant_initializer(0))"%i)
                exec("z_conv5_2_va_%s=tf.nn.conv2d(a_conv5_1_va_out_%s, w_conv5_2_va_%s, strides=[1, 1, 1, 1],padding='SAME') + b_conv5_2_va_%s"%(i,i,i,i))
                exec("a_conv5_2_va_%s=tf.nn.relu(z_conv5_2_va_%s)"%(i,i))
                exec("a_conv5_2_va_out_%s = a_conv5_2 + a_conv5_2_va_%s"%(i,i))

            with tf.variable_scope(key3 + str(i)) as scope:
                exec("w_conv5_3_va_%s=tf.get_variable('weights', [1, 1, 512, 512],initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))"%i)
                exec("b_conv5_3_va_%s=tf.get_variable('biases', [512],initializer=tf.constant_initializer(0))"%i)
                exec("z_conv5_3_va_%s=tf.nn.conv2d(a_conv5_2_va_out_%s, w_conv5_3_va_%s, strides=[1, 1, 1, 1],padding='SAME') + b_conv5_3_va_%s"%(i,i,i,i))
                exec("a_conv5_3_va_%s=tf.nn.relu(z_conv5_3_va_%s)"%(i,i))
                exec("a_conv5_3_va_out_%s = a_conv5_3 + a_conv5_3_va_%s"%(i,i))

            with tf.variable_scope(key4 + str(i)) as scope:
                exec("w_conv5_3_down_va_%s=tf.get_variable('weights', [1, 1, 512, 32],initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))"%i)
                exec("b_conv5_3_down_va_%s=tf.get_variable('biases', [32],initializer=tf.constant_initializer(0))"%i)
                exec("z_conv5_3_down_va_%s=tf.nn.conv2d(a_conv5_3_va_out_%s, w_conv5_3_down_va_%s, strides=[1, 1, 1, 1],padding='SAME') + b_conv5_3_down_va_%s"%(i,i,i,i))
                exec("a_conv5_3_down_va_%s=tf.nn.relu(z_conv5_3_down_va_%s)"%(i,i))
                exec("a_conv5_3_down_va_out_%s = a_conv5_3_down + a_conv5_3_down_va_%s"%(i,i))

            with tf.variable_scope(key5 + str(i)) as scope:
                exec("z_conv5_3_down_deconv_%s = tf.nn.conv2d_transpose(a_conv5_3_down_va_out_%s, w_conv5_3_down_deconv,[self.batch_num, self.max_size[0], self.max_size[1], 32],strides=[1, 16, 16, 1], padding='SAME',name='z') + b_conv5_3_down_deconv"%(i,i))
                exec("a_conv5_3_down_deconv_%s = tf.nn.relu(z_conv5_3_down_deconv_%s)"%(i,i))

            with tf.variable_scope(key6 + str(i)) as scope:
                exec("z_conv5_3_down_final_%s = tf.nn.conv2d(a_conv5_3_down_deconv_%s, w_conv5_3_down_final, strides=[1, 1, 1, 1],padding='SAME') + b_conv5_3_down_final"%(i,i))
                exec("a_conv5_3_down_final_%s = tf.nn.sigmoid(z_conv5_3_down_final_%s)"%(i,i))


        final = a_conv5_3_down_final
        for i in range(2, 33):
            final += eval("a_conv5_3_down_final_%s"%i)
        final = final / 32

        # name_str = 'best_{}'.format(i)
        # name_str = str("123")

        # # 2conv3
        # with tf.variable_scope('conv3_1_2') as scope:
        #     w_conv3_1_2 = tf.get_variable('weights', [3, 3, 128, 256],
        #                                   initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
        #     b_conv3_1_2 = tf.get_variable('biases', [256],
        #                                   initializer=tf.constant_initializer(0))
        #     z_conv3_1_2 = tf.nn.conv2d(pool2, w_conv3_1_2, strides=[1, 1, 1, 1],
        #                                padding='SAME') + b_conv3_1_2
        #     a_conv3_1_2 = tf.nn.relu(z_conv3_1_2)
        #
        # with tf.variable_scope('conv3_2_2') as scope:
        #     w_conv3_2_2 = tf.get_variable('weights', [3, 3, 256, 256],
        #                                   initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
        #     b_conv3_2_2 = tf.get_variable('biases', [256],
        #                                   initializer=tf.constant_initializer(0))
        #     z_conv3_2_2 = tf.nn.conv2d(a_conv3_1_2, w_conv3_2_2, strides=[1, 1, 1, 1],
        #                                padding='SAME') + b_conv3_2_2
        #     a_conv3_2_2 = tf.nn.relu(z_conv3_2_2)
        #
        # with tf.variable_scope('conv3_3_2') as scope:
        #     w_conv3_3_2 = tf.get_variable('weights', [3, 3, 256, 256],
        #                                   initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
        #     b_conv3_3_2 = tf.get_variable('biases', [256],
        #                                   initializer=tf.constant_initializer(0))
        #     z_conv3_3_2 = tf.nn.conv2d(a_conv3_2_2, w_conv3_3_2, strides=[1, 1, 1, 1],
        #                                padding='SAME') + b_conv3_3_2
        #     a_conv3_3_2 = tf.nn.relu(z_conv3_3_2)
        #
        # pool3_2 = tf.nn.max_pool(a_conv3_3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        #                          padding='SAME', name='pool3_2')
        # # 2conv4
        # with tf.variable_scope('conv4_1_2') as scope:
        #     w_conv4_1_2 = tf.get_variable('weights', [3, 3, 256, 512],
        #                                   initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
        #     b_conv4_1_2 = tf.get_variable('biases', [512],
        #                                   initializer=tf.constant_initializer(0))
        #     z_conv4_1_2 = tf.nn.conv2d(pool3_2, w_conv4_1_2, strides=[1, 1, 1, 1],
        #                                padding='SAME') + b_conv4_1_2
        #     a_conv4_1_2 = tf.nn.relu(z_conv4_1_2)
        #
        # with tf.variable_scope('conv4_2_2') as scope:
        #     w_conv4_2_2 = tf.get_variable('weights', [3, 3, 512, 512],
        #                                   initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
        #     b_conv4_2_2 = tf.get_variable('biases', [512],
        #                                   initializer=tf.constant_initializer(0))
        #     z_conv4_2_2 = tf.nn.conv2d(a_conv4_1_2, w_conv4_2_2, strides=[1, 1, 1, 1],
        #                                padding='SAME') + b_conv4_2_2
        #     a_conv4_2_2 = tf.nn.relu(z_conv4_2_2)
        #
        # with tf.variable_scope('conv4_3_2') as scope:
        #     w_conv4_3_2 = tf.get_variable('weights', [3, 3, 512, 512],
        #                                   initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
        #     b_conv4_3_2 = tf.get_variable('biases', [512],
        #                                   initializer=tf.constant_initializer(0))
        #     z_conv4_3_2 = tf.nn.conv2d(a_conv4_2_2, w_conv4_3_2, strides=[1, 1, 1, 1],
        #                                padding='SAME') + b_conv4_3_2
        #     a_conv4_3_2 = tf.nn.relu(z_conv4_3_2)
        #
        # pool4_2 = tf.nn.max_pool(a_conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        #                          padding='SAME', name='pool4_2')
        # # 2conv5
        # with tf.variable_scope('conv5_1_2') as scope:
        #     w_conv5_1_2 = tf.get_variable('weights', [3, 3, 512, 512],
        #                                   initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
        #     b_conv5_1_2 = tf.get_variable('biases', [512],
        #                                   initializer=tf.constant_initializer(0))
        #     z_conv5_1_2 = tf.nn.conv2d(pool4_2, w_conv5_1_2, strides=[1, 1, 1, 1],
        #                                padding='SAME') + b_conv5_1_2
        #     a_conv5_1_2 = tf.nn.relu(z_conv5_1_2)
        #
        # with tf.variable_scope('conv5_2_2') as scope:
        #     w_conv5_2_2 = tf.get_variable('weights', [3, 3, 512, 512],
        #                                   initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
        #     b_conv5_2_2 = tf.get_variable('biases', [512],
        #                                   initializer=tf.constant_initializer(0))
        #     z_conv5_2_2 = tf.nn.conv2d(a_conv5_1_2, w_conv5_2_2, strides=[1, 1, 1, 1],
        #                                padding='SAME') + b_conv5_2_2
        #     a_conv5_2_2 = tf.nn.relu(z_conv5_2_2)
        #
        # with tf.variable_scope('conv5_3_2') as scope:
        #     w_conv5_3_2 = tf.get_variable('weights', [3, 3, 512, 512],
        #                                   initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
        #     b_conv5_3_2 = tf.get_variable('biases', [512],
        #                                   initializer=tf.constant_initializer(0))
        #     z_conv5_3_2 = tf.nn.conv2d(a_conv5_2_2, w_conv5_3_2, strides=[1, 1, 1, 1],
        #                                padding='SAME') + b_conv5_3_2
        #     a_conv5_3_2 = tf.nn.relu(z_conv5_3_2)
        #
        # with tf.variable_scope('conv5_3_2_down') as scope:
        #     w_conv5_3_2_down = tf.get_variable('weights', [1, 1, 512, 64],
        #                                        initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
        #     b_conv5_3_2_down = tf.get_variable('biases', [64],
        #                                        initializer=tf.constant_initializer(0))
        #     z_conv5_3_2_down = tf.nn.conv2d(a_conv5_3_2, w_conv5_3_2_down, strides=[1, 1, 1, 1],
        #                                     padding='SAME') + b_conv5_3_2_down
        #     a_conv5_3_2_down = tf.nn.relu(z_conv5_3_2_down)
        # with tf.variable_scope('conv5_3_2_down_deconv') as scope:
        #     w_conv5_3_2_down_deconv = tf.get_variable('weights', [32, 32, 64, 64],
        #                                               initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
        #     b_conv5_3_2_down_deconv = tf.get_variable('biases', [1],
        #                                               initializer=tf.constant_initializer(0))
        #     z_conv5_3_2_down_deconv = tf.nn.conv2d_transpose(a_conv5_3_2_down, w_conv5_3_2_down_deconv,
        #                                                      [self.batch_num, self.max_size[0], self.max_size[1], 64],
        #                                                      strides=[1, 16, 16, 1], padding='SAME',
        #                                                      name='z') + b_conv5_3_2_down_deconv
        #     a_conv5_3_2_down_deconv = tf.nn.sigmoid(z_conv5_3_2_down_deconv)


        # Add to store dicts
        self.outputs['conv1_1'] = a_conv1_1
        self.outputs['conv1_2'] = a_conv1_2
        self.outputs['pool1'] = pool1
        self.outputs['conv2_1'] = a_conv2_1
        self.outputs['conv2_2'] = a_conv2_2
        self.outputs['pool2'] = pool2
        self.outputs['conv3_1'] = a_conv3_1
        self.outputs['conv3_2'] = a_conv3_2
        self.outputs['conv3_3'] = a_conv3_3
        self.outputs['pool3'] = pool3
        self.outputs['conv4_1'] = a_conv4_1
        self.outputs['conv4_2'] = a_conv4_2
        self.outputs['conv4_3'] = a_conv4_3
        self.outputs['pool4'] = pool4
        self.outputs['conv5_1'] = a_conv5_1
        self.outputs['conv5_2'] = a_conv5_2
        self.outputs['conv5_3'] = a_conv5_3
        self.outputs['conv5_3_down'] = a_conv5_3_down


        self.outputs['conv5_1_va'] = a_conv5_1_va
        self.outputs['conv5_2_va'] = a_conv5_2_va

        self.outputs['conv5_3_va'] = a_conv5_3_va
        self.outputs['conv5_3_down_va'] = a_conv5_3_down_va
        self.outputs['conv5_3_down_deconv'] = a_conv5_3_down_deconv
        self.outputs['conv5_3_down_final'] = a_conv5_3_down_final
        for i in range(2, 33):

            self.outputs[eval("'conv5_1_va_%s'"%i)] = eval("a_conv5_1_va_%s"%i)
            self.outputs[eval("'conv5_2_va_%s'"%i)] = eval("a_conv5_2_va_%s"%i)

            self.outputs[eval("'conv5_3_va_%s'"%i)] = eval("a_conv5_3_va_%s"%i)
            self.outputs[eval("'conv5_3_down_va_%s'"%i)] = eval("a_conv5_3_down_va_%s"%i)
            self.outputs[eval("'conv5_3_down_deconv_%s'"%i)] = eval("a_conv5_3_down_deconv_%s"%i)
            self.outputs[eval("'conv5_3_down_final_%s'"%i)] = eval("a_conv5_3_down_final_%s"%i)

            # eval("self.outputs['conv5_3_down_va_%s'] = a_conv5_3_down_va_%s"%(i,i))
            # eval("self.outputs['conv5_3_down_deconv_%s'] = a_conv5_3_down_deconv_%s"%(i,i))
            # eval("self.outputs['conv5_3_down_final_%s'] = a_conv5_3_down_final_%s"%(i,i))
        self.outputs['final'] = final
        # self.outputs['conv3_1_2'] = a_conv3_1_2
        # self.outputs['conv3_2_2'] = a_conv3_2_2
        # self.outputs['conv3_3_2'] = a_conv3_3_2
        # self.outputs['conv4_1_2'] = a_conv4_1_2
        # self.outputs['conv4_2_2'] = a_conv4_2_2
        # self.outputs['conv4_3_2'] = a_conv4_3_2
        # self.outputs['conv5_1_2'] = a_conv5_1
        # self.outputs['conv5_2_2'] = a_conv5_2
        # self.outputs['conv5_3_2'] = a_conv5_3
        # self.outputs['conv5_3_2_down'] = a_conv5_3_2_down
        # self.outputs['conv5_3_2_down_deconv'] = a_conv5_3_2_down_deconv

        self.layers['conv1_1'] = {'weights': w_conv1_1, 'biases': b_conv1_1}
        self.layers['conv1_2'] = {'weights': w_conv1_2, 'biases': b_conv1_2}
        self.layers['conv2_1'] = {'weights': w_conv2_1, 'biases': b_conv2_1}
        self.layers['conv2_2'] = {'weights': w_conv2_2, 'biases': b_conv2_2}
        self.layers['conv3_1'] = {'weights': w_conv3_1, 'biases': b_conv3_1}
        self.layers['conv3_2'] = {'weights': w_conv3_2, 'biases': b_conv3_2}
        self.layers['conv3_3'] = {'weights': w_conv3_3, 'biases': b_conv3_3}
        self.layers['conv4_1'] = {'weights': w_conv4_1, 'biases': b_conv4_1}
        self.layers['conv4_2'] = {'weights': w_conv4_2, 'biases': b_conv4_2}
        self.layers['conv4_3'] = {'weights': w_conv4_3, 'biases': b_conv4_3}
        self.layers['conv5_1'] = {'weights': w_conv5_1, 'biases': b_conv5_1}
        self.layers['conv5_2'] = {'weights': w_conv5_2, 'biases': b_conv5_2}
        self.layers['conv5_3'] = {'weights': w_conv5_3, 'biases': b_conv5_3}


        self.layers['conv5_1_va'] = {'weights': w_conv5_1_va, 'biases': b_conv5_1_va}
        self.layers['conv5_2_va'] = {'weights': w_conv5_2_va, 'biases': b_conv5_2_va}

        self.layers['conv5_3_va'] = {'weights': w_conv5_3_va, 'biases': b_conv5_3_va}
        self.layers['conv5_3_down_va'] = {'weights': w_conv5_3_down_va, 'biases': b_conv5_3_down_va}
        self.layers['conv5_3_down_deconv'] = {'weights': w_conv5_3_down_deconv, 'biases': b_conv5_3_down_deconv}
        self.layers['conv5_3_down_final'] = {'weights': w_conv5_3_down_final, 'biases': b_conv5_3_down_final}
        for i in range(2, 33):
            self.layers[eval("'conv5_1_va_%s'"%i)] = eval("{'weights': w_conv5_1_va_%s, 'biases': b_conv5_1_va_%s}"%(i,i))
            self.layers[eval("'conv5_2_va_%s'"%i)] = eval("{'weights': w_conv5_2_va_%s, 'biases': b_conv5_2_va_%s}"%(i,i))

            self.layers[eval("'conv5_3_va_%s'"%i)] = eval("{'weights': w_conv5_3_va_%s, 'biases': b_conv5_3_va_%s}"%(i,i))
            self.layers[eval("'conv5_3_down_va_%s'"%i)] = eval("{'weights': w_conv5_3_down_va_%s, 'biases': b_conv5_3_down_va_%s}"%(i,i))
            self.layers[eval("'conv5_3_down_deconv_%s'"%i)] = {'weights': w_conv5_3_down_deconv, 'biases': b_conv5_3_down_deconv}
            self.layers[eval("'conv5_3_down_final_%s'"%i)] = {'weights': w_conv5_3_down, 'biases': b_conv5_3_down}
        # self.layers['conv3_1_2'] = {'weights': w_conv3_1_2, 'biases': b_conv3_1_2}
        # self.layers['conv3_2_2'] = {'weights': w_conv3_2_2, 'biases': b_conv3_2_2}
        # self.layers['conv3_3_2'] = {'weights': w_conv3_3_2, 'biases': b_conv3_3_2}
        # self.layers['conv4_1_2'] = {'weights': w_conv4_1_2, 'biases': b_conv4_1_2}
        # self.layers['conv4_2_2'] = {'weights': w_conv4_2_2, 'biases': b_conv4_2_2}
        # self.layers['conv4_3_2'] = {'weights': w_conv4_3_2, 'biases': b_conv4_3_2}
        # self.layers['conv5_1_2'] = {'weights': w_conv5_1_2, 'biases': b_conv5_1_2}
        # self.layers['conv5_2_2'] = {'weights': w_conv5_2_2, 'biases': b_conv5_2_2}
        # self.layers['conv5_3_2'] = {'weights': w_conv5_3_2, 'biases': b_conv5_3_2}
        # self.layers['conv5_3_2_down'] = {'weights': w_conv5_3_2_down, 'biases': b_conv5_3_2_down}
        # self.layers['conv5_3_2_down_deconv'] = {'weights': w_conv5_3_2_down_deconv, 'biases': b_conv5_3_2_down_deconv}


"""Baseline model"""


class Net(Network):
    def __init__(self, config):
        self.num_classes = config['num_classes']
        self.batch_num = config['batch_num']
        self.max_size = config['max_size']
        self.weight_decay = config['weight_decay']
        self.base_lr = config['base_lr']
        self.momentum = config['momentum']
        # img seg mask train test val
        self.img = tf.placeholder(tf.float32,
                                  [self.batch_num, self.max_size[0], self.max_size[1], 3])
        self.seg = tf.placeholder(tf.float32,
                                  [self.batch_num, self.max_size[0], self.max_size[1], 1])

        self.layers = {}
        self.outputs = {}
        self.set_up()  # checked

    def set_up(self):
        self.add_conv(self.img, self.num_classes)  # add convolution layer checked
        # self.add_deconv(bilinear=False)           #add deconvolution layer checked
        self.add_loss_op()  # add loss checked
        self.add_weight_decay()  # add weight decay checked
        self.add_train_op()  # add train op checked

    """Extract parameters from ckpt file to npy file"""

    # def extract(self, data_path, session, saver):
    #     saver.restore(session, data_path)
    #     scopes = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1',
    #               'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1',
    #               'conv5_2', 'conv5_3']
    #     data_dict = {}
    #     for scope in scopes:
    #         [w, b] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    #         data_dict[scope] = {'weights': w.eval(), 'biases': b.eval()}
    #     file_name = data_path[0:-5]
    #     np.save(file_name, data_dict)
    #     ipdb.set_trace()
    #     return file_name + '.npy'

    # """Add the deconv(upsampling) layer to get dense prediction"""
    # def add_deconv(self, bilinear=False):
    #     conv6 = self.get_output('conv_merge')
    #
    #     with tf.variable_scope('deconv') as scope:
    #         # Learn from scratch
    #         if not bilinear:
    #             w_deconv = tf.get_variable('weights', [8, 8, self.num_classes, self.num_classes],
    #                                        initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
    #         # Using fiexed bilinearing upsampling filter
    #         else:
    #             w_deconv = tf.get_variable('weights', trainable=True,
    #                                        initializer=bilinear_upsample_weights(16, self.num_classes))
    #
    #         b_deconv = tf.get_variable('biases', [self.num_classes],
    #                                    initializer=tf.constant_initializer(0))
    #         z_deconv = tf.nn.conv2d_transpose(conv6, w_deconv,
    #                                           [self.batch_num, self.max_size[0], self.max_size[1], self.num_classes],
    #                                           strides=[1, 4, 4, 1], padding='SAME', name='z') + b_deconv
    #         a_deconv = z_deconv
    #
    #     # Add to store dicts
    #     self.outputs['deconv'] = a_deconv
    #     self.layers['deconv'] = {'weights': w_deconv, 'biases': b_deconv}
    #
    # def bilinear_upsample_weights(factor, number_of_classes):
    #     """
    #     Create weights matrix for transposed convolution with bilinear filter
    #     initialization.
    #     """
    #     filter_size = get_kernel_size(factor)
    #
    #     weights = np.zeros((filter_size,
    #                         filter_size,
    #                         number_of_classes,
    #                         number_of_classes), dtype=np.float32)
    #
    #     upsample_kernel = upsample_filt(filter_size)
    #
    #     for i in xrange(number_of_classes):
    #         weights[:, :, i, i] = upsample_kernel
    #
    #     return weights

    """Add pixelwise softmax loss"""

    def add_loss_op(self):
        gt_reshape = tf.reshape(self.seg, [-1, self.num_classes])
        print(gt_reshape)

        pred1 = self.get_output('conv5_3_down_final')
        pred1_reshape = tf.reshape(pred1, [-1, 1])
        for i in range(2,33):
            exec("pred%s = self.get_output('conv5_3_down_final_%i')"%(i,i))
            exec("pred%s_reshape = tf.reshape(pred%s, [-1, 1])"%(i,i))

        pred_all1 = tf.zeros(pred1_reshape.shape)
        for i in range(32):
            pred_all1 += eval("pred%s_reshape"%(i+1))
        pred_ave1 = pred_all1 / 32
        # sum1 = tf.reduce_mean(pred_all1, axis=3, keep_dims=True)
        # sum1 = tf.reshape(sum1, [-1, 1])
        _pred_ave1 = tf.stop_gradient(pred_ave1)
        loss1 = 0

        for i in range(1, 33):
            # G1 = 'pred{}_reshape'.format(i)
            # G1 = exec("pred%s_reshape"%i)
            # exec("G1 = pred%s_reshape"%i)
            # exec("loss1 += tf.reduce_mean(tf.square(G1 - gt_reshape)) - 0.1 * tf.reduce_mean(tf.square(G1 - _pred_ave1))")
            loss1 = loss1 + eval("0.5 * tf.reduce_mean(tf.square(pred%s_reshape - gt_reshape)) - 0.1 * tf.reduce_mean(tf.square(pred%s_reshape - _pred_ave1))"%(i,i))
        self.loss1 = loss1

        # pred1 = self.get_output('conv5_3_down_deconv')
        # pred2 = self.get_output('conv5_3_2_down_deconv')
        # pred1_reshape = tf.reshape(pred1, [-1, 64])
        # pred2_reshape = tf.reshape(pred2, [-1, 64])
        # gt_reshape = tf.reshape(self.seg, [-1, self.num_classes])
        # sum1 = tf.reduce_mean(pred1, axis=3, keep_dims=True)
        # sum2 = tf.reduce_mean(pred2, axis=3, keep_dims=True)
        # sum1 = tf.reshape(sum1, [-1, 1])
        # sum2 = tf.reshape(sum2, [-1, 1])
        # _sum1 = tf.stop_gradient(sum1)
        # _sum2 = tf.stop_gradient(sum2)
        # print(gt_reshape)
        # loss1 = 0
        # loss2 = 0
        #
        # for i in range(64):
        #     G1 = tf.expand_dims(pred1_reshape[:, i], -1)
        #     G2 = tf.expand_dims(pred2_reshape[:, i], -1)
        #     loss1 += 0.5 * tf.reduce_mean(tf.square(G1 - gt_reshape)) - 0.01 * tf.reduce_mean(
        #         tf.square(G1 - _sum2)) - 0.1 * tf.reduce_mean(tf.square(G1 - _sum1))
        #     loss2 += 0.5 * tf.reduce_mean(tf.square(G2 - gt_reshape)) - 0.01 * tf.reduce_mean(
        #         tf.square(G2 - _sum1)) - 0.1 * tf.reduce_mean(tf.square(G2 - _sum2))
        # self.loss1 = loss1
        # self.loss2 = loss2

    """Add weight decay"""

    def add_weight_decay(self):
        for key in self.layers:
            w = self.layers[key]['weights']
            # self.loss += self.weight_decay * tf.nn.l2_loss(w)
            # self.loss1 += self.weight_decay * tf.nn.l2_loss(w)
            # self.loss2 += self.weight_decay * tf.nn.l2_loss(w)

    """Set up training optimization"""

    def add_train_op(self):
        self.train_op1 = tf.train.AdamOptimizer(self.base_lr).minimize(self.loss1)
        # self.train_op2 = tf.train.AdamOptimizer(self.base_lr).minimize(self.loss2)


class Net_test(Net):
    def __init__(self, config):
        Net.__init__(self, config)

    def set_up(self):
        self.add_conv(self.img, self.num_classes)
