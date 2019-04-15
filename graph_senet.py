import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import flatten
import numpy as np
slim = tf.contrib.slim

weight_decay = 0.0005
momentum = 0.9

init_learning_rate = 0.001
cardinality = 4# how many split ?
blocks = 8 # res_block ! (split + transition)
depth = 64 # out channel

"""
So, the total number of layers is (3*blokcs)*residual_layer_num + 2
because, blocks = split(conv 2) + transition(conv 1) = 3 layer
and, first conv layer 1, last dense layer 1
thus, total number of layers = (3*blocks)*residual_layer_num + 2
"""


image_size = 128
img_channels = 3
point_number = 136
reduction_ratio = 4

batch_size = 1
iteration = 60000//batch_size
# 128 * 391 ~ 50,000
test_iteration = 10
total_epochs = 100


def conv_layer(input, filter, kernel, stride):
    net = slim.conv2d(input, filter, kernel)
    if stride > 1:
        net = tf.nn.max_pool(net, ksize=[1, stride * 2 - 1, stride * 2 - 1, 1],
                             strides=[1, stride, stride, 1], padding="SAME")
    return net


def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')


def Average_pooling(x, pool_size=[3, 3], stride=2, padding='SAME'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Relu(x):
    return tf.nn.relu(x)

def Sigmoid(x):
    return tf.nn.sigmoid(x)

def Concatenation(layers):
    return tf.concat(layers, axis=3)


def Fully_connected(x, units):
    return tf.layers.dense(inputs=x, use_bias=False, units=units)


class SE_ResNeXt():
    def __init__(self, x, is_training, keep_prob):
        self.model = self.Build_SEnet(x, keep_prob, is_training=is_training)

    def first_layer(self, x):
        x = conv_layer(x, filter=32, kernel=[3, 3], stride=2)
        return x

    def transform_layer(self, x, stride):
        x = conv_layer(x, filter=depth, kernel=[1, 1], stride=1)
        x = conv_layer(x, filter=depth, kernel=[3, 3], stride=stride)
        return x

    def transition_layer(self, x, out_dim):
        x = conv_layer(x, filter=out_dim, kernel=[1, 1], stride=1)
        return x

    def split_layer(self, input_x, stride):
        layers_split = list()
        for i in range(cardinality):
            splits = self.transform_layer(input_x, stride=stride)
            layers_split.append(splits)

        return Concatenation(layers_split)

    def squeeze_excitation_layer(self, input_x, out_dim, ratio):

        squeeze = Global_Average_Pooling(input_x)

        excitation = Fully_connected(squeeze, units=out_dim / ratio)
        excitation = Relu(excitation)
        excitation = Fully_connected(excitation, units=out_dim)
        excitation = Sigmoid(excitation)
        excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
        scale = input_x * excitation

        return scale

    def residual_layer(self, input_x, out_dim, res_block=blocks):
        # split + transform(bottleneck) + transition + merge
        # input_dim = input_x.get_shape().as_list()[-1]

        for i in range(res_block):
            input_dim = int(np.shape(input_x)[-1])

            if input_dim * 2 == out_dim:
                flag = True
                stride = 2
                channel = input_dim // 2
            else:
                flag = False
                stride = 1

            x = self.split_layer(input_x, stride=stride)
            x = self.transition_layer(x, out_dim=out_dim)
            x = self.squeeze_excitation_layer(x, out_dim=out_dim, ratio=reduction_ratio)

            if flag is True:
                pad_input_x = Average_pooling(input_x)
                pad_input_x = tf.pad(pad_input_x,
                                     [[0, 0], [0, 0], [0, 0], [channel, channel]])  # [?, height, width, channel]
            else:
                pad_input_x = input_x

            input_x = Relu(x + pad_input_x)

        return input_x

    def Build_SEnet(self, input_x, keep_prob, is_training=True,
                    weight_decay=0.0001,
                    batch_norm_decay=0.997,
                    batch_norm_epsilon=1e-5,
                    batch_norm_scale=True):

        batch_norm_params = {
            'is_training': is_training,
            'decay': batch_norm_decay,
            'epsilon': batch_norm_epsilon,
            'scale': batch_norm_scale,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
        }

        endpoints = {}

        with slim.arg_scope(
                [slim.conv2d],
                weights_regularizer=slim.l2_regularizer(weight_decay),
                weights_initializer=slim.variance_scaling_initializer(),
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                    input_x = self.first_layer(input_x)

                    x = self.residual_layer(input_x, out_dim=64)
                    print(x)
                    x = self.residual_layer(x, out_dim=128)
                    print(x)
                    x = self.residual_layer(x, out_dim=256)
                    print(x)
                    x = self.residual_layer(x, out_dim=512)
                    print(x)

                    x = Global_Average_Pooling(x)
                    x = flatten(x)
                    # Flatten tensor into a batch of vectors

                    # Dense layer 1, a fully connected layer.
                    dense1 = tf.layers.dense(
                        inputs=x,
                        units=1024,
                        activation=tf.nn.relu,
                        use_bias=True)

                    dense1 = tf.nn.dropout(dense1, keep_prob=keep_prob)

                    # Dense layer 2, also known as the output layer.
                    logits = tf.layers.dense(
                        inputs=dense1,
                        units=136,
                        activation=None,
                        use_bias=True,
                        name="logits")
                    return logits


def get_one_batch_data(batch_size, type):

    file_list = []
    if type == 0:
        file_name = "data/300W-LP/train.tfrecord"
    else:
        file_name = "data/300W-LP/test.tfrecord"

    file_list += tf.gfile.Glob(file_name)

    reader    = tf.TFRecordReader()

    filename_queue = tf.train.string_input_producer(
        file_list, num_epochs=None, shuffle=True
    )
    _, serialized_example = reader.read(filename_queue)

    #TODO train
    if type == 0:
        batch       = tf.train.shuffle_batch([serialized_example],
                                         batch_size,
                                         capacity=batch_size,
                                         min_after_dequeue=batch_size//2)
    else:
        batch       = tf.train.batch([serialized_example],
                                         batch_size,
                                         capacity=batch_size)

    features    = tf.parse_example(batch, features ={
        'image': tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([136], tf.int64),
    })

    images = features["image"]
    labels = features["label"]

    images_batch = tf.decode_raw(images, tf.uint8)

    #TODO 对图像数据进行reshape,按照打包的数据大小
    images_batch = tf.cast(tf.reshape(images_batch,
                                      [batch_size,
                                       image_size,
                                       image_size,
                                       3]), tf.float32)


    ##如果需要转灰度图, 由于数据打包为BGR数据，所有这里进行调整。
    labels = tf.cast(labels, tf.float32)
    #TODO 对finger map数据进行resize到输入graph尺寸

    return images_batch, labels


# image_size = 32, img_channels = 3, class_num = 10 in cifar10
x = tf.placeholder(tf.float32, shape=[1, image_size, image_size, img_channels])
label = tf.placeholder(tf.float32, shape=[1, point_number])

x = tf.transpose(x, [0, 3, 1, 2])
x = tf.transpose(x, [0, 2, 3, 1])

logits = SE_ResNeXt(x, is_training=False, keep_prob=1.0).model

cost = tf.losses.mean_squared_error(
    labels=label, predictions=logits)

reg_set = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
l2_loss = tf.add_n(reg_set)

# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# with tf.control_dependencies(update_ops):

global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(0.001, global_step,
                                decay_steps=1000,
                                decay_rate=0.95,
                                staircase=True)
# ##更新 BN
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):
    train = tf.train.AdamOptimizer(lr).minimize(l2_loss + cost, global_step)

saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=sess, coord=coord)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    ckpt = tf.train.get_checkpoint_state('model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    output_graph_def = tf.graph_util. \
        convert_variables_to_constants(sess,
                                       sess.graph.as_graph_def(),
                                       ['logits/BiasAdd'])

    with tf.gfile.FastGFile('saved_model/landmark_{}.pb'.format(ckpt.model_checkpoint_path.replace("/", "-")), 'wb') as f:
                    f.write(output_graph_def.SerializeToString())