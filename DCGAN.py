'''
    1.数据 data provider：<图像数据，随即向量>
    2.计算图构建：生成器Gnnerator，
                 判断器Discriminator,
                 将G D 合成DCGAN网络结构，然后计算loss
    3.训练过程
'''
import os
import datetime
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data  # 直接从tensorflow自带的手写数字数据库
mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True)  # 导入mnist，这里只用了train数据集 ,而且用不上类别
output_dir = './local_run'                                        # 输出文件夹
if not tf.gfile.Exists(output_dir):
    tf.gfile.MkDir(output_dir)

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # 尝试调用GPU
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

# 模型参数
def get_defalut_params():
    return tf.contrib.training.HParams(
        z_dim=100,                          # 随即向量长度，<向量，图像>，要重排成矩阵才可以反卷积
        init_conv_size=4,                   # 向量变成矩阵的大小？初始化卷积核大小
        g_channels=[128, 64, 32, 1],        # 生成器G，各个反卷积层的通道数目 mnist是单通道，所以最后是1
                                            # 其反卷积层大小，步长等也应该定义，为了简单后面统一
        d_channels=[32, 64, 128, 256],      # 判别器D，各个卷积层的通道数目 (这里步长明显是2)
        batch_size=128,
        learning_rate=0.002,
        beta1=0.5,                          # 损失函数中，有个betal,beta2
        img_size=32,                        # 生成图像大小， 这里是生成正方形 4->32 只是经过了43层
    )
hps = get_defalut_params()
# print(hps.z_dim, mnist.train.images.shape)

# 数据处理
class MnistData(object):
    '''mnist_train的图片大小28*28， 要变成img_size大小 才方便D判断'''
    '''卷积和反卷积都是2的倍数，所以28*28不方便'''
    def __init__(self, mnist_train, z_dim, img_size):
        self._data = mnist_train
        self._example_num = len(self._data)                 # 每个图片都要生成对应的向量
        self._z_data = np.random.standard_normal(           # 正态分布随机初始
            (self._example_num, z_dim))
        self._indicator = 0
        self._resize_mnist_img(img_size)                    # 28*28变成32*32
        self._random_shuffle()

    def _random_shuffle(self):
        p = np.random.permutation(self._example_num)
        self._z_data = self._z_data[p]
        self._data = self._data[p]

    def _resize_mnist_img(self, img_size):
        '''使用PIL对图片进行缩放  np矩阵不能当成图片直接缩放，所以
            1.numpy -> PIL img
            2.PIL img -> resize
            3.resize PIL img -> numpy
        '''
        data = np.asarray(self._data * 255, np.uint8)          # mnist数据集是归一化了的[0,1] uint8:0-255
        # [_example_num, 784] -> [_example_num, 28, 28]          向量变成图像
        data = data.reshape((self._example_num, 28, 28))
        new_data = []
        for i in range(self._example_num):                     # 图像缩放操作
            img = data[i]
            img = Image.fromarray(img)                         # numpy -> PIL img
            img = img.resize((img_size, img_size))               # PIL img放大成img_size
            img = np.asarray(img)                              # resize PIL img -> numpy
            img = img.reshape((img_size, img_size, 1))         # 添加一纬，作为通道，卷积的时候要用
            new_data.append(img)
        new_data = np.asarray(new_data, dtype=np.float32)      # 列表里若干矩阵，这里把它变成大矩阵
        new_data = new_data / 127.5 - 1                        # 归一到[-1,1]  tanh的结果是[-1,1]
        '''self._data : [num_example, img_size, img_size, 1]'''
        self._data = new_data                                  # 32*32 的图像

    def next_batch(self, batch_size):
        end_indicator = self._indicator + batch_size
        if end_indicator < self._example_num:
            self._random_shuffle()
            self._indicator = 0
            end_indicator = self._indicator + batch_size
        assert end_indicator < self._example_num                  # 加上batch_size后比样本大
        batch_data = self._data[self._indicator:end_indicator]
        batch_z = self._z_data[self._indicator:end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_z

mnist_dadta = MnistData(mnist.train.images, hps.z_dim, hps.img_size)
# batch_data, batch_z = mnist_dadta.next_batch(5)
# print(batch_z.shape, batch_data.shape)


# 计算图的构建 先构建G D， 然后再组合为DCGAN
'''生成器G都是反卷积操作(向量生成图片)   这里将4层封装'''
def conv2d_transposs(inputs, out_channel, name, training, with_bn_relu=True):
    with tf.variable_scope(name):
        conv2d_trans = tf.layers.conv2d_transpose(inputs, out_channel,
                                [5, 5], strides=(2, 2), padding='SAME')
        if with_bn_relu:
            bn = tf.layers.batch_normalization(conv2d_trans, training=training)  # GAN需要每层归一化，方便收敛
            return tf.nn.relu(bn)
        else:
            return conv2d_trans                 # 生成器G的最后一层是不要batch_normal,relu的，只要tanh
# GAN的 生成器G
class Generator(object):
    ''' chanels每层通道数，
        反卷积核大小和步长conv2d_transposs中固定了，
        初始化卷积核大小nit_conv_size
        生成图片img_size * img_size
    '''
    def __init__(self, chanels, init_conv_size):
        self._channels = chanels
        self._init_conv_size = init_conv_size
        self._reuse = False                    # 生成器G可能不只是使用一次，构建图之后可能会重用它，与session公用

    def __call__(self, inputs, training):      # 允许一个类的实例像函数一样被调用：x(a, b) 调用 x.__call__(a, b)
        inputs = tf.convert_to_tensor(inputs)  # 将inputs转成tensor, 因为计算图tensorflow中是用tensor
        with tf.variable_scope('generator', reuse=self._reuse):
            '''将向量和一个矩阵连接(类似feature map)，这样才能反卷积
                做法： 随机向量 -> 全连接层 -> 得到跟大的向量 self._chanels[0] * init_conv_sieze^2
                -> reshape -> 得到[init_conv_size, init_conv_size, channels[0]]
            '''
            with tf.variable_scope('inputs_conv'):
                # 输入长度 经过全连接层 到更大输出长度 重新变成矩阵，归一化，非线性变化，
                fc = tf.layers.dense(inputs, self._channels[0] * self._init_conv_size * self._init_conv_size)
                # 重新排列，这个-1是自动的意思，其实就是batch_size大小 cov0可以看做是个卷积输出
                conv0 = tf.reshape(fc, [-1, self._init_conv_size, self._init_conv_size, self._channels[0]])
                bn0 = tf.layers.batch_normalization(conv0, training=training)
                relu0 = tf.nn.relu(bn0)
            # self._channels[0]用了，只剩三个反卷积
            deconv_inputs = relu0
            for i in range(1, len(self._channels)):
                # 最后一层不做relu
                with_bn_relu = (i != len(self._channels) - 1)
                deconv_inputs = conv2d_transposs(deconv_inputs,
                                self._channels[i], 'deconv-%d' % i, training, with_bn_relu)
            # 得到最后一层，送入Tanh
            img_inpus = deconv_inputs
            with tf.variable_scope('generate_imgs'):
                # imgs 取值范围:[-1, 1]
                imgs = tf.tanh(img_inpus, name='imgs')
        self._reuse = True                                          # 以后遇到使用G的时候，可以重用
        '''定义全局变量，保存生成器G的所有参数  因为G D是分开训练的'''
        self.variables = tf.get_collection(                          # 获得generator下所有变量,后面训练的时候用
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        return imgs


# GAN的 判别器D
'''判别器D是卷积操作，这里是4次'''
def conv2d(inputs, out_channel, name, training):
    # 定义了个激活函数 x>0时，取x, x<0时，x * leak
    def leaky_relu(x, leak=0.2, name=''):
        return tf.maximum(x, x * leak, name=name)
    with tf.variable_scope(name):
        conv2d_output = tf.layers.conv2d(inputs, out_channel,
                                [5, 5], strides=(2, 2), padding='SAME')
        bn = tf.layers.batch_normalization(conv2d_output, training=training)
        return leaky_relu(bn, name='outputs')

class Discriminator(object):
    def __init__(self, channels):
        self._chanels = channels
        self._reuse = False

    def __call__(self, inputs, training):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)  # 之后就可以正常的卷积网络操作

        conv_inputs = inputs
        with tf.variable_scope('discriminator', reuse=self._reuse):
            for i in range(len(self._chanels)):
                conv_inputs = conv2d(conv_inputs, self._chanels[i], 'conv-%d' % i, training)
            # 展平最后一个卷积结果，然后全连接到类别上，即输出两个节点   两类[真实，不真实]
            fc_inputs = conv_inputs
            with tf.variable_scope('fc'):
                flatten = tf.layers.flatten(fc_inputs)
                logits = tf.layers.dense(flatten, 2, name='logits')     # 两类[真实，不真实]
        self._reuse = True
        self.variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')    # 获得discriminator下所有变量,后面训练的时候用
        return logits


# DCGAN 网络结构  计算图
class DCGAN(object):
    def __init__(self, hps):
        g_channels = hps.g_channels
        d_channels = hps.d_channels
        self._batch_size = hps.batch_size
        self._init_conv_size = hps.init_conv_size
        self._z_dim = hps.z_dim
        self._img_size = hps.img_size

        self._generator = Generator(g_channels, self._init_conv_size)
        self._discriminator = Discriminator(d_channels)

    '''构建计算图，也是结构，用于数据填充，计算'''
    def build(self):
        '''向量用来训练生成G'''
        self._z_placeholder = tf.placeholder(tf.float32, (self._batch_size, self._z_dim))                           # 每行向量生成图片
        '''真实图像用来训练判别器D'''
        self._img_placeholder = tf.placeholder(tf.float32, (self._batch_size, self._img_size, self._img_size, 1))   # [图片序号，长，宽，通道]
        generated_imgs = self._generator(self._z_placeholder, training=True)                # G生成的图像， 假图像
        fake_img_logits = self._discriminator(generated_imgs, training=True)                # 假图像判断结果
        real_img_logits = self._discriminator(self._img_placeholder, training=True)         # 真图像判断结果
        '''定义损失函数，两个，分开训练
            判别器，越真越好
            生成器，尽量避开判别器，让D判断为真
        '''
        # 生成器损失函数
        loss_on_fake_to_real = tf.reduce_mean(                          # 真的用1表示，计算两者均值
            tf.nn.sparse_softmax_cross_entropy_with_logits(             # 假的图片，判断为真
                labels = tf.ones([self._batch_size], dtype=tf.int64),
                logits = fake_img_logits))
        # 判别器损失函数
        loss_on_fake_to_fake = tf.reduce_mean(                         # 假的判断为假的
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.zeros([self._batch_size], dtype=tf.int64),
                logits=fake_img_logits))
        loss_on_real_to_real = tf.reduce_mean(                         # 真的判断为真的
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.ones([self._batch_size], dtype=tf.int64),
                logits=real_img_logits))
        # 总的损失函数 collection类似字典实现的 keys:value 存取后，算总的加上，这样也方面查询
        tf.add_to_collection('g_losses', loss_on_fake_to_real)
        tf.add_to_collection('d_losses', loss_on_fake_to_fake)
        tf.add_to_collection('d_losses', loss_on_real_to_real)

        loss = {
            'g': tf.add_n(tf.get_collection('g_losses'), name='total_g_loss'),
            'd': tf.add_n(tf.get_collection('d_losses'), name='total_d_loss')
        }
        return self._z_placeholder, self._img_placeholder, generated_imgs, loss

    '''构建训练方式，在build调用后，必须回调'''
    def build_train_op(self, losses, learning_rate, beta1):
        # 分别训练生成器G和判别器D
        g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        # 将上面的优化方式(optimizer)用于对应损失函数上的变量中
        g_opt_op = g_opt.minimize(losses['g'], var_list=self._generator.variables)
        d_opt_op = d_opt.minimize(losses['d'], var_list=self._discriminator.variables)
        # G D 是交差训练的 tf高端操作control_dependencies，先执行参数列表，再执行后面
        '''这里直接返回执行模块名字，程序效果就是交替{[g_opt_op, d_opt_op]，[g_opt_op, d_opt_op]}'''
        with tf.control_dependencies([g_opt_op, d_opt_op]):
            return tf.no_op(name='train')

dcgan = DCGAN(hps)
z_placeholder, img_placeholder, generated_imgs, losses = dcgan.build()
train_op = dcgan.build_train_op(losses, hps.learning_rate, hps.beta1)

# 训练流程
'''打印GAN的一些输出，这里是图片，可以batch_size组合一起打印 例如128=8*16'''
def combine_imgs(batch_imgs, img_size, rows=8, clos=16):
    # batch_imgs: [batch_size, img_size, img_size, 1]
    result_big = []
    for i in range(rows):
        row_imgs = []
        for j in range(clos):
            img = batch_imgs[clos * i + j]                      # 取的是第一维，batch_size, img:[img_size,img_size, 1]
            img = img.reshape((img_size, img_size))
            img = (img + 1) * 127.5                             # 反归一化，这样用来输出图片
            row_imgs.append(img)
        row_imgs = np.hstack(row_imgs)                           # 合并, 按行合并
        result_big.append(row_imgs)                             # 一行一行的加入
    # result_big_img:[8*32, 16*32]，其中每32*32是个图片
    result_big_img = np.vstack(result_big)  # 按列合并
    result_big_img = np.asarray(result_big_img, np.uint8)       # 转换数据格式 0-255
    result_big_img = Image.fromarray(result_big_img)            # 矩阵变成图像
    return result_big_img

'''训练流程'''
init_op = tf.global_variables_initializer()
train_steps = 2000
starttime = datetime.datetime.now()
with tf.Session() as sess:
    sess.run(init_op)
    for step in range(train_steps):
        batch_imgs, batch_z = mnist_dadta.next_batch(hps.batch_size)
        fetches = [train_op, losses['g'], losses['d']]         # 每调用一次优化，就打印损失
        should_sample = (step + 1) % 50 == 0
        if should_sample:
            fetches += [generated_imgs]                        # 生成图片加入
        output_value = sess.run(fetches,
                                feed_dict={
                                    z_placeholder: batch_z,
                                    img_placeholder: batch_imgs
                                })
        _, g_loss_vla, d_loss_val = output_value[0:3]
        tf.logging.info('step: %4d, g_loss: %4.3f, d_loss:%4.3f' % (step, g_loss_vla, d_loss_val))

        if should_sample:                                       # 输出图像
            gen_imgs_val = output_value[3]
            gen_imgs_path = os.path.join(output_dir, '%05d-gen.jpg' % (step + 1))        # 生成的图像
            ground_img_path = os.path.join(output_dir, '%05d-ground.jpg' % (step + 1))   # 真是图像
            # 小图像拼成大图像
            gen_img = combine_imgs(gen_imgs_val, hps.img_size)
            ground_img = combine_imgs(batch_imgs, hps.img_size)
            # 保存，输出图像
            gen_img.save(gen_imgs_path)
            ground_img.save(ground_img_path)

    endtime = datetime.datetime.now()
    seconds = (endtime - starttime).seconds
    start = starttime.strftime('%Y-%m-%d %H:%M')
    # 100 秒
    # 分钟
    minutes = seconds // 60
    second = seconds % 60
    print((endtime - starttime))
    timeStr = str(minutes) + '分钟' + str(second) + "秒"
    print("程序从 " + start + ' 开始运行,运行时间为：' + timeStr)











