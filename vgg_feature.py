# -*- coding:utf-8 -*-

"""
@Brief
VGG2：Compute perceptual loss
source：Crowd Counting via Adversarial Cross-Scale Consistency Pursuit
        https://pan.baidu.com/s/1mjPpKqG
VGG16: http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz

@Description
vgg_2 model is a pre-trained VGG-16 model at layer relu2_2
"""
import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

class VGG2:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, None, None, 3])
        self.net = None
        self.end_points = None
        self.vgg16_path_name = "./vgg_16.ckpt"
        self.vgg2_name = "vgg_2.ckpt"
        self.vgg2_path = "./vgg2_model/"
        self.sess = tf.Session()
        self.vgg_2()

    def vgg_2(self, scope='vgg_16'):
        """
        The feature extraction model vgg_2
        :param scope: 
        """
        with tf.variable_scope(scope, 'vgg_16', [self.x]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                outputs_collections=end_points_collection):
                net = slim.repeat(self.x, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                self.net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                self.end_points = slim.utils.convert_collection_to_dict(end_points_collection)

    def vgg_2_2(self, scope='vgg_16'):
        """
        The feature extraction model vgg_2
        :param scope: 
        """
        with tf.variable_scope(scope, 'vgg_16', [self.x], reuse=True) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                outputs_collections=end_points_collection):
                net = slim.repeat(self.x, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                self.net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                self.end_points = slim.utils.convert_collection_to_dict(end_points_collection)

    def vgg_2_save(self):
        """
        Build VGG2 model with VGG-16 parameters
        """
        with tf.Session() as sess:
            # Initialize VGG2 model
            saver = tf.train.Saver()
            saver.restore(sess, self.vgg16_path_name)
            # Save VGG2 model
            saver.save(sess, self.vgg2_path + self.vgg2_name)
            # Close session
            sess.close()

    def vgg_2_load(self):
        """
        Load vgg2 model
        """
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.vgg2_path)
        saver.restore(self.sess, ckpt.model_checkpoint_path)


    def run(self, data):
        """
        Utilize VGG2 to extract image features
        :param: data: Density estimation map with perceptual feature to be extracted 
        :return: image feature
        """
        feature = self.sess.run(self.net, feed_dict={self.x: data})
        return feature


class VGGFTest(tf.test.TestCase):
    """
    Testing the VGG-2 model
    """

    def testBuild(self):
        batch_size = 1
        height, width = 240, 240
        with self.test_session():
            inputs = tf.random_uniform((batch_size, height, width, 3))
            vgg2 = VGG2()
            vgg2.x = inputs
            vgg2.vgg_2()

            expected_names = ['vgg_16/conv1/conv1_1/weights',
                              'vgg_16/conv1/conv1_1/biases',
                              'vgg_16/conv1/conv1_2/weights',
                              'vgg_16/conv1/conv1_2/biases',
                              'vgg_16/conv2/conv2_1/weights',
                              'vgg_16/conv2/conv2_1/biases',
                              'vgg_16/conv2/conv2_2/weights',
                              'vgg_16/conv2/conv2_2/biases',
                              ]
            model_variables = [v.op.name for v in slim.get_model_variables()]
            self.assertSetEqual(set(model_variables), set(expected_names))


if __name__ == "__main__":
    
    # TF model testing
    # Simulated test data
    batch = 4
    h, w = 240, 240
    images = np.ones((batch, h, w, 3))

    # Beginning of the test 
    vgg = VGG2()
    vgg.vgg_2_save()
    vgg.vgg_2_load()

    im_f = vgg.run(data=images)
