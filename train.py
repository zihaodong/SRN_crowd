"""
Train script that does full training of SRN model. It saves the model every epoch.

Before training make sure of the following:

1) The global constants are set i.e. NUM_TRAIN_IMGS, NUM_VAL_IMGS, NUM_TEST_IMGS.
2) The images for training, validation and testing should have proper heirarchy
   and proper file names. Details about the heirarchy and file name convention are
   provided in the README.

Command: python train.py --log_dir <log_dir_path> --num_epochs <num_of_epochs> --learning_rate <learning_rate> --session_id <session_id> 

Thanks to the contribution of Aditya Vora
"""
import tensorflow as tf
import src.srn as srn
import os
import numpy as np
import matplotlib.image as mpimg
import scipy.io as sio
import time
import argparse
import sys
from vgg_feature import VGG2
from glob import glob
from src.utils_data import *
from tensorflow.python.ops import array_ops


def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    """Compute focal loss for predictions.

    Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.

    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    # sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    sigmoid_p = prediction_tensor
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    
    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
    
    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)


def load(sess, saver, checkpoint_dir):
    print(" [*] Reading checkpoints...")

    model_dir = "%s_%s_%s" % ("ST_data", args.batch_size, 240) #batch_size=1
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False

def save(sess, checkpoint_dir, step,data_set_name,batch_size,output_size,saver):
    model_name = "mp_srn.model"
    model_dir = "%s_%s_%s" % (data_set_name, batch_size, output_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

def main(args):
    """
    Main function to execute the training.
    Performs training, validation after each epoch and testing after full epoch training.
    :param args: input command line arguments which will set the learning rate, number of epochs, data root etc.
    :return: None
    """
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.80)
    vgg2 = VGG2()
    vgg2.vgg_2_load()

    # Image and Density map Input
    real_data = tf.placeholder(tf.float32, shape=[args.batch_size, 240, 240,6])
    image_place_holder = real_data[:, :, :, :3]
    d_map_place_holder = real_data[:, :, :, 3:6]

    # Point map Input
    count_data = tf.placeholder(tf.float32, shape=[args.batch_size, 240, 240,3])

    # Intialization of loss function 
    euc_loss = 0
    loss_c = 0
    p_loss = 0
    seg_loss = 0
 
    n_levels = 3
    all_output = 0

    # Build the architecture of SRN
    d_map_est,dddd = srn.build(image_place_holder)

    # Build VGG2 for calculating the perceived loss  
    vgg2.x = d_map_place_holder
    vgg2.vgg_2_2()
    f_real_mp = vgg2.net


    ## Calculation of loss function at n_levels scales
    for i in range(n_levels):

        # 1. Euclidean loss for density map
        _, hi, wi, _ = d_map_est[i].get_shape().as_list()
        gt_i = tf.image.resize_images(d_map_place_holder, [hi, wi], method=0)
        euc_loss += tf.reduce_mean((gt_i - d_map_est[i]) ** 2)

        # 2. Focal loss for point segmentation
        _, hd, wd, _ = dddd[i].get_shape().as_list()
        gt_d = tf.image.resize_images(count_data, [hd, wd], method=0)
        seg_loss += focal_loss(dddd[i],gt_d)

        # 3. Counting loss for crowd count
        gt_i_c = tf.reduce_sum(gt_i,1)
        d_map_est_c = tf.reduce_sum(d_map_est[i],1)
        loss_c += tf.reduce_mean(((d_map_est_c - gt_i_c)/(gt_i_c + 1)) ** 2)
        
        # 4. Perception loss based on VGG-2
        vgg2.x = d_map_est[i]
        vgg2.vgg_2_2()
        f_fake_mp = vgg2.net
        _, h2, w2, _ = f_fake_mp.get_shape().as_list()
        gt_2 = tf.image.resize_images(f_real_mp, [h2, w2], method=0)
        p_loss += tf.reduce_mean(tf.abs(gt_2 - f_fake_mp) * tf.abs(gt_2 - f_fake_mp))

    # Final weighted loss function
    all_loss = euc_loss + 0.1 * seg_loss + 0.1 * loss_c + 0.1 * p_loss

    loss_summary = tf.summary.scalar('all_loss', all_loss)
    g_l_fake_sum = tf.summary.image("prediction", d_map_est[-1])
    dddd_l_fake_sum = tf.summary.image("point_segmentation", dddd[-1])

    d_s_op = tf.train.AdamOptimizer(args.learning_rate, beta1=args.beta1,beta2=args.beta2).minimize(all_loss)
    saver = tf.train.Saver()


    with tf.Session(config=tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)) as sess:
       
        # Initial the variable and create the new seesion
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # Merge profiles and output Graph structure to log files 
        merged_summary_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./logs", sess.graph)

        # If there is a breakpoint, the model will be loaded
        start_time = time.time()
        if load(sess,saver,args.checkpoint_dir):
             print(" [*] Load SUCCESS")
        else:
             print(" [!] Load failed...")
        counter = 1


        for epoch in range(args.num_epochs):
            # Get the training image list 
            data = glob('{}/*.jpg'.format(args.train_im_dir))
            np.random.shuffle(data)

            # Configure: Maximum number of training samples
            batch_idx_set = min(len(data), args.train_size)
            batch_idx_set /= args.batch_size  #batch_size = 1
            batch_idx_set = int(np.floor(batch_idx_set))

            # Start model training for this batch
            for idx in range(0, batch_idx_set):

                # Get the training data for this batch
                batch_files = data[idx * args.batch_size: (idx + 1) * args.batch_size]
                ALL = [load_data(batch_file, args) for batch_file in batch_files]
                batch = [ALL[0][0],ALL[1][0],ALL[2][0],ALL[3][0]]
                count_batch = [ALL[0][1],ALL[1][1],ALL[2][1],ALL[3][1]]

                # convert to numpy array
                batch_images = np.array(batch).astype(np.float32)
                count_images = np.array(count_batch).astype(np.float32)

                _ = sess.run([d_s_op], feed_dict={real_data: batch_images,count_data: count_images})
                counter += 1 # global iteration number

                # save profiles and prediction results 
                if np.mod(counter, 100) == 0:
                    summary_str = sess.run(merged_summary_op, feed_dict={real_data: batch_images,count_data: count_images})
                    writer.add_summary(summary_str, counter)

                    f_l = d_map_est[i].eval({real_data: batch_images,count_data: count_images})

                    r_sum = sum(sum(batch[0][:, :, 3]))
                    f_l_sum = sum(sum(sum(f_l[0]))) / 3

                    print("\n******************************************************************")
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, real: %.4f, l_fake: %.4f"
                            % (epoch, idx, batch_idx_set, time.time() - start_time, r_sum, f_l_sum))
                    print("******************************************************************\n")

                    im_path = "sample/"
                    im_name = "fake_large_" + str(epoch) + ".jpg"
                    cv2.imwrite(im_path + im_name, f_l[0])

                # Print training process information for each step
                if np.mod(counter, 10) == 0:
                    # Capture the loss of SRN model
                    err_euc_loss = all_loss.eval({real_data: batch_images,count_data: count_images})
                    # Print the training information
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f"
                           % (epoch, idx, batch_idx_set, time.time() - start_time, err_euc_loss))

                # The model is saved once for each training batch
                if np.mod(counter, 10) == 0:
                    save(sess, args.checkpoint_dir, counter, "ST_data",args.batch_size,240,saver)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default = './logs', type=str)
    parser.add_argument('--num_epochs', default = 1397, type=int)
    parser.add_argument('--learning_rate', default = 0.00005, type=float)
    parser.add_argument('--session_id', default = 2, type=int)
    parser.add_argument('--train_im_dir', dest='train_im_dir', default='./data/train_im/', help='training image path')
    parser.add_argument('--train_gt_dir', dest='train_gt_dir', default='./data/train_gt/', help='training density map path')
    parser.add_argument('--count_dir', dest='count_dir', default='./data/train_count_gt/', help='training point map path')    
    parser.add_argument('--test_im_dir', dest='test_im_dir', default='./data/test_im/', help='testing image path')
    parser.add_argument('--test_gt_dir', dest='test_gt_dir', default='./data/test_gt/', help='testing densit map path')

    parser.add_argument('--beta1', dest='beta1', type=float, default=0.9, help='factor of momentum')
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999, help='RMSProp factor')
    parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='Maximum number of images used for training')
    parser.add_argument('--load_size', dest='load_size', type=int, default=720, help='input image size')
    parser.add_argument('--fine_size', dest='fine_size', type=int, default=240, help='crop size')
    parser.add_argument('--test_dir', dest='test_dir', default='test/', help='test sample are saved here')
    parser.add_argument('--sample_dir', dest='sample_dir', default='sample/', help='sample are saved here')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='checkpoints/', help='trained models are saved here')

    args = parser.parse_args()
    main(args)
