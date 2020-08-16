import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import cv2
import glob
import tensorflow as tf
import sys
import time
from tqdm import tqdm
from becnn import becnn

if not os.path.exists('results_416'):  # output directory
    os.mkdir('results_416')

def normalize(images):
    return np.array([image/65535.0 for image in images])
    

def downscale(images):
    downs = [[[[0 for p in range(3)] for k in range(1024)] for j in range(436)] for i in range(len(images))]
    for ii in range(len(images)):
        for j in range(len(images[ii])):
            for k in range(len(images[ii][j])):
                for p in range(len(images[ii][j][k])):
                    tmp = bin(images[ii][j][k][p])
                    tmp_quan = tmp[:-12]+'000000000000'
                    downs[ii][j][k][p] = int (tmp_quan,2)
    downs_np = np.array(downs, dtype=np.uint16)
    return downs_np
    


x = tf.placeholder(tf.float32, [None,436, 1024, 3])  # height width 436 1024
downscaled = tf.placeholder(tf.float32, [None,436, 1024, 3])
is_training = tf.placeholder(tf.bool, [])

model = becnn(x, downscaled, is_training, 1)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()
saver.restore(sess, './latest')  # model name

pics = glob.glob('../test.png') # input images directory

tt=0
for i in tqdm(range(len(pics))):
    x_t1 = cv2.imread(pics[i],3)
    x_t1n = x_t1[np.newaxis,:,:,:]
    
    low_bit = downscale(x_t1n)
    cv2.imwrite('tmp.png',low_bit[0])

    starttime=time.time()
    x_t1 = cv2.imread('tmp.png',3)
    x_t1n = x_t1[np.newaxis,:,:,:]
    low_bit_float = normalize(x_t1n)

    fake = sess.run(model.imitation,
        feed_dict={x: low_bit_float, downscaled: low_bit_float, is_training: False})

    
    full_name = pics[i].split('/')[-1]
    pure_name = full_name.split('.')[0]
    clipped = np.clip(fake[0],0,1)
    im = np.uint16(clipped*65535.0)
    
    cv2.imwrite('results_416/'+pure_name+'_becnn_416.png',im)  #output

    print time.time()-starttime
