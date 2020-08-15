import numpy as np
import os
import cv2
import glob
import tensorflow as tf
import sys
from tqdm import tqdm
from becnn import becnn

if not os.path.exists('results_816'):
    os.mkdir('results_816')

def normalize(images):
    return np.array([image/65535.0 for image in images])
    

def downscale(images):
    #print images
    downs = [[[[0 for p in range(3)] for k in range(1024)] for j in range(436)] for i in range(len(images))]
    for ii in range(len(images)):
        for j in range(len(images[ii])):
            for k in range(len(images[ii][j])):
                for p in range(len(images[ii][j][k])):
                    tmp = bin(images[ii][j][k][p])
                    tmp_quan = tmp[:-8]+'00000000'
                    downs[ii][j][k][p] = int (tmp_quan,2)
    downs_np = np.array(downs, dtype=np.uint16)
    return downs_np
    

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement=True) 
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True 


x = tf.placeholder(tf.float32, [None,436, 1024, 3])
downscaled = tf.placeholder(tf.float32, [None,436, 1024, 3])
is_training = tf.placeholder(tf.bool, [])

model = becnn(x, downscaled, is_training, 1)
sess = tf.Session(config=config)
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()
saver.restore(sess, './latest')

#pic = './677.png'
pics = glob.glob('../Source50/*')

    
for i in tqdm(range(len(pics))):#
#for i in range(1):#len(pics)):#
    x_t1 = cv2.imread(pics[i],3)
    #x_t1 = cv2.imread(pic,3)
    x_t1n = x_t1[np.newaxis,:,:,:]
    raw = normalize(x_t1n)
    #print raw
    low_bit = downscale(x_t1n)
    low_bit_float = normalize(low_bit)

    fake = sess.run(model.imitation,
        feed_dict={x: raw, downscaled: low_bit_float, is_training: False})

    
    full_name = pics[i].split('/')[-1]
    #full_name = pic.split('/')[-1]
    pure_name = full_name.split('.')[0]
    clipped = np.clip(fake[0],0,1)
    im = np.uint16(clipped*65535.0)
    imc = np.clip(im,0,65535)
    
    #cv2.imwrite('result_dsp816/'+pure_name+'_dsp_816.png',imc)
    cv2.imwrite('./results_816/'+pure_name+'_becnn_816.png',imc)
        
        
