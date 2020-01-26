import numpy as np
import tensorflow as tf
import cv2
import tqdm
from jGAN import Network

import time
from sklearn.neighbors import NearestNeighbors

LEARNING_RATE = 1e-4
BATCH_SIZE = 50

#berlin
# dataset = '/media/data2/anubha/zeroshotImgRetrieval/berlin/Dataset' 
# PRETRAIN_PATH = './pretrain_berlin/'

#sketchy
dataset = '/media/data2/anubha/zeroshotImgRetrieval/sketchy/Dataset' 
PRETRAIN_PATH = './pretrain_sketchy/'

def train():
	with tf.device('/device:GPU:1'):
	    x = tf.placeholder(tf.float32, [None, 2048])
	    y = tf.placeholder(tf.float32, [None, 2048])
	    is_training = tf.placeholder(tf.bool, [])

	    model = Network(x, y, is_training, batch_size=BATCH_SIZE)
	    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
	    global_step = tf.Variable(0, name='global_step', trainable=False)
	    epoch = tf.Variable(0, name='epoch', trainable=False)

	    opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
	    g_train_op = opt.minimize(model.g_loss, global_step=global_step, var_list=model.gan_variables)
	    d_train_op = opt.minimize(model.d_loss, global_step=global_step, var_list=model.d_variables)
	    gan_train_op = opt.minimize(model.gan_loss, global_step=global_step, var_list=model.gan_variables)

	    init_op = tf.global_variables_initializer()
	    sess.run(init_op)

	    print('loading dataset')
		#test data
	    t1_data = np.load(dataset+'/testData.npy')
	    test_img_label = np.load(dataset+'/testLabel.npy')
	    test_label_sketch = np.load(dataset+'/AlltestAttribute_label.npy')
	    t2_data = np.load(dataset+'/AlltestAttribute.npy')

	    print("t1_data",t1_data.shape)
	    print("t2_data",t2_data.shape)

	    #train data
	    fs = np.load(dataset+'/trainAttribute.npy')
	    fx = np.load(dataset+'/trainData.npy')
	    N = len(fs)
	    print('dataset loaded')
	    print(N)
	    print("fs",fs.shape)
	    print("fx",fx.shape)

	    step_num = int(N / BATCH_SIZE)
	    idx = [i for i in range(N)]

	    e = 0
	    while(e<=26):
	        sess.run(tf.assign(epoch, tf.add(epoch, 1)))
	        print('epoch: {}'.format(sess.run(epoch)))
	        np.random.shuffle(idx)
	        #training
	        d_loss_value = 0
	        g_loss_value = 0
	        gan_loss_value = 0
	        for i in tqdm.tqdm(range(step_num)):
	            x_batch,y_batch = getbatch(idx[i * BATCH_SIZE:(i + 1) * BATCH_SIZE],fs,fx)

	            if e > 4: 
	                sess.run(g_train_op, feed_dict={x: x_batch, y: y_batch, is_training: True})
	                # Compute the average loss
	                g_loss = sess.run(model.g_loss, feed_dict={x: x_batch, y: y_batch, is_training: True})
	                g_loss_value += g_loss

	                _, gan_loss,d_loss= sess.run([gan_train_op,model.gan_loss, model.d_loss], feed_dict={x: x_batch, y: y_batch, is_training: True})
	                gan_loss_value += gan_loss
	                d_loss_value += d_loss
	            else:
	                sess.run(g_train_op, feed_dict={x: x_batch, y: y_batch, is_training: True})
	                # Compute the average loss
	                g_loss = sess.run(model.g_loss, feed_dict={x: x_batch, y: y_batch, is_training: True})
	                g_loss_value += g_loss

	        #print("d_loss", d_loss_value/step_num)
	        print("g_loss", g_loss_value/step_num)
	        print("d_loss", d_loss_value/step_num)
	        print("gan_loss", gan_loss_value/step_num)

	        if(e%5 == 0):
	            saver = tf.train.Saver()
	            saver.save(sess, PRETRAIN_PATH + str(sess.run(epoch)), write_meta_graph=True)

	            out_feature1 = []
	            N1 = len(t2_data)
	            idx_inner = [i for i in range(N1)]
	            step_num1 = int(N1 / BATCH_SIZE)
	            for p in range(step_num1):
	                t2_batch = getbatch_test(idx_inner[p * BATCH_SIZE:(p + 1) * BATCH_SIZE],t2_data)
	                f = sess.run(model.gen_y, feed_dict={x: t2_batch, is_training: False})
	                out_feature1.append(f)
	            out_feature1 = np.reshape(out_feature1,[np.shape(out_feature1)[0]*np.shape(out_feature1)[1],np.shape(out_feature1)[2]])
	            print(np.shape(out_feature1))
	            #===================Precision and mAP Calculation======================================================
	            pred_img = out_feature1
	            test_img = t1_data

	            NEIGH_NUM = 200
	            nbrs = NearestNeighbors(n_neighbors=NEIGH_NUM, metric='cosine', algorithm='brute',n_jobs=-1).fit(test_img)
	            distances, indices = nbrs.kneighbors(pred_img)

	            distances=np.array(distances)
	            indices=np.array(indices)
	            retrieved_classes = test_img_label[indices]
	            results = np.zeros(retrieved_classes.shape)

	            for idx_prec in range(results.shape[0]):
	                results[idx_prec] = (retrieved_classes[idx_prec] == test_label_sketch[idx_prec])
	            precision_200 = np.mean(np.mean(results, axis=1))
	            temp = [np.arange(200) for ii in range(results.shape[0])]
	            mAP_term = 1.0/(np.stack(temp, axis=0) + 1)
	            mAP = np.mean(np.multiply(mapChange(results), mAP_term), axis=1)
	            print('The mAP for test_sketches is ' + str(np.mean(mAP)))
	            print(precision_200)


	        e = e + 1

def mapChange(inputArr):
    dup = np.copy(inputArr)
    for idx in range(inputArr.shape[1]):
        if (idx != 0):
            dup[:,idx] = dup[:,idx-1] + dup[:,idx]
    return np.multiply(dup, inputArr)

#top 2048 features--- without word2vec
def getbatch(idx,fs,fx):
	s = []
	x = []
	for i in range(len(idx)):
		s.append(fs[idx[i]][0:2048])
		x.append(fx[idx[i]])
	s = np.asarray(s)
	x = np.asarray(x)
	return s,x

def getbatch_test(idx,fx):
	x = []
	for i in range(len(idx)):
		x.append(fx[idx[i]][0:2048])
	x = np.asarray(x)
	return x

if __name__ == '__main__':
    train()
