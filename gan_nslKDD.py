import tensorflow as tf
#import tensorflow.compat.v1 as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy
import time
import pandas as pd
import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression as lr
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.compose import ColumnTransformer

'-------------------------------------------------'
c_name = np.arange(0,43)
data = pd.read_csv('/home/xzero000/dataset/NSL-KDD/KDDtrain_Probe.csv',header = None,names = c_name)

# data[0] & data[3] swap
def count_e(arr):
    ele = np.array([],dtype = '<U1')
    e_c = np.array([],dtype = 'int32')
    for i in range(len(arr)):
        if arr[i] in ele:
            x = np.where(ele == arr[i])
            e_c[x] += 1
        else:
            y = np.array([arr[i]])
            ele = np.append(ele,y)
            c = 1
            e_c = np.append(e_c,c)
    return ele,e_c

def c_2(arr,ele,e_c):
    for i in range(len(arr)):
        if arr[i] in ele:
            x = np.where(ele == arr[i])
            e_c[x] += 1
        else:
            y = np.array([arr[i]])
            ele = np.append(ele,y)
            c = 1
            e_c = np.append(e_c,c)
    return ele,e_c

d_1,d_1c = count_e(data[1])
print('1')
d_2,d_2c = count_e(data[2])
print('2')
d_3,d_3c = count_e(data[3])
print('3')
t_1,t_1c = count_e(data[41])
print('count ok')

X_train_0 = data[c_name[0:41]]
y_train = data[c_name[41]]
y_train = y_train.values


print('data load ok! ')
new_c = np.append(c_name[0],c_name[4:41])
new_c_hand = np.array([0, 4,  5,  7,  8,  9, 10, 12, 15, 16, 17, 18, 22, 23,31,32])
ss = StandardScaler()

###The Box-Cox transformation can only be applied to strictly positive data
ct = ColumnTransformer([('', ss, new_c_hand)], remainder='passthrough')
##ct = ColumnTransformer([('', 'passthrough', new_c_hand)], remainder='passthrough')
X_train = ct.fit_transform(X_train_0)

# X_train[0][38] = 'tcp', [39] = 'http', [40] = 'SF'
def rr(x):
    if len(x[0]) == 0:
        return 0
    else:
        return (x[0][0]+1)
for i in range(len(X_train)):
    x1 = np.where(d_1 == X_train[i][16])
    x2 = np.where(d_2 == X_train[i][17])
    x3 = np.where(d_3 == X_train[i][18])
    y  = np.where(t_1 == y_train[i])
    X_train[i][16:19] = [x1[0][0]+1,x2[0][0]+1,x3[0][0]+1]
    y_train[i] = rr(y)

print('ss OK! ')

X_train = X_train.astype('float64')
y_train = y_train.astype('float64')

'--------------------------------------------------'


#generator
def generator(z,reuse = False):
    with tf.variable_scope('generator',reuse = reuse):
        #initializers
        w_init = tf.truncated_normal_initializer(mean = 0, stddev = 0.02)
        b_init = tf.constant_initializer(0.)

        #1st hidden layer
        G_w0 = tf.get_variable('G_w0',[100,256],initializer = w_init)
        G_b0 = tf.get_variable('G_b0',[256],initializer = b_init)
        G_fc0 = tf.nn.relu(tf.matmul(z,G_w0)+G_b0)

        #2nd hidden layer
        G_w1 = tf.get_variable('G_w1',[256,512],initializer = w_init)
        G_b1 = tf.get_variable('G_b1',[512],initializer = b_init)
        G_fc1 = tf.nn.relu(tf.matmul(G_fc0,G_w1)+G_b1)

        #output hidden layer
        G_w3 = tf.get_variable('G_w3',[512,41],initializer = w_init)
        G_b3 = tf.get_variable('G_b3',[41],initializer = b_init)
        f_image = tf.nn.tanh(tf.matmul(G_fc1,G_w3) + G_b3)
        
    return f_image

# discriminator
def discriminator(image,drop_out,reuse = False):
    with tf.variable_scope('discriminator',reuse = reuse):
        #initializers
        w_init = tf.truncated_normal_initializer(mean = 0, stddev = 0.02)
        b_init = tf.constant_initializer(0.)

        #1st hidden layer
        D_w0 = tf.get_variable('D_w0',[41,512],initializer = w_init)
        D_b0 = tf.get_variable('D_b0',[512],initializer = b_init)
        D_fc0 = tf.nn.relu(tf.matmul(image,D_w0)+D_b0)
        D_fc0 = tf.nn.dropout(D_fc0,drop_out)

        #3rd hidden layer
        D_w2 = tf.get_variable('D_w2',[512,256],initializer = w_init)
        D_b2 = tf.get_variable('D_b2',[256],initializer = b_init)
        D_fc2 = tf.nn.relu(tf.matmul(D_fc0,D_w2)+D_b2)
        D_fc2 = tf.nn.dropout(D_fc2,drop_out)
             
        #output layer
        D_w3 = tf.get_variable('D_w3',[256,1],initializer = w_init)
        D_b3 = tf.get_variable('D_b3',[1],initializer = b_init)
        output = tf.sigmoid(tf.matmul(D_fc2,D_w3) + D_b3)
        
        return output

#save image
def save_f_image(index,z_sample):
    fig = plt.figure(figsize = (4,4))
    gs = gridspec.GridSpec(4,4)
    gs.update(wspace = .005, hspace = 0.05)

    for i,sample in enumerate(z_sample):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28,28),cmap = 'gray')
    plt.savefig('gan_fig/{}.png'.format(str(index).zfill(3)),bbox_inches = 'tight')
    plt.close(fig)    

#input placeholder
x = tf.placeholder(tf.float32, shape = (None,41))
z = tf.placeholder(tf.float32, shape = (None,100))
drop_out = tf.placeholder(tf.float32)

#Generate fake image and discriminate real and fake
f_sample = generator(z)
D_real = discriminator(x,drop_out)
D_fake = discriminator(f_sample,drop_out,reuse = True)

# loss for generator and discriminator
D_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(D_real,1e-4,1.0))+ tf.log(tf.clip_by_value(1-D_fake,1e-4,1.0)))
G_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(D_fake,1e-4,1.0)))
'tf.log(tf.clip_by_value(tf.sigmoid(self.scores),1e-8,1.0)'
"""
eps = 1e-2
D_loss = tf.reduce_mean(-tf.log(D_real + eps) - tf.log(1 - D_fake + eps))
G_loss = tf.reduce_mean(-tf.log(D_fake + eps))
"""

#set hyper paramatrics
batch_size = 100
l_rate = 0.01
train_epoch = 10000
epoch = 0
index = 0

#trainable variables for each network
t_vars = tf.trainable_variables()
D_vars = [var for var in t_vars if 'D_' in var.name]
G_vars = [var for var in t_vars if 'G_' in var.name]

#optimizer for generator and dirvriminator
learning_rate = tf.placeholder(tf.float32,shape = [])
D_train = tf.train.AdamOptimizer(l_rate).minimize(D_loss,var_list = D_vars)
G_train = tf.train.AdamOptimizer(l_rate).minimize(G_loss,var_list = G_vars)

#setup tf session
gpu_options = tf.GPUOptions(allow_growth = True)
sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))

init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()

#normalize

def train_to_batch(batch_size,train_set):
    l = len(train_set)
    rr = l%batch_size
    rr = batch_size - rr
    add = []
    for i in range(rr):
        tmp = np.random.randint(l)
        tmp_x = train_set[tmp]
        train_set = np.vstack((train_set,tmp_x))
    return train_set

train_set = train_to_batch(batch_size,X_train)

#default_noise
test_z = np.random.normal(0,1,(16,100))

t1 = time.time()
print('start train')

while epoch <= train_epoch:
    G_losses = []
    D_losses = []
    #sess.graph.finalize(), for prevent memory explode
    fake_packages = []

    #Save all fake image
    if epoch %1 ==0:
        fake_p = sess.run(f_sample,{z:test_z,drop_out:0.0})
        fake_packages.append(fake_p)
        #save_f_image(index,fake_image)
        #index += 1
        if epoch %10 ==0:
            print('generate package: \n')
            print(repr(fake_p[0]))

    #start training
    for i in range(0,train_set.shape[0],batch_size):
        sess.graph.finalize()
        input_x = train_set[i:i+batch_size]
        input_z = np.random.normal(0,1,(batch_size,100))

        sess.run(D_train,{x:input_x,z:input_z,drop_out:0.3})
        loss_d = sess.run(D_loss,{x:input_x,z:input_z,drop_out:0.3})
        D_losses.append(loss_d)

        input_z = np.random.normal(0,1,(batch_size,100))

        sess.run(G_train,{z:input_z,drop_out:0.3})
        loss_g = sess.run(G_loss,{z:input_z,drop_out:0.3})
        G_losses.append(loss_g)
        if (i/batch_size)%100 == 0:
            print('%d,'%(i/batch_size),end = '')

    epoch += 1
    if epoch == 5:
        l_rate = 0.01
    if epoch == 20:
        l_rate = 0.001
    if epoch == 50:
        l_rate = 0.0002
    t2 = time.time()
    print("\n@epoch of {}, D_loss:{}, G_loss:{}, time: {}".format(epoch,np.mean(D_losses),np.mean(G_losses),t2-t1))
    t1 = t2
save_path = saver.save(sess,'/home/xzero000/GAN/model/gan_nslKDD_Probe_epoch%d.ckpt' %epoch)
print('Model saved in path:%s'%save_path)

g_size = len(train_set)*2
test_g = np.random.normal(0,1,(g_size,100))
g_p = sess.run(f_sample,{z:test_z,drop_out:0.0})
with open('/home/xzero000/dataset/NSL-KDD/generateset/g_Probe_test1_epoch%d.csv' %epoch,'w',newline = '') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(g_p)

