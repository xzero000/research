import tensorflow as tf

import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import time
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer

'packet to img and img to packet'
def p_to_i(arr,bound):
    'arr is origin array, size should be 1xlength'
    'boumd is out put image weight and height, means output being boundXbound '
    b = bound
    out = np.zeros((b,b))
    l = len(arr)
    k = int(l**0.5)
    if k*k < l:
        k = k+1
    'k is kernel size'
    for i in range(b):
        s = (i*k)%l
        for j in range(b):
            p = (s+j)%l
            out[i][j] = arr[p]
    return out

def i_to_p(arr_2D,length):
    'arr_2D is a img'
    'length means output array, 2D image into 1D array'
    'arr_2D = mxm --> 1xlength array'
    out = np.zeros(length)
    tmp1 = np.zeros(length)
    tmp2 = np.zeros(length)
    b = len(arr_2D)
    k = int(length**0.5)
    if k*k < length:
        k = k+1
    for i in range(b):
        s = (i*k)%length
        for j in range(b):
            p = (s+j)%length
            tmp1[p] += arr_2D[i][j]
            tmp2[p] += 1
    out = tmp1/tmp2
    return out

'read data and preprocessing'
c_name = np.arange(0,43)
##data = pd.read_csv('KDDTrain+.txt',header = None,names = c_name)
##data_t = pd.read_csv('KDDTest+.txt',header = None, names = c_name)
'--------------- only ''normal'',''attack'' ----------------------'
#data = pd.read_csv('NSL-KDD/KDDtrain_normal_attack_4type_shuffle.csv',header = None,names = c_name)
data = pd.read_csv('NSL-KDD/KDDtrain_R2L.csv',header = None,names = c_name)


# data[0] & data[3] swap

'--- if ''KDDtrain_normal_attack_4type_shuffle.csv''------'
'---t_1 = normal dos probe R2L U2R-----'
t_1 = np.array(['normal', 'Dos','Probe',  'R2L', 'U2R'])
d_1 = np.array(['tcp', 'udp', 'icmp'])
d_2 = np.array(['ftp_data', 'other', 'private', 'http', 'remote_job', 'name',
       'netbios_ns', 'eco_i', 'mtp', 'telnet', 'finger', 'domain_u',
       'supdup', 'uucp_path', 'Z39_50', 'smtp', 'csnet_ns', 'uucp',
       'netbios_dgm', 'urp_i', 'auth', 'domain', 'ftp', 'bgp', 'ldap',
       'ecr_i', 'gopher', 'vmnet', 'systat', 'http_443', 'efs', 'whois',
       'imap4', 'iso_tsap', 'echo', 'klogin', 'link', 'sunrpc', 'login',
       'kshell', 'sql_net', 'time', 'hostnames', 'exec', 'ntp_u',
       'discard', 'nntp', 'courier', 'ctf', 'ssh', 'daytime', 'shell',
       'netstat', 'pop_3', 'nnsp', 'IRC', 'pop_2', 'printer', 'tim_i',
       'pm_dump', 'red_i', 'netbios_ssn', 'rje', 'X11', 'urh_i',
       'http_8001', 'aol', 'http_2784', 'tftp_u', 'harvest'])
d_3 = np.array(['SF', 'S0', 'REJ', 'RSTR', 'SH', 'RSTO', 'S1', 'RSTOS0', 'S3', 'S2', 'OTH'])

##X_train_0, X_test_0, y_train, y_test = \
##    train_test_split(data[c_name[0:41]],data[c_name[41]],test_size = 0.2)
##train_test_split(data[column_name[1:10]],data[column_name[10]],test_size=0.25,random_state=33)
## 取前10列为X，第10列为y，并且分割；random_state参数的作用是为了保证每次运行程序时都以同样的方式进行分割

X_train_0 = np.array(data[c_name[0:41]])
y_train = np.array(data[c_name[41]])
print('data load ok! ')

'one hot encoder'
def rr(x):
    if len(x[0]) == 0:
        return 0
    else:
        return (x[0][0]+1)
for i in range(len(X_train_0)):
    x1 = np.where(d_1 == X_train_0[i][1])
    x2 = np.where(d_2 == X_train_0[i][2])
    x3 = np.where(d_3 == X_train_0[i][3])
    X_train_0[i][1:4] = [x1[0][0]+1,x2[0][0]+1,x3[0][0]+1]
    y  = np.where(t_1 == y_train[i])
    y_train[i] = rr(y)

'standard scalar and minmax ,and normalizer'    
new_c = np.append(c_name[0],c_name[4:41])
new_c_hand = np.array([0, 4,  5,  7,  8,  9, 10, 12, 15, 16, 17, 18, 22, 23,31,32])
ss = StandardScaler()
mm = MinMaxScaler()

ct = ColumnTransformer([('', ss, new_c_hand)], remainder='passthrough')
X_train_s = ct.fit_transform(X_train_0)
arr = np.arange(len(new_c_hand))
ct = ColumnTransformer([('', mm, arr)], remainder='passthrough')
X_train_m = ct.fit_transform(X_train_s)
print('ss OK! ')

nor = Normalizer(norm='max')
nor_c = np.array([16,17,18,22])
ct = ColumnTransformer([('', nor, nor_c)], remainder='passthrough')
X_train = ct.fit_transform(X_train_m)
print('nor OK!')

X_train_0 = []
X_train_s = []
X_train_m = []
y_train = y_train.astype('float64')


'gan'

def discriminator(images, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:
        
        # First convolutional and pool layers
        # This finds 32 different 5 x 5 pixel features
        d_w1 = tf.get_variable('d_w1', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))
        d1 = tf.nn.conv2d(input=images, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
        d1 = d1 + d_b1
        d1 = tf.nn.relu(d1)
        d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Second convolutional and pool layers
        # This finds 64 different 5 x 5 pixel features
        d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
        d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
        d2 = d2 + d_b2
        d2 = tf.nn.relu(d2)
        d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # First fully connected layer
        d_w3 = tf.get_variable('d_w3', [7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0))
        d3 = tf.reshape(d2, [-1, 7 * 7 * 64])
        d3 = tf.matmul(d3, d_w3)
        d3 = d3 + d_b3
        d3 = tf.nn.relu(d3)

        # Second fully connected layer
        d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))
        d4 = tf.matmul(d3, d_w4) + d_b4

        # d4 contains unscaled values
        return d4

def generator(z, batch_size, z_dim):
    # From z_dim to 56*56 dimension
    g_w1 = tf.get_variable('g_w1', [z_dim, 3136], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b1 = tf.get_variable('g_b1', [3136], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g1 = tf.matmul(z, g_w1) + g_b1
    g1 = tf.reshape(g1, [-1, 56, 56, 1])
    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
    g1 = tf.nn.relu(g1)

    # Generate 50 features
    g_w2 = tf.get_variable('g_w2', [3, 3, 1, z_dim/2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b2 = tf.get_variable('g_b2', [z_dim/2], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME')
    g2 = g2 + g_b2
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
    g2 = tf.nn.relu(g2)
    g2 = tf.image.resize_images(g2, [56, 56])

    # Generate 25 features
    g_w3 = tf.get_variable('g_w3', [3, 3, z_dim/2, z_dim/4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b3 = tf.get_variable('g_b3', [z_dim/4], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
    g3 = g3 + g_b3
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
    g3 = tf.nn.relu(g3)
    g3 = tf.image.resize_images(g3, [56, 56])

    # Final convolution with one output channel
    g_w4 = tf.get_variable('g_w4', [1, 1, z_dim/4, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b4 = tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
    g4 = g4 + g_b4
    g4 = tf.sigmoid(g4)

    # Dimensions of g4: batch_size x 28 x 28 x 1
    return g4


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
    plt.savefig('img2/16_train_{}.png'.format(str(index).zfill(3)),bbox_inches = 'tight')
    plt.close(fig) 

#set train set to batchsize*n
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

""" See the fake image we make """

# Define the plceholder and the graph
z_dimensions = 100
z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions])

# For generator, one image for a batch
generated_image_output = generator(z_placeholder, 1, z_dimensions)
z_batch = np.random.normal(0, 1, [1, z_dimensions])

##with tf.Session() as sess:
##    sess.run(tf.global_variables_initializer())
##    generated_image = sess.run(generated_image_output,
##                                feed_dict={z_placeholder: z_batch})
##    generated_image = generated_image.reshape([28, 28])
##    plt.imshow(generated_image, cmap='Greys')
##    plt.savefig("img2/test_img.png")


""" For Training GAN """

tf.reset_default_graph()
batch_size = 50

z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder') 
# z_placeholder is for feeding input noise to the generator

x_placeholder = tf.placeholder(tf.float32, shape = [None,28,28,1], name='x_placeholder') 
# x_placeholder is for feeding input images to the discriminator

Gz = generator(z_placeholder, batch_size, z_dimensions) 
# Gz holds the generated images

Dx = discriminator(x_placeholder) 
# Dx will hold discriminator prediction probabilities
# for the real MNIST images

Dg = discriminator(Gz, reuse_variables=True)
# Dg will hold discriminator prediction probabilities for generated images

# Two Loss Functions for discriminator
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dx, labels = tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.zeros_like(Dg)))

# Loss function for generator
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.ones_like(Dg)))


# Get the varaibles for different network
tvars = tf.trainable_variables()

d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

print([v.name for v in d_vars])
print([v.name for v in g_vars])


# Train the discriminator
d_trainer_fake = tf.train.AdamOptimizer(0.0003).minimize(d_loss_fake, var_list=d_vars)
d_trainer_real = tf.train.AdamOptimizer(0.0003).minimize(d_loss_real, var_list=d_vars)

# Train the generator
g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)


# """ For setting TensorBoard """

# From this point forward, reuse variables
tf.get_variable_scope().reuse_variables()

# tf.summary.scalar('Generator_loss', g_loss)
# tf.summary.scalar('Discriminator_loss_real', d_loss_real)
# tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)

# images_for_tensorboard = generator(z_placeholder, batch_size, z_dimensions)
# tf.summary.image('Generated_images', images_for_tensorboard, 5)
# merged = tf.summary.merge_all()
# logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
# writer = tf.summary.FileWriter(logdir, sess.graph)


'set train set'

X_train = train_to_batch(batch_size,X_train)
'change X_train into img'
bound = 28
lx = len(X_train)
X_train_img = np.zeros((lx,bound,bound))
for i in range(lx):
    tmp = p_to_i(X_train[i],bound)
    X_train_img[i] = tmp
train_set = X_train_img

""" Start Training Session """
print('start trainning')
epoch = 100000
saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Pre-train discriminator
for i in range(300):
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    #real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    start = (batch_size*i)%len(train_set)
    input_set = train_set[start:start+batch_size].reshape([batch_size, 28, 28, 1])    
    _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                           {x_placeholder: input_set, z_placeholder: z_batch})

    if(i % 100 == 0):
        print('pre-train: ',i," dLossReal:", dLossReal, "dLossFake:", dLossFake)

# Train generator and discriminator together
for i in range(epoch):
    #real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    start = (batch_size*i)%len(train_set)
    input_set = train_set[start:start+batch_size].reshape([batch_size, 28, 28, 1]) 
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])

    # Train discriminator on both real and fake images
    _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                           {x_placeholder: input_set, z_placeholder: z_batch})

    # Train generator
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    _ = sess.run(g_trainer, feed_dict={z_placeholder: z_batch})

    # if i % 10 == 0:
    #     # Update TensorBoard with summary statistics
    #     z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    #     summary = sess.run(merged, {z_placeholder: z_batch, x_placeholder: real_image_batch})
    #     writer.add_summary(summary, i)
    
    if i % 10000 == 0:
        # Save the model every 1000 iteration
        save_path = saver.save(sess, "nslkddmodel/model{}.ckpt".format(i))
        print("epoch %d ,Model saved in file: %s" % (i,save_path))

    if i % 2000 == 0:
        # Every 100 iterations, show a generated image
        print("Iteration:", i, "at", datetime.datetime.now())
        z_t = np.random.normal(0, 1, size=[16, z_dimensions])
        generated_images = generator(z_placeholder, 1, z_dimensions)
        images = sess.run(generated_images, {z_placeholder: z_t})
        #plt.imshow(images[0].reshape([28, 28]), cmap='Greys')
        #plt.savefig("img/image{}.png".format(i))
        #save_f_image(i,images)
        

        # Show discriminator's estimate
        im = images[0].reshape([1, 28, 28, 1])
        result = discriminator(x_placeholder)
        estimate = sess.run(result, {x_placeholder: im})
        print("Estimate:", estimate)

print('Generate Data')
ww = []
for i in range(10):
    z_t = np.random.normal(0, 2, size=[1000, z_dimensions])
    generated_images = generator(z_placeholder, 1, z_dimensions)
    images = sess.run(generated_images, {z_placeholder: z_t})
    li = len(images)
    f = 41
    g_p = np.zeros((li,f))
    for j in range(li):
        tmp = i_to_p(images[j],f)
        g_p[j] = tmp
        ww.append(tmp)
with open('g_nslkdd/g_R2L_test1_epoch_%d.csv' %epoch,'w',newline = '') as csvfile:
    writer = csv.writer(csvfile)
    #for i in range(len(ww)):
    writer.writerows(ww)
#save_f_image(i,images)
print('OK!!!')
