import tensorflow as tf
import numpy as np
import sklearn
import os
import cv2

# Data Preparation:

par_dir = "E:/Tensorflow/notMNIST/notMNIST_small"
path = os.listdir(par_dir)
image_list = []
label_list = []
label=0

for folder in path:
    images = os.listdir(par_dir + '/' + folder)
    for image in images:
        if(os.path.getsize(par_dir +'/'+ folder +'/'+ image) > 0):
            img = cv2.imread(par_dir +'/'+ folder +'/'+ image, 1)
            image_list.append(img)
            label_list.append(label)
        else:
            print('File ' + par_dir +'/'+ folder +'/'+ image + ' is empty')
    label += 1

print("Looping done")

image_array = np.empty([len(image_list),28, 28, 3])
for x in range(len(image_list)):
    image_array[x] = np.array(image_list[x])

image_array = image_array.astype(np.float32)

label_array = np.array(label_list)

one_hot = np.eye(10)[label_array]

image_data, one_hot = sklearn.utils.shuffle(image_array, one_hot)

print("Data ready. Bon Apetiet!")


image_train, label_train = image_data[0:12800], one_hot[0:12800]
image_test, label_test = image_data[12800:17920], one_hot[12800:17920]

def get_train_image(input):
    
    batch_images = image_train[(input*batch_size):((input+1)*batch_size)]
    batch_label = label_train[(input*batch_size):((input+1)*batch_size)]
    return batch_images, batch_label

def get_test_image(input):

    batch_images = image_test[(input*batch_size):((input+1)*batch_size)]
    batch_label = label_test[(input*batch_size):((input+1)*batch_size)]
    return batch_images, batch_label

# HyperParameters:

learning_rate = 0.01
batch_size = 128
epochs = 1
dropout = 0.7
n_samples = 12800
t_samples = 5120

# Parameters:

X = tf.placeholder(tf.float32, [batch_size, 28, 28, 3], name="X")
Y = tf.placeholder(tf.float32, [batch_size, 10], name="Y")

# Convolution layers:

#Convolution layer 1:

kernel1 = tf.Variable(tf.random_normal([5,5,3,32], stddev=0.1, dtype=tf.float32), name="filter1")
bias1 = tf.Variable(tf.zeros([32]), dtype=tf.float32, name="bias1")

conv1 = tf.nn.conv2d(X, kernel1, strides=[1,1,1,1], padding='SAME')
relu1 = tf.nn.relu(conv1+bias1, name='ReLU1')
pool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='max_pool1')

# Convolution Layer 2:

kernel2 = tf.Variable(tf.random_normal([5,5,32,64], stddev=0.1, dtype=tf.float32), name='kernel2')
bias2 = tf.Variable(tf.zeros([64]), dtype=tf.float32, name='bias2')

conv2 = tf.nn.conv2d(pool1, kernel2, strides=[1,1,1,1], padding='SAME')
relu2 = tf.nn.relu(conv2+bias2, name='ReLU2')
pool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool2')

# Flattern:

length = 7*7*64

flat = tf.reshape(pool2, [-1, length])

# Fully Connected:

w1 = tf.Variable(tf.random_normal([length,1024], stddev=0.1, dtype=tf.float32), name='w1')
b1 = tf.Variable(tf.zeros([1024]), dtype=tf.float32, name='b1')

fc1 = tf.matmul(flat,w1) + b1
fc1 = tf.nn.relu(fc1, name='ReLU_FC1')

fc1 = tf.nn.dropout(fc1, dropout, name='FC1_dropout')

# Softmax Layer:

weight = tf.Variable(tf.random_normal([1024,10], stddev=0.1, dtype=tf.float32), name='Weight')
b = tf.Variable(tf.zeros([10]), dtype=tf.float32, name='bias')

logit = tf.matmul(fc1,weight) + b
log = tf.reduce_mean(logit)
entropy = tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = logit)
loss = tf.reduce_mean(entropy)

# Optimization:

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

# The loop:

with tf.Session() as sess:

    sess.run(init)

    # Training Loop:

    for i in range(epochs):

        total_loss = 0

        loops = int(n_samples/batch_size)

        for j in range(loops):

            X_batch, Y_batch = get_train_image(j)
            _,loss_batch,logits = sess.run([optimizer,loss,log], feed_dict={X:X_batch, Y:Y_batch})
            total_loss += loss_batch
            #print("Loss: {0}".format(loss_batch))
        print('Average loss epoch: {0}'.format(total_loss))

    print("Optimization done!")

    t_loop = int(t_samples/batch_size)
    total_correct = 0

    # Verification Loop:
    
    for k in range(t_loop):
        X_batch, Y_batch = get_test_image(k)
        _,loss_batch,logits_batch = sess.run([optimizer, loss, logit], feed_dict={X:X_batch, Y:Y_batch})
        preds = tf.nn.softmax(logits_batch)
        correct_preds = tf.equal(tf.argmax(preds,1), tf.argmax(Y_batch,1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
        total_correct += sess.run(accuracy)

    print("Final result: {0}".format(total_correct/t_samples))
