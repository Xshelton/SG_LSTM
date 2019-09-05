import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
global count_i
count_i=0
global count_i2
count_i2=0



lr = 0.001                #learning rate
training_iters = 5000000   #
                           #
batch_size = 128          #batchsize
 
n_inputs = 16         # our embedding is 256 dimensions , here we set n_inputs as 16
n_steps = 16           # time steps 16 cuz 16*16=256
n_hidden_units = 128    # neurons in hidden layer, also 
#the number of units in the LSTM cell.
n_classes = 2       # label zero or one

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}
def RNN(X, weights, biases):
    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)  #num_units
    init_state = cell.zero_state(batch_size, dtype=tf.float32)   
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 10)
    return results

pred = RNN(x, weights, biases)
print(pred)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
def batches(batch_size, features, labels):
               global list_bx
               global list_by
               assert len(features) == len(labels)
               output_batches = []
               list_bx=[]
               list_by=[]
               sample_size = len(features)
               for start_i in range(0, sample_size, batch_size):
                   end_i = start_i + batch_size
                   batch_x = features[start_i:end_i]
                   list_bx.append(batch_x)
                   #print(type(batch_x))
                   batch_y = labels[start_i:end_i]
                   #output_batches.append(batch)
                   list_by.append(batch_y)
def next_batches(i):
            global list_bx
            global list_by
            global count_i
            if count_i<len(list_bx)-2:
                batch_x=list_bx[count_i]
                batch_y=list_by[count_i]
                count_i+=1
            else:
                batch_x=list_bx[count_i]
                batch_y=list_by[count_i]
                count_i=0
            batch_x=batch_x.values
            batch_y=batch_y.values
            return batch_x,batch_y
def next_test_batches(i):
            global test_bx
            global test_by
            global count_i2
            if count_i2<len(test_bx)-2:
                batch_x=test_bx[count_i2]
                batch_y=test_by[count_i2]
                count_i2+=1
            else:
                batch_x=test_bx[count_i2]
                batch_y=test_by[count_i2]
                count_i2=0
            batch_x=batch_x.values
            batch_y=batch_y.values
            return batch_x,batch_y
def test_batches(batch_size, features, labels):
            global test_bx
            global test_by
            assert len(features) == len(labels)
            output_batches = []
            test_bx=[]
            test_by=[]
            sample_size = len(features)
            for start_i in range(0, sample_size, batch_size):
                end_i = start_i + batch_size
                batch_x = features[start_i:end_i]
                test_bx.append(batch_x)
                #print(type(batch_x))
                batch_y = labels[start_i:end_i]
                #output_batches.append(batch)
                test_by.append(batch_y)     

saver = tf.train.Saver(max_to_keep=1)
import pandas as pd
#This part is for the Training and Testing.
#Name the Training set  like X_train1.csv and testing set like X_test1.csv
#and change the file_number into 1
#After initializing: you will see sth like this
#the result in training set 0
#0.6328125
#the result in test set 0
#0.546875

file_number=5
for gg in range(file_number,file_number+1):#for five fold cross validation  
   df=pd.read_csv('X_train{}.csv'.format(gg))
   label=df['label']
        #df=df.drop(['id'],axis=1)
   df=df.drop(['0_mirna'],axis=1)
   df=df.drop(['1_gene'],axis=1)
   df=df.drop(['label'],axis=1)
   list_label2=[]
   for i in range(0,len(label)):
      if label[i]==1:
             list_label2.append(0)
      else:
             list_label2.append(1)
   list_2_pd=pd.DataFrame(list_label2)
   label=pd.concat([label,list_2_pd],axis=1)
   X_train=df
   y_train=label
   print(len(X_train))
   print(len(y_train))
   df2=pd.read_csv('X_test{}.csv'.format(gg))
   label2=df2['label']
        #df=df.drop(['id'],axis=1)
   df2=df2.drop(['0_mirna'],axis=1)
   df2=df2.drop(['1_gene'],axis=1)
   df2=df2.drop(['label'],axis=1)
   list_label22=[]
   for i in range(0,len(label2)):
      if label2[i]==1:
             list_label22.append(0)
      else:
             list_label22.append(1)
   list_22_pd=pd.DataFrame(list_label22)
   label2=pd.concat([label2,list_22_pd],axis=1)
   X_test=df2
   y_test=label2
   print(len(X_test))
   print(len(y_test))


   test_batches(batch_size,X_test,y_test)
   batches(batch_size, X_train, y_train)         
   with tf.Session() as sess:
 
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        
        batch_xs, batch_ys = next_batches(step)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        #print(batch_xs.shape)
        
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        
        if step % 200 == 0:
            print('the result in training set',step)
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
            }))
            print('the result in test set',step)
            x_test_batch,y_test_batch=next_test_batches(i)
            x_test_batch = x_test_batch.reshape([batch_size, n_steps, n_inputs])
            print(sess.run(accuracy, feed_dict={
            x: x_test_batch,
            y: y_test_batch,
            
            }))
            saver.save(sess=sess, save_path="./models{}/my_data.ckpt".format(gg), global_step=step+1)
        step += 1
   print('{}fold/5folds model generation finished'.format(gg))

