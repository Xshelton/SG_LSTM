import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
global count_i
count_i=0
global count_i2
count_i2=0
from sklearn.metrics import roc_curve, auc

lr = 0.001                #learning rate，
training_iters = 100000   #
batch_size = 111          #batch size
 
n_inputs = 16         # MNIST data input (img shape: 28*28)
n_steps = 16           # time steps
n_hidden_units = 128    # neurons in hidden layer, also 
#the number of units in the LSTM cell.
n_classes = 2       # MNIST classes (0-9 digits)

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
#print(pred)
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

def roc(y_test,y_score):
 print('begain to plot the roc curve')
 fpr,tpr,threshold = roc_curve(y_test, y_score)
 roc_auc = auc(fpr,tpr)
 print('load picture')
 plt.figure()
 lw = 2
 plt.figure(figsize=(10,10))
 plt.plot(fpr, tpr, color='darkorange'
          ,
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) 
 plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
 plt.xlim([0.0, 1.0])
 plt.ylim([0.0, 1.05])
 plt.xlabel('False Positive Rate')
 plt.ylabel('True Positive Rate')
 plt.title('Receiver operating characteristic example')
 plt.legend(loc="lower right")
 plt.show()

 
import pandas as pd
for gg in range(5,6):
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
      test_label=[]
      y_score=[]
      #you have to change the model root to the fold that you train your model
      module_file = tf.train.latest_checkpoint("I://DNN模型重新测试/LSTM测试/测试24 SG相乘128 五折交叉验证/LSTM for test/models{}".format(gg))
      saver.restore(sess,module_file)
      print('model{}reload successfully'.format(gg))
      th=int(len(X_test)/batch_size)#6272个样本 但是实际上 只覆盖掉了
      
      for i in range(0,th):#128一个轮回
    #
       x_test_batch,y_test_batch=next_test_batches(i)
       x_test_batch=x_test_batch.reshape([batch_size, n_steps, n_inputs])

       print(sess.run(accuracy, feed_dict={
            x: x_test_batch,
            y: y_test_batch,
           
            }))
       y2 = sess.run(correct_pred,feed_dict={x:x_test_batch,y: y_test_batch})
       y3 = sess.run(pred,feed_dict={x:x_test_batch,y: y_test_batch})

       for j in range(0,len(y3)):
           test_label.append(y_test_batch[j][0])
           y_score.append(y3[j][0])
      print(len(test_label),'/',len(X_test))#25125个测试样本
      roc(test_label,y_score)
      #X_test.to_csv('LSTM_x_test.csv')
      np.save('SG_LSTM_y_score{}'.format(gg),y_score)
      np.save('SG_LSTM_y_label{}'.format(gg),test_label)
      print('file save successfully')
   
