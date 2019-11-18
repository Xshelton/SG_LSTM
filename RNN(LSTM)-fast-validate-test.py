import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
global count_i
count_i=0
global count_i2
count_i2=0
from sklearn.metrics import roc_curve, auc
your_dataset_file_name='AXB_383_gene_default.csv'
your_file_path="C://Users/shelton/Desktop/fast validate/models"


lr = 0.001                #learning rate，用于梯度下降
training_iters = 100000   #训练的循环次数
batch_size = 128          #随机梯度下降时，每次选择batch的大小
 
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
    # hidden layer for input to cell
    ########################################
 
    # transpose the inputs shape from
    # X ==> (128 batch * 28 steps, 28 inputs)
    # for reshape, one shape dimension can be -1. 
    # In this case, the value is inferred from the length of 
    # the array and remaining dimensions.
    X = tf.reshape(X, [-1, n_inputs])
 
    # into hidden
    # X_in = (128 batch * 28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
 
# cell
    ##########################################
    #The implementation of Basic LSTM recurrent network cell.
    #We add forget_bias (default: 1) to the biases of the forget gate
    #in order to reduce the scale of forgetting in the beginning of the training.
    #It does not allow cell clipping, a projection layer, 
    #and does not use peep-hole connections: it is the basic baseline.
    #For advanced models, please use the full tf.nn.rnn_cell.LSTMCell.
    
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)  #num_units
 #int, The number of units in the LSTM cell.
    #cell = tf.contrib.rnn.GRUCell(n_hidden_units)
    
    # lstm cell is divided into two parts (c_state, h_state)
    # for the basic RNN, its states is only h_state
    # zero_state returns zero-filled state tensor(s).
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
 
    # You have 2 options for following step.
    # 1: tf.nn.rnn(cell, inputs);
    # 2: tf.nn.dynamic_rnn(cell, inputs).
    # If use option 1, you have to modify the shape of X_in, go and check out [2].
    # In here, we go for option 2. (which is recommended.)
    # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
    # Make sure the time_major is changed accordingly.
    # outputs is a list which stores all outputs for each step, final_state is the output of the last one state
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
 
# hidden layer for output as the final results
    #############################################
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']
 
    # # or
    # unpack to list [(batch, outputs)..] * steps
    # Analogously to tf.pack and tf.unpack, 
    # we're renamed TensorArray.pack and TensorArray.unpack 
    # to TensorArray.stack and TensorArray.unstack.
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

import pandas as pd
df=pd.read_csv('AXB_383_gene_default.csv')
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
X=df
Y=label

        #print(len(label))
        #print(label)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2, random_state=0)#5折交叉验证

print(len(X_test))#一开始有6388个测试集


test_batches(batch_size,X_test,y_test)
batches(batch_size, X_train, y_train)
def roc(y_test,y_score):
 print('模型载入完毕')
 fpr,tpr,threshold = roc_curve(y_test, y_score)
 roc_auc = auc(fpr,tpr)
 print('开始载入图片')
 plt.figure()
 lw = 2
 plt.figure(figsize=(10,10))
 plt.plot(fpr, tpr, color='darkorange'
          ,
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
 plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
 plt.xlim([0.0, 1.0])
 plt.ylim([0.0, 1.05])
 plt.xlabel('False Positive Rate')
 plt.ylabel('True Positive Rate')
 plt.title('Receiver operating characteristic example')
 plt.legend(loc="lower right")
 plt.show()
 
with tf.Session() as sess:
   test_label=[]
   y_score=[]
   try:
    module_file = tf.train.latest_checkpoint(your_file_path)
    saver.restore(sess,module_file)
    print('model load successfully')
   except:
    print('model load fail,please check out the path')
   itera=int(len(X_test)/128)
   for i in range(0,itera):#128一个轮回
    #
    x_test_batch,y_test_batch=next_test_batches(i)
    x_test_batch=x_test_batch.reshape([batch_size, n_steps, n_inputs])
    #print(type(x_test_batch))
    #print(x_test_batch.shape)
    #print(y_test_batch.shape)
    #print(y_test_batch)
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
   np.save('LSTM_y_score',y_score)
   np.save('LSTM_y_label',test_label)
   print('We produced the y_score and y_label for your test set, named LSTM_y_score and LSTM_y_label')
   
