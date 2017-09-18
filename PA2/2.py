from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot = True)
import tensorflow as tf
sess = tf.InteractiveSession()
import numpy as np
x = tf.placeholder(tf.float32,shape = [None,784])
y = tf.placeholder(tf.float32,shape = [None,10])	
def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides = [1,1,1,1],padding = "SAME")

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize = [1,2,2,1],strides = [1,2,2,1],padding = "SAME")

noise = tf.Variable(tf.zeros([784]),name="noise")

#var = [W_conv1]

#x = x+noise

#x = tf.add(x,noise)

x_noise = tf.add(x,noise)

W_conv1 = tf.Variable(tf.truncated_normal([3,3,1,32],stddev = 0.1),name = "W1")
b_conv1 = tf.Variable(tf.truncated_normal([32],stddev = 0.1),name = "B1")

x_image = tf.reshape(x_noise,[-1,28,28,1])
tf.summary.image('input',x_image,3)

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = tf.Variable(tf.truncated_normal([3,3,32,32],stddev = 0.1),name = "W2")
b_conv2 = tf.Variable(tf.truncated_normal([32],stddev = 0.1),name = "B2")

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = tf.Variable(tf.truncated_normal([7*7*32,500],stddev = 0.1),name = "W3")
b_fc1 = tf.Variable(tf.truncated_normal([500],stddev = 0.1),name = "B3")

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

W_fc2 = tf.Variable(tf.truncated_normal([500,10],stddev = 0.1),name = "W4")
b_fc2 = tf.Variable(tf.truncated_normal([10],stddev = 0.1),name = "B4")

logits = tf.matmul(h_fc1,W_fc2) + b_fc2

var = [noise]

with tf.name_scope("cross_entropy"):
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = y))
  tf.summary.scalar('cross entropy',cross_entropy)

with tf.name_scope("train"):
  train_step = tf.train.AdamOptimizer().minimize(cross_entropy,var_list=var)

with tf.name_scope("accuracy"):
  correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
  tf.summary.scalar('accuracy',accuracy)

with tf.name_scope("test_accuracy"):
  correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(y,1))
  test_accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
  tf.summary.scalar('test_accuracy',test_accuracy)


merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("/home/psycholearner/projects/DL4CV-EE6132-/PA2/11")
writer.add_graph(sess.graph)

writer2 = tf.summary.FileWriter("/home/psycholearner/projects//DL4CV-EE6132-/PA2/12")
writer2.add_graph(sess.graph)

writer3 = tf.summary.FileWriter("/home/psycholearner/projects//DL4CV-EE6132-/PA2/13")
writer3.add_graph(sess.graph)

writer4 = tf.summary.FileWriter("/home/psycholearner/projects//DL4CV-EE6132-/PA2/14")
writer4.add_graph(sess.graph)

writer5 = tf.summary.FileWriter("/home/psycholearner/projects//DL4CV-EE6132-/PA2/15")
writer5.add_graph(sess.graph)

writer6 = tf.summary.FileWriter("/home/psycholearner/projects//DL4CV-EE6132-/PA2/16")
writer6.add_graph(sess.graph)

writer7 = tf.summary.FileWriter("/home/psycholearner/projects//DL4CV-EE6132-/PA2/17")
writer7.add_graph(sess.graph)

writer8 = tf.summary.FileWriter("/home/psycholearner/projects//DL4CV-EE6132-/PA2/18")
writer8.add_graph(sess.graph)

writer9 = tf.summary.FileWriter("/home/psycholearner/projects//DL4CV-EE6132-/PA2/19")
writer9.add_graph(sess.graph)

writer10 = tf.summary.FileWriter("/home/psycholearner/projects//DL4CV-EE6132-/PA2/20")
writer10.add_graph(sess.graph)

saver = tf.train.Saver({"W1":W_conv1,"W2":W_conv2,"W3":W_fc1,"W4":W_fc2,"B1":b_conv1,"B2":b_conv2,"B3":b_fc1,"B4":b_fc2})

sess.run(tf.global_variables_initializer())

batch = mnist.train.next_batch(28)





for i in range(1000):
  

  if i % 1 == 0:

    n = {x: np.array([batch[0][27]]),y: np.array([batch[1][9]])}
    s = sess.run(merged_summary,feed_dict = n)

    writer10.add_summary(s,i)

    #k = sess.run(merged_summary,feed_dict = {x: mnist.test.images,y: mnist.test.labels})

    #writer2.add_summary(k,i)



    #train_accuracy = sess.run(accuracy.eval(feed_dict={x: batch[0], y: batch[1]}))
    [train_accuracy] = sess.run([accuracy],feed_dict = n)
    print('step %d, training accuracy %g' % (i, train_accuracy))

  saver.restore(sess, "/home/psycholearner/projects/DL4CV-EE6132-/PA2/model.ckpt")
  #save_path = saver.save(sess, "/home/psycholearner/projects/DL4CV-EE6132-/PA2/model.ckpt")
  sess.run(train_step,feed_dict = n)

