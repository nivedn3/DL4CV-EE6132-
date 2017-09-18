from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot = True)
import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32,shape = [None,784])
y = tf.placeholder(tf.float32,shape = [None,10])	
noise = tf.Variable(tf.zeros([784],dtype=tf.float32),name="noise")
#x = tf.add(x,noise)
x_noise = tf.Variable(tf.zeros([784],dtype=tf.float32),name="noise")


def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides = [1,1,1,1],padding = "SAME")

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize = [1,2,2,1],strides = [1,2,2,1],padding = "SAME")


W_conv1 = tf.Variable(tf.truncated_normal([3,3,1,32],stddev = 0.1),name = "W1")
b_conv1 = tf.Variable(tf.truncated_normal([32],stddev = 0.1),name = "B1")

x_image = tf.reshape(x,[-1,28,28,1])

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


with tf.name_scope("cross_entropy"):
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = y ))
  tf.summary.scalar('cross entropy',cross_entropy)

with tf.name_scope("train"):
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


with tf.name_scope("accuracy"):
  correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
  tf.summary.scalar('accuracy',accuracy)

with tf.name_scope("test_accuracy"):
  correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(y,1))
  test_accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
  tf.summary.scalar('test_accuracy',test_accuracy)


merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("/home/psycholearner/projects//DL4CV-EE6132-/PA2/1")
writer.add_graph(sess.graph)

writer2 = tf.summary.FileWriter("/home/psycholearner/projects//DL4CV-EE6132-/PA2/2")
writer2.add_graph(sess.graph)



saver = tf.train.Saver()



sess.run(tf.global_variables_initializer())

for i in range(1000):
  batch = mnist.train.next_batch(50)

  if i % 25 == 0:

    s = sess.run(merged_summary,feed_dict = {x: batch[0],y: batch[1]})

    writer.add_summary(s,i)

    k = sess.run(merged_summary,feed_dict = {x: mnist.test.images,y: mnist.test.labels})

    writer2.add_summary(k,i)


    #train_accuracy = sess.run(accuracy.eval(feed_dict={x: batch[0], y: batch[1]}))
    [train_accuracy] = sess.run([accuracy],feed_dict = {x: batch[0], y:batch[1]})
    print('step %d, training accuracy %g' % (i, train_accuracy))

  #saver.restore(sess, "/home/psycholearner/projects//DL4CV-EE6132-/PA2/model.ckpt")
  save_path = saver.save(sess, "/home/psycholearner/projects/DL4CV-EE6132-/PA2/model.ckpt")
  sess.run(train_step,feed_dict = {x: batch[0],y:batch[1]})