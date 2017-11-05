import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32,shape = [None,784])
x_image = tf.reshape(x,shape = [-1,28,28,1])
tf.summary.image('input',x_image,3)

#layers
l_1 = 150

ew_1 = tf.Variable(tf.random_normal([784, l_1]),name = "ew_1")

dw_1 = tf.Variable(tf.random_normal([l_1, 784]),name = "dw_1")

eb_1 = tf.Variable(tf.random_normal([l_1]), name =  "eb_1")

db_1 = tf.Variable(tf.random_normal([784]),name = "db_1")


def encoder(x):

	en1 = tf.nn.sigmoid(tf.matmul(x,ew_1) + eb_1)
	return en1

def decoder(x):

	dn1 = tf.nn.sigmoid(tf.matmul(x,dw_1) + db_1)
	return dn1

en_out = encoder(x)
de_out = decoder(en_out)
en_image = tf.reshape(de_out,shape=[-1,28,28,1])
tf.summary.image('out',en_image,3)


with tf.name_scope("loss"):

  loss = tf.reduce_mean(tf.pow(de_out - x,2))
  #loss = tf.reduce_mean(cross_entropy + 0.01 * regularizer)
  tf.summary.scalar('loss',loss)

with tf.name_scope("train"):

  train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

with tf.name_scope("accuracy"):

  correct_prediction = tf.equal(de_out,x)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
  tf.summary.scalar('accuracy',accuracy)

merged_summary = tf.summary.merge_all()
#writer = tf.summary.FileWriter("/home/psycholearner/projects/DL4CV-EE6132-/PA4/st75_train")
#writer.add_graph(sess.graph)

writer2 = tf.summary.FileWriter("/home/psycholearner/projects/DL4CV-EE6132-/PA4/st75_test")
writer2.add_graph(sess.graph)

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())


saver.restore(sess, "/home/psycholearner/projects//DL4CV-EE6132-/PA4/st75_model.ckpt")
'''
for i in range(10000):

	batch = mnist.train.next_batch(50)
	
	if i%3 == 0:
		s = sess.run(merged_summary,feed_dict = {x: batch[0]})
		writer.add_summary(s,i)

	if i % 20 == 0:
		batch2 = mnist.test.next_batch(5000) 
		s2 = sess.run(merged_summary,feed_dict = {x:batch2[0]} )
		writer2.add_summary(s2,i)

	sess.run(train_step, feed_dict={x: batch[0]})

save_path = saver.save(sess, "/home/psycholearner/projects/DL4CV-EE6132-/PA4/st75_model.ckpt")
'''
test_batch = mnist.test.next_batch(3)
s2 = sess.run(merged_summary,feed_dict = {x:test_batch[0]})
writer2.add_summary(s2,1)