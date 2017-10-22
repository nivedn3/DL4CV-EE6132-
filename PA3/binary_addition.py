import tensorflow as tf
import numpy as np 
sess = tf.InteractiveSession()

def data(number,size):

	a = []
	b = []
	out = []
	for i in range(number):
		a_in = np.random.choice([0,1],size)
		a_in = a_in.tolist()
		b_in = np.random.choice([0,1],size)
		b_in = b_in.tolist()
		a_str = ','.join(str(x) for x in a_in).replace(',','')
		b_str = ','.join(str(x) for x in b_in).replace(',','')
		c = bin(int(a_str,2) + int(b_str,2)).split('b')[1]
		c = [int(i) for i in list(c)]
		c_out = np.array(c)
		if len(c_out) == size:
			c_out = np.insert(c_out,0,0)
		if len(c_out) < size:
			while(len(c_out) != size+1):
				c_out = np.insert(c_out,0,0)
		test = []
		for j in range(len(a_in)):
			test.append(a_in[j])
			test.append(b_in[j])
		a.append(test)
		#b.append(b_in)
		out.append(c_out)

	return a,out

size = 3

x = tf.placeholder(tf.float32,shape = [None,3,2])
y = tf.placeholder(tf.float32,shape = [None,size+1]) 


w = tf.Variable(tf.random_normal([5,4]))
b = tf.Variable(tf.random_normal([4]))

rnn_inp = tf.unstack(x,size,1)

lstm = tf.contrib.rnn.BasicLSTMCell(5)

outputs, states = tf.contrib.rnn.static_rnn(lstm, rnn_inp, dtype=tf.float32)

logits = tf.nn.sigmoid(tf.matmul(outputs[-1], w) + b)

logits = tf.add(logits,tf.scalar_mul(-0.5,tf.ones_like(logits)))

logits = tf.nn.relu(logits)

logits = tf.scalar_mul(1000000,logits)

logits = tf.clip_by_value(logits,0,1)

with tf.name_scope("cross_entropy"):
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = y))
  tf.summary.scalar('cross entropy',cross_entropy)

with tf.name_scope("train"):
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


with tf.name_scope("accuracy"):
  correct_prediction = tf.equal(logits,y)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
  tf.summary.scalar('accuracy',accuracy)


merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("/home/psycholearner/projects//DL4CV-EE6132-/PA3/13")
writer.add_graph(sess.graph)

sess.run(tf.global_variables_initializer())

for i in range(10000):

	a,batch_y = data(128,3)

	batch_x = np.array(a)
	batch_x = batch_x.reshape(128,3,2)

	batch_y = np.array(batch_y)

	#batch_x = batch_x.reshape((128, 28, 28))

	#batch_x =  []
	#for i in range(128):
	#	batch_x.append(np.array([([x,y]) for x,y in zip(batch_a[i],batch_b[i])]))		



	#batch_x =  np.array(batch_x)
	#batch_y =  np.array(batch_y)
	#print batch_x[0].shape
	#print batch_y[0].shape

	'''batch_x = [1,0,0,1,1,0]
	batch_x = np.array(batch_x)
	batch_x = batch_x.reshape(1,3,2)
	
	print batch_x.shape


	batch_y = [0,1,1,1]
	batch_y = np.array(batch_y)
	batch_y = batch_y.reshape(1,4)
	#batch_x = []'''
	'''for j in a:
		batch_x.append([np.array([x,y]) for x,y in zip(j[0],j[1])])

	batch_x =  np.array(batch_x)
	#print batch_x
	batch_x.reshape(128,3,2)
	batch_y =  np.array(batch_y)
	#print batch_y.shape
	batch_y.reshape(128,4)'''


	#batch_x = tf.convert_to_tensor(batch_x)
	#batch_y =  tf.convert_to_tensor(batch_y)


	if i % 25 == 0:
		#s = sess.run(merged_summary,feed_dict = {x: batch_x,y: batch_y})
		#writer.add_summary(s,i)

		#k = sess.run(merged_summary,feed_dict = {x: mnist.test.images,y: mnist.test.labels})

		#writer2.add_summary(k,i)


		#train_accuracy = sess.run(accuracy.eval(feed_dict={x: batch[0], y: batch[1]}))
		[train_accuracy] = sess.run([accuracy],feed_dict = {x: batch_x, y:batch_y})
		#logits = sess.run([accuracy],feed_dict = {x: batch_x, y:batch_y})
		print('step %d, training accuracy %g' % (i, train_accuracy))
		#[test_acc] = sess.run([test_accuracy],feed_dict = {x: mnist.test.images, y:mnist.test.labels})
		#print('step %d, test accuracy %g' % (i, test_acc))

		#saver.restore(sess, "/home/psycholearner/projects//DL4CV-EE6132-/PA2/model.ckpt")
	
	sess.run(train_step,feed_dict = {x:batch_x,y:batch_y})

'''
test_data = mnist.test.images[:128].reshape((-1, 28, 28))
test_label = mnist.test.labels[:128]
print("Testing Accuracy:",sess.run([accuracy], feed_dict={x: test_data, y: test_label}))
'''

