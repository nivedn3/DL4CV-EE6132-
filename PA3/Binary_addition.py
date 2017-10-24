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
		#a_in = [1,0,0,0,0]
		b_in = np.random.choice([0,1],size)
		b_in = b_in.tolist()
		#b_in = [1,0,0,0,0]
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

size = 5
hs = 5

x = tf.placeholder(tf.float32,shape = [None,size,2])
y = tf.placeholder(tf.float32,shape = [None,size+1]) 


w = tf.Variable(tf.random_normal([hs,size+1]))
b = tf.Variable(tf.random_normal([size+1]))

rnn_inp = tf.unstack(x,size,1)

lstm = tf.contrib.rnn.BasicRNNCell(hs)

outputs, states = tf.contrib.rnn.static_rnn(lstm, rnn_inp, dtype=tf.float32)

logits = tf.sigmoid(tf.matmul(outputs[-1], w) + b)

logitst = tf.add(logits,tf.scalar_mul(-0.5,tf.ones_like(logits)))

logitst = tf.nn.relu(logits)

logitst = tf.scalar_mul(1000000,logits)

logitst = tf.clip_by_value(logits,0,1)

logitsc = tf.cast(logitst,tf.int32)

yc = tf.cast(y,tf.int32)

with tf.name_scope("cross_entropy"):
  #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = y))
  cross_entropy =  tf.losses.mean_squared_error(labels = y, predictions = logits)
  tf.summary.scalar('cross entropy',cross_entropy)

with tf.name_scope("train"):
  train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)


with tf.name_scope("accuracy"):
  correct_prediction = tf.equal(logitsc,yc)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
  tf.summary.scalar('accuracy',accuracy)


merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("/home/psycholearner/projects//DL4CV-EE6132-/PA3/200")
writer.add_graph(sess.graph)


writer2 = tf.summary.FileWriter("/home/psycholearner/projects/DL4CV-EE6132-/PA3/201")
writer2.add_graph(sess.graph)

sess.run(tf.global_variables_initializer())

for i in range(20000):

	a,batch_y = data(500,5)

	batch_x = np.array(a)
	batch_x = batch_x.reshape(500,5,2)
	batch_x = [j[::-1] for j in batch_x]
	batch_x = np.array(batch_x)
	batch_x.astype(float)
	batch_y = np.array(batch_y)
	#batch_y.astype(float)
	


	if i % 25 == 0:
		s = sess.run(merged_summary,feed_dict = {x: batch_x,y: batch_y})
		writer.add_summary(s,i)

		at,batch_yt = data(500,5)
		batch_xt = np.array(at)
		batch_xt = batch_xt.reshape(500,5,2)
		batch_xt = [j[::-1] for j in batch_xt]
		batch_xt = np.array(batch_xt)
		batch_xt.astype(float)
		batch_yt = np.array(batch_yt)
		k = sess.run(merged_summary,feed_dict = {x: batch_xt,y: batch_yt})

		
		writer2.add_summary(k,i)


		#train_accuracy = sess.run(accuracy.eval(feed_dict={x: batch[0], y: batch[1]}))
		#[train_accuracy] = sess.run([cross_entropy],feed_dict = {x: batch_x, y:batch_y})
		#[test] = sess.run([accuracy],feed_dict = {x: batch_x, y:batch_y})
		#logits = sess.run([accuracy],feed_dict = {x: batch_x, y:batch_y})
		#print('step %d, training accuracy %g %g' % (i, train_accuracy,test))
		#[test_acc] = sess.run([test_accuracy],feed_dict = {x: mnist.test.images, y:mnist.test.labels})
		#print('step %d, test accuracy %g' % (i, test_acc))

		#saver.restore(sess, "/home/psycholearner/projects//DL4CV-EE6132-/PA2/model.ckpt")
	sess.run(train_step,feed_dict = {x:batch_x,y:batch_y})

'''
test_data = mnist.test.images[:128].reshape((-1, 28, 28))
test_label = mnist.test.labels[:128]
print("Testing Accuracy:",sess.run([accuracy], feed_dict={x: test_data, y: test_label}))
'''
a,batch_y = data(500,5)
batch_x = np.array(a)
batch_x = batch_x.reshape(500,5,2)
batch_x = [j[::-1] for j in batch_x]
batch_x = np.array(batch_x)
batch_x.astype(float)
batch_y = np.array(batch_y)

print("Testing Accuracy:",sess.run([accuracy], feed_dict={x: batch_x, y: batch_y}))

