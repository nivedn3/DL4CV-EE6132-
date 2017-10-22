import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32,shape = [None,28,28])
y = tf.placeholder(tf.float32,shape = [None,10])

#considering forward and backward cells
w = tf.Variable(tf.random_normal([2*128,10]))
b = tf.Variable(tf.random_normal([10]))

rnn_inp = tf.unstack(x,28,1)

lstmf = tf.contrib.rnn.BasicLSTMCell(128)

lstmb = tf.contrib.rnn.BasicLSTMCell(128)

outputs, state1,state2 = tf.contrib.rnn.static_bidirectional_rnn(lstmf,lstmb, rnn_inp, dtype=tf.float32)

logits = tf.matmul(outputs[-1], w) + b

logits = tf.nn.softmax(logits)


#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
#rain_op = optimizer.minimize(loss_op)
#Theres a difference in the logits and prediction address that 
regularizer = tf.nn.l2_loss(w)

with tf.name_scope("cross_entropy"):
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = y))
  cross_entropy = tf.reduce_mean(cross_entropy + 0.01 * regularizer)
  tf.summary.scalar('cross entropy',cross_entropy)

with tf.name_scope("train"):
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


with tf.name_scope("accuracy"):
  correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
  tf.summary.scalar('accuracy',accuracy)



merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("/home/psycholearner/projects/DL4CV-EE6132-/PA3/11")
writer.add_graph(sess.graph)

writer2 = tf.summary.FileWriter("/home/psycholearner/projects/DL4CV-EE6132-/PA3/12")
writer2.add_graph(sess.graph)


sess.run(tf.global_variables_initializer())


for i in range(10000):
  batch_x,batch_y = mnist.train.next_batch(128)

  
  batch_x = batch_x.reshape((128, 28, 28))

  if i % 200 == 0:

    s = sess.run(merged_summary,feed_dict = {x: batch_x,y: batch_y})
    writer.add_summary(s,i)

    validate_data = mnist.test.images[:500].reshape((-1, 28, 28))
    validate_label = mnist.test.labels[:500]

    k = sess.run(merged_summary,feed_dict = {x: validate_data,y: validate_label})

    writer2.add_summary(k,i)

    
    #train_accuracy = sess.run(accuracy.eval(feed_dict={x: batch[0], y: batch[1]}))
    [train_accuracy] = sess.run([accuracy],feed_dict = {x: batch_x, y:batch_y})
    print('step %d, training accuracy %g' % (i, train_accuracy))
    #[test_acc] = sess.run([test_accuracy],feed_dict = {x: mnist.test.images, y:mnist.test.labels})
    #print('step %d, test accuracy %g' % (i, test_acc))

  #saver.restore(sess, "/home/psycholearner/projects//DL4CV-EE6132-/PA2/model.ckpt")
  sess.run(train_step,feed_dict = {x: batch_x,y:batch_y})




test_data = mnist.test.images[:10000].reshape((-1, 28, 28))
test_label = mnist.test.labels[:10000]
print("Testing Accuracy:",sess.run([accuracy], feed_dict={x: test_data, y: test_label}))