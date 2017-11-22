import tensorflow as tf
from tensorflow import matmul
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math
sess = tf.InteractiveSession()

class DataDistribution(object):
    def __init__(self):
        self.mu = 2
        self.sigma = 0.2

    def sample(self, N):

        samples = np.random.normal(self.mu, self.sigma,N)
       	samples.sort()
        return samples

class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        return np.linspace(-self.range, self.range, N) + \
            np.random.random(N) * 0.01


def weight(shape):

	values = tf.truncated_normal(shape,stddev = 0.1)
	return tf.Variable(values)

G0 = weight([1,10])
G1 = weight([10,1])
Gb0 = weight([10])
Gb1 = weight([1])

W0 = weight([1,20]) 
W1 = weight([20,20])
W2 = weight([20,20])
W3 = weight([20,1])
Wb0 = weight([20])
Wb1 = weight([20])
Wb2 = weight([20])
Wb3 = weight([1])

def generator(input):	
    h0 = tf.nn.softplus(matmul(input, G0) + Gb0)
    h1 = matmul(h0,G1) + Gb1
    return h1

def discriminator(input):
    h0 = tf.tanh(matmul(input, W0) + Wb0)
    h1 = tf.tanh(matmul(h0, W1) + Wb1)
    h2 = tf.tanh(matmul(h1, W2) + Wb2)
    h3 = tf.sigmoid(matmul(h2,W3) + Wb3)
    return h3

def optimizer(loss, var_list):
    initial_learning_rate = 0.005
    decay = 0.95
    num_decay_steps = 150
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase=True
    )
    optimizer = tf.train.AdamOptimizer().minimize(
        loss,
        global_step=batch,
        var_list=var_list
    )
    return optimizer

vars = tf.trainable_variables()
d_params = [W0,W1,W2,W3,Wb2,Wb3,Wb0,Wb1]
g_params = [Gb1,Gb0,G1,G0]


with tf.variable_scope('G'):
    z1 = tf.placeholder(tf.float32, shape=(None, 1))
    G = generator(z1)


with tf.variable_scope('D') as scope:
    x1 = tf.placeholder(tf.float32, shape=(None, 1))
    D1 = discriminator(x1)
    scope.reuse_variables()
    D2 = discriminator(G)


loss_d = tf.reduce_mean(-tf.log(D1) - tf.log(1 - D2))
loss_g = tf.reduce_mean(-tf.log(D2))


opt_d = optimizer(loss_d, d_params)
opt_g = optimizer(loss_g, g_params)

sess.run(tf.global_variables_initializer())

for step in xrange(1000)	:

    # update discriminator

    batch_size = 50
    data = DataDistribution()
    x = data.sample(batch_size)
    x = np.reshape(x, (batch_size, 1))
    gen = GeneratorDistribution(1)
    z = gen.sample(batch_size)
    sess.run([loss_d, opt_d], feed_dict={
            x1: x,
            z1: np.reshape(z, (batch_size, 1))
    })

    # update generator
    z = gen.sample(batch_size)
    sess.run([loss_g, opt_g], {
        z1: np.reshape(z, (batch_size, 1))
    })
    g = sess.run(G,feed_dict = {x1:x,z1:np.reshape(z,(batch_size,1))})


mu = 2
variance = 0.2
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x,mlab.normpdf(x, mu, sigma))

hist,bins=np.histogram(g,bins=20)
width=(bins[1]-bins[0])
center=(bins[:-1]+bins[1:])
plt.bar(center,hist,align='center',width=width)
#plt.hist(g,bins=40,normed=True,)
#plt.plot(b)
print g

plt.show()