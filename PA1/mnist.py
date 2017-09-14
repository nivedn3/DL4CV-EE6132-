import numpy as np 
import random
class Network(object):
	def __init__(self,size):
		self.biases=[np.random.normal(0, 0.08,[y,1]) for y in size[1:]]
		self.weights=[np.random.normal(0, 0.08,[y,x]) for x,y in zip(size[:-1],size[1:])]
		self.num_layers=len(size)
## Calculating the  gradient descent
	def gradient_descend(self,training_data,test_data,epochs,mini_batch_length,eta):
		if test_data:
			n_test=len(test_data)
		n=len(training_data)
		print n
		test_loss = []
		train_loss = []
		count = 0
		for j in xrange(3):	
			random.shuffle(training_data)
			mini_batches=[training_data[w:w+mini_batch_length] for w in xrange(0,n,mini_batch_length)]
			for i in mini_batches:
				self.update(i,eta)

				count = count + 1

				if count % 50 == 0:
					if test_data:
						eval_test = self.evaluate(test_data)
						print "Test Epoch {0}: {1} / {2}".format(count,eval_test, n_test)
						test_loss.append(eval_test)
					else:
						print "Epoch {0} complete".format(count)
					eval_train = self.evaluate(training_data)
					print "Train Epoch {0}: {1} / {2}".format(count,eval_train, n)
					train_loss.append(eval_train)

		print "train-acc",train_loss
		print "\ntest-acc",test_loss
		l = test_data[:20]
		k = [net.feedforward(x.reshape(784,1)) for (x,y) in l]
		
		print k

		p = [y.reshape(10,1) for (x,y) in l]
		print p

# updating the parameters
	def update(self,mini_batches,eta):
		derb=[np.zeros(b.shape) for b in self.biases]
		derw=[np.zeros(w.shape) for w in self.weights]
		velow = [np.zeros(w.shape) for w in self.weights]
		velob = [np.zeros(b.shape) for b in self.biases]
		for i in mini_batches:
			del_w,	del_b = self.backprop(i[0].reshape(784,1),i[1].reshape(10,1))
			derw=[w+sw for w,sw in zip(derw,del_w)]
			derb=[b+sb for b,sb in zip(derb,del_b)]

		alpha = 0.9
		velow = [alpha*v - (eta/len(mini_batches))*p for v,p in zip(velow,derw)]

		self.weights = [(1-eta*0.05/len(mini_batches))*w + v for w,v in zip(self.weights,velow)]
		
		velob = [alpha*v - (eta/len(mini_batches))*c for v,c in zip(velob,derb)]

		self.biases = [b + v for b,v in zip(self.biases,velob)]
	def evaluate(self,t_data):
		
		k = [net.feedforward(x.reshape(784,1)) for (x,y) in t_data]
		j = []
		for i in k:
			j.append(np.argmax(i))
		pi = [y.reshape(10,1) for (x,y) in t_data]
		p = []
		for i in pi:
			p.append(np.argmax(i))
		c = 0
		for (i,j) in zip(j,p):
			if i==j:
				c = c+1
		return c
	def feedforward(self,a):
		i = 0
		for b,w in zip(self.biases,self.weights):
			if i < 3:
				a = sig(np.dot(w,a)+b)
				i = i + 1
			else:
				a = np.dot(w,a) + b
				return self.numeric_stability(a)
	def numeric_stability(self,a):
		max_a = np.max(a)
		exp_sum = np.sum(np.exp(a - max_a))
		a = np.exp(a - max_a)/exp_sum
		return a
	def backprop(self,i,j):	
		delta=[]
		z=[]
		del_a=[]
		del_b=[]
		a = np.array(i)
		count=0
		act = [i]
		k = 0
		for w,b in zip(self.weights,self.biases):	

			if k < 3:
				z.append(np.dot(w,a) + b)

				a = sig(z[count])
				count = count+1
				k = k + 1
				act.append(a)
			else:
				z.append(np.dot(w,a) + b)
				a = self.numeric_stability(z[count])
				act.append(a)

		deltaL=(-j + a)

		delta.append(deltaL)

		del_a.append(np.dot(deltaL,np.transpose(act[-2])))

		for l in range(self.num_layers-2):
			zeta=(sig_prime(z[-l-2])*np.dot(np.transpose(self.weights[-l-1]),delta[-1]))
			
			delta.append(zeta)
			del_a.append(np.dot(delta[-1],np.transpose(act[-l-3])))
		del_b=delta[::-1]
		del_a=del_a[::-1]
		return del_a,del_b
		

def sig(n):
	return 1.0/(1.0+np.exp(-n))

def sig_prime(n):
	return sig(n)*(1-sig(n))


if __name__=='__main__':

	def data_parser(x):
		temp = []
		for i in range(len(x[0])):
			k = [x[0][i],x[1][i]]
			temp.append(k)
		return temp

	import tensorflow as tf
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	training_data = mnist.train.next_batch(55000)
	test_data = mnist.test.next_batch(10000)
	train = data_parser(training_data)
	test = data_parser(test_data)
	print train[0][1]
	net=Network([784,1000,500,250,10])
	net.gradient_descend(train,test,3,64,0.01)
