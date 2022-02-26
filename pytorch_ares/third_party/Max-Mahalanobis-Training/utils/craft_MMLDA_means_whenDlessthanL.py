import tensorflow as tf

L = 10 #Number of classes
d = 2 #Dimension of features
lr = 0.0001 #Learning rate
mean_var = 1
steps = 10000 #optimization steps

z = tf.get_variable("auxiliary_variable", [d, L]) #dxL
x = z / tf.norm(z, axis=0, keepdims=True) #dxL, normalized in each column
XTX = tf.matmul(x, x, transpose_a=True) - 2 * tf.eye(L)#LxL, each element is the dot-product of two means, the diag elements are -1
cost = tf.reduce_max(XTX) #single element
opt = tf.train.AdamOptimizer(learning_rate=lr)
opt_op = opt.minimize(cost)
with tf.Session() as sess:
	sess.run(tf.initializers.global_variables())
	for i in range(steps):
		_, loss = sess.run([opt_op, cost])
		min_distance2 = loss
		print('Step %d, min_distance2: %f'%(i, min_distance2))

	mean_logits = sess.run(x)

mean_logits = mean_var * mean_logits.T 
import scipy.io as sio
sio.savemat('/MMC/kernel_paras/meanvar1_featuredim'+str(d)+'_class'+str(L)+'.mat', {'mean_logits': mean_logits})
