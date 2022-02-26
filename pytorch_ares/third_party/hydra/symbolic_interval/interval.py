'''
Interval class definitions
** Top contributor: Shiqi Wang
** This file is part of the symbolic interval analysis library.
** Copyright (c) 2018-2019 by the authors listed in the file LICENSE
** and their institutional affiliations.
** All rights reserved.
'''

from __future__ import print_function

import numpy as np
import torch
import warnings


class Interval():
	'''Naive interval class

	Naive interval propagation is low-cost (only around two times slower 
	than regular NN propagation). However, the output range provided is 
	loose. This is because the dependency of inputs are ignored.
	See ReluVal https://arxiv.org/abs/1804.10829 for more details of
	the tradeoff.

	Naive interval propagation are used for many existing training
	schemes:
	(1) DiffAi: http://proceedings.mlr.press/v80/mirman18b/mirman18b.pdf
	(2) IBP: https://arxiv.org/pdf/1810.12715.pdf
	These training schemes are fast but the robustness of trained models
	suffers from the loose estimations of naive interval propagation.
	
	Args:
		lower: numpy matrix of the lower bound for each layer nodes
		upper: numpy matrix of the upper bound for each layer nodes
		lower and upper should have the same shape of input for 
		each layer
		no upper value should be less than corresponding lower value

	* :attr:`l` and `u` keeps the upper and lower values of the
	  interval. Naive interval propagation using them to propagate.

	* :attr:`c` and `e` means the center point and the error range 
	  of the interval. Symbolic interval propagation using to propagate
	  since it can keep the dependency more efficiently. 

	* :attr:`mask` is used to keep the estimation information for each
	  hidden node. It has the same shape of the ReLU layer input. 
	  for each hidden node, before going through ReLU, let [l,u] denote
	  a ReLU's input range. It saves the value u/(u-l), which is the
	  slope of estimated output dependency. 0 means, given the input
	  range, this ReLU's input will always be negative and the output 
	  is always 0. 1 indicates, it always stays positive and the
	  output will not change. Otherwise, this node is estimated during 
	  interval propagation and will introduce overestimation error. 
	'''
	def __init__(self, lower, upper, use_cuda=False):
		if(not isinstance(self, Inverse_interval)):
			assert not ((upper-lower)<0).any(), "upper less than lower"
		self.l = lower
		self.u = upper
		self.c = (lower+upper)/2
		self.e = (upper-lower)/2
		self.mask = []
		self.use_cuda = use_cuda


	def update_lu(self, lower, upper):
		'''Update this interval with new lower and upper numpy matrix

		Args:
			lower: numpy matrix of the lower bound for each layer nodes
			upper: numpy matrix of the upper bound for each layer nodes
		'''
		# if(not isinstance(self, Inverse_interval)):
			# assert not ((upper-lower)<0).any(), "upper less than lower"
		self.l = lower
		self.u = upper
		self.c = (lower+upper)/2
		self.e = (upper-lower)/2


	def update_ce(self, center, error):
		'''Update this interval with new error and center numpy matrix

		Args:
			lower: numpy matrix of the lower bound for each layer nodes
			upper: numpy matrix of the upper bound for each layer nodes
		'''
		# if(not isinstance(self, Inverse_interval)):
		# 	assert not (error<0).any(), "upper less than lower"
		self.c = center
		self.e = error
		self.u = self.c+self.e
		self.l = self.c-self.e


	def __str__(self):
		'''Print function
		'''
		string = "interval shape:"+str(self.c.shape)
		string += "\nlower:"+str(self.l)
		string += "\nupper:"+str(self.u)
		return string
	

	def worst_case(self, y, output_size):
		'''Calculate the wrost case of the analyzed output ranges.
		In details, it returns the upper bound of other label minus 
		the lower bound of the target label. If the returned value is 
		less than 0, it means the worst case provided by interval
		analysis will never be larger than the target label y's. 
		'''
		assert y.shape[0] == self.l.shape[0] == self.u.shape[0],\
				"wrong input shape"
		
		for i in range(y.shape[0]):
			t = self.l[i, y[i]]
			self.u[i] = self.u[i]-t
			self.u[i, y[i]] = 0.0
		return self.u


class Inverse_interval(Interval):
	def __init__(self, lower, upper, use_cuda=False):
		assert lower.shape[0]==upper.shape[0], "each symbolic"+\
					"should have the same shape"
		
		Interval.__init__(self, lower, upper)
		self.use_cuda = use_cuda
		self.shape = list(self.c.shape[1:])
		self.n = list(self.c[0].reshape(-1).size())[0]
		self.input_size = self.n
		self.batch_size = self.c.shape[0]

	def worst_case(self, y, output_size):
		assert y.shape[0] == self.l.shape[0] == self.u.shape[0],\
				"wrong input shape"
		'''Taking the norm of the inverse interval for the worst case
		'''
		u = self.c.abs()+self.e.abs()
		return u


class Symbolic_interval(Interval):
	'''Symbolic interval class

	Symbolic interval analysis is a state-of-the-art tight output range 
	analyze method. It captured the dependencies ignored by naive
	interval propagation. As the tradeoff, the cost is much higher than
	naive interval and regular propagations. To maximize the tightness,
	symbolic linear relaxation is used. More details can be found in 
	Neurify: https://arxiv.org/pdf/1809.08098.pdf

	There are several similar methods which can provide close tightness
	(1) Convex polytope: https://arxiv.org/abs/1711.00851
	(2) FastLin: https://arxiv.org/abs/1804.09699
	(3) DeepZ: https://files.sri.inf.ethz.ch/website/papers/DeepZ.pdf
	This lib implements symbolic interval analysis, which can provide
	one of the tightest and most efficient analysis among all these 
	methods.

	Symbolic interval analysis is used to verifiably robust train the
	networks in MixTrain, providing state-of-the-art efficiency and 
	verifiable robustness. See https://arxiv.org/abs/1811.02625 for more
	details.
	Similar training methods include:
	(1) Scaling defense: https://arxiv.org/abs/1805.12514
	(2) DiffAI: http://proceedings.mlr.press/v80/mirman18b/mirman18b.pdf
	
	* :attr:`shape` is the input shape of ReLU layers.

	* :attr:`n` is the number of hidden nodes in each layer.

	* :attr:`idep` keeps the input dependencies.

	* :attr:`edep` keeps the error dependency introduced by each
	  overestimated nodes.
	'''
	def __init__(self, lower, upper, epsilon=0, norm="linf", use_cuda=False):
		assert lower.shape[0]==upper.shape[0], "each symbolic"+\
					"should have the same shape"
		
		Interval.__init__(self, lower, upper)
		self.use_cuda = use_cuda
		self.shape = list(self.c.shape[1:])
		self.n = list(self.c[0].reshape(-1).size())[0]
		self.input_size = self.n
		self.batch_size = self.c.shape[0]
		self.epsilon = epsilon
		self.norm = norm
		if(self.use_cuda):
			self.idep = torch.eye(self.n, device=\
					self.c.get_device()).unsqueeze(0)
		else:
			self.idep = torch.eye(self.n).unsqueeze(0)
		self.edep = []
		self.edep_ind = []
		
		

	'''Calculating the upper and lower matrix for symbolic intervals.
	To make concretize easier, convolutional layer nodes will be 
	extended first.
	'''
	def concretize(self):
		self.extend()
		if self.norm=="linf":

			e = (self.idep*self.e.view(self.batch_size,\
					self.input_size, 1)).abs().sum(dim=1)

		elif self.norm == "l2":
			# idep = (self.idep*self.idep)\
			#  			.sum(dim=1, keepdim=False).sqrt()
			idep = torch.norm(self.idep, dim=1, keepdim=False)

			e = idep*self.epsilon

		elif self.norm == "l1":
			idep = self.idep.abs().max(dim=1, keepdim=False)[0]

			e = idep*self.epsilon

		if self.edep:
			#print("sym e1", e)
			for i in range(len(self.edep)):
				e = e + self.edep_ind[i].t().mm(self.edep[i].abs())
			#print("sym e2", e)

		self.l = self.c - e
		self.u = self.c + e

		return self


	'''Extending convolutional layer nodes to a two-dimensional vector.
	'''
	def extend(self):
		self.c = self.c.reshape(self.batch_size, self.n)
		self.idep = self.idep.reshape(-1, self.input_size, self.n)

		for i in range(len(self.edep)):
			self.edep[i] = self.edep[i].reshape(-1, self.n)


	'''Convert the extended layer back to the shape stored in `shape`.
	'''
	def shrink(self):
		self.c = self.c.reshape(tuple([-1]+self.shape))
		self.idep = self.idep.reshape(tuple([-1]+self.shape))

		for i in range(len(self.edep)):
			self.edep[i] = self.edep[i].reshape(\
				tuple([-1]+self.shape))


	'''Calculate the wrost case of the analyzed output ranges.
	Return the upper bound of other output dependency minus target's
	output dependency. If the returned value is less than 0, it means
	the worst case provided by interval analysis will never be larger
	than the target label y's. 
	'''
	def worst_case(self, y, output_size):

		assert y.shape[0] == self.l.shape[0] == self.batch_size,\
								"wrong label shape"
		if(self.use_cuda):
			kk = torch.eye(output_size, dtype=torch.uint8,\
				requires_grad=False, device=y.get_device())[y]
		else:
			kk = torch.eye(output_size, dtype=torch.uint8,\
					requires_grad=False)[y]

		c_t = self.c.masked_select(kk).unsqueeze(1)
		self.c = self.c - c_t

		idep_t = self.idep.masked_select(\
					kk.view(self.batch_size,1,output_size)).\
					view(self.batch_size, self.input_size,1)
		self.idep = self.idep-idep_t

		for i in range(len(self.edep)):
			edep_t = self.edep[i].masked_select((self.edep_ind[i].\
						mm(kk.type_as(self.edep_ind[i]))).type_as(kk)).\
						view(-1,1)
			self.edep[i] = self.edep[i]-edep_t

		self.concretize()

		return self.u


class mix_interval(Symbolic_interval):
	
	def __init__(self, lower, upper, epsilon=0, norm="linf", use_cuda=False):
		assert lower.shape[0]==upper.shape[0], "each symbolic"+\
					"should have the same shape"
		
		Symbolic_interval.__init__(self, lower, upper)
		self.use_cuda = use_cuda
		self.shape = list(self.c.shape[1:])
		self.n = list(self.c[0].reshape(-1).size())[0]
		self.input_size = self.n
		self.batch_size = self.c.shape[0]
		self.epsilon = epsilon
		self.norm = norm
		if(self.use_cuda):
			self.idep = torch.eye(self.n, device=\
					self.c.get_device()).unsqueeze(0)
		else:
			self.idep = torch.eye(self.n).unsqueeze(0)
		self.edep = []
		self.edep_ind = []
		self.nl = self.l.clone().detach()
		self.nu = self.u.clone().detach()
		self.nc = self.c.clone().detach()
		self.ne = self.e.clone().detach()
		
		

	'''Calculating the upper and lower matrix for symbolic intervals.
	To make concretize easier, convolutional layer nodes will be 
	extended first.
	'''
	def concretize(self):
		self.extend()
		if self.norm=="linf":

			e = (self.idep*self.e.view(self.batch_size,\
					self.input_size, 1)).abs().sum(dim=1)

		elif self.norm == "l2":
			# idep = (self.idep*self.idep)\
			#  			.sum(dim=1, keepdim=False).sqrt()
			idep = torch.norm(self.idep, dim=1, keepdim=False)

			e = idep*self.epsilon

		elif self.norm == "l1":
			idep = self.idep.abs().max(dim=1, keepdim=False)[0]

			e = idep*self.epsilon

		if self.edep:
			#print("sym e1", e)
			for i in range(len(self.edep)):
				e = e + self.edep_ind[i].t().mm(self.edep[i].abs())
			#print("sym e2", e)

		self.l = self.c - e
		self.u = self.c + e

		self.l = torch.where(self.l>self.nl, self.l, self.nl)
		self.u = torch.where(self.u<self.nu, self.u, self.nu)

		return self


	'''Extending convolutional layer nodes to a two-dimensional vector.
	'''
	def extend(self):
		self.c = self.c.reshape(self.batch_size, self.n)
		self.idep = self.idep.reshape(-1, self.input_size, self.n)

		for i in range(len(self.edep)):
			self.edep[i] = self.edep[i].reshape(-1, self.n)

		self.nc = self.nc.reshape(self.batch_size, self.n)
		self.ne = self.ne.reshape(self.batch_size, self.n)
		self.nl = self.nl.reshape(self.batch_size, self.n)
		self.nu = self.nu.reshape(self.batch_size, self.n)


	'''Convert the extended layer back to the shape stored in `shape`.
	'''
	def shrink(self):
		self.c = self.c.reshape(tuple([-1]+self.shape))
		self.idep = self.idep.reshape(tuple([-1]+self.shape))

		for i in range(len(self.edep)):
			self.edep[i] = self.edep[i].reshape(\
				tuple([-1]+self.shape))

		self.nc = self.nc.reshape(tuple([-1]+self.shape))
		self.ne = self.ne.reshape(tuple([-1]+self.shape))
		self.nl = self.nl.reshape(tuple([-1]+self.shape))
		self.nu = self.nu.reshape(tuple([-1]+self.shape))

	'''Calculate the wrost case of the analyzed output ranges.
	Return the upper bound of other output dependency minus target's
	output dependency. If the returned value is less than 0, it means
	the worst case provided by interval analysis will never be larger
	than the target label y's. 
	'''
	def worst_case(self, y, output_size):

		assert y.shape[0] == self.l.shape[0] == self.batch_size,\
								"wrong label shape"
		if(self.use_cuda):
			kk = torch.eye(output_size, dtype=torch.uint8,\
				requires_grad=False, device=y.get_device())[y]
		else:
			kk = torch.eye(output_size, dtype=torch.uint8,\
					requires_grad=False)[y]

		c_t = self.c.masked_select(kk).unsqueeze(1)
		self.c = self.c - c_t

		idep_t = self.idep.masked_select(\
					kk.view(self.batch_size,1,output_size)).\
					view(self.batch_size, self.input_size,1)
		self.idep = self.idep-idep_t

		for i in range(len(self.edep)):
			edep_t = self.edep[i].masked_select((self.edep_ind[i].\
						mm(kk.type_as(self.edep_ind[i]))).type_as(kk)).\
						view(-1,1)
			self.edep[i] = self.edep[i]-edep_t

		self.concretize()

		return self.u 




class Center_symbolic_interval(Interval):
	def __init__(self, lower, upper, use_cuda=False):
		assert lower.shape[0]==upper.shape[0], "each symbolic"+\
					"should have the same shape"
		
		Interval.__init__(self, lower, upper)
		self.use_cuda = use_cuda
		self.shape = list(self.c.shape[1:])
		self.n = list(self.c[0].reshape(-1).size())[0]
		self.input_size = self.n
		self.batch_size = self.c.shape[0]
		if(self.use_cuda):
			self.idep = torch.eye(self.n, device=\
					self.c.get_device()).unsqueeze(0)
		else:
			self.idep = torch.eye(self.n).unsqueeze(0)
		

	'''Calculating the upper and lower matrix for symbolic intervals.
	To make concretize easier, convolutional layer nodes will be 
	extended first.
	'''
	def concretize(self):
		self.extend()
		e = (self.idep*self.e.view(self.batch_size,\
					self.input_size, 1)).abs().sum(dim=1)

		self.l = self.c - e
		self.u = self.c + e

		return self


	'''Extending convolutional layer nodes to a two-dimensional vector.
	'''
	def extend(self):
		self.c = self.c.reshape(self.batch_size, self.n)
		self.idep = self.idep.reshape(-1, self.input_size, self.n)


	'''Convert the extended layer back to the shape stored in `shape`.
	'''
	def shrink(self):
		self.c = self.c.reshape(tuple([-1]+self.shape))
		self.idep = self.idep.reshape(tuple([-1]+self.shape))


	'''Calculate the wrost case of the analyzed output ranges.
	Return the upper bound of other output dependency minus target's
	output dependency. If the returned value is less than 0, it means
	the worst case provided by interval analysis will never be larger
	than the target label y's. 
	'''
	def worst_case(self, y, output_size):

		assert y.shape[0] == self.l.shape[0] == self.batch_size,\
								"wrong label shape"
		if(self.use_cuda):
			kk = torch.eye(output_size, dtype=torch.uint8,\
				requires_grad=False, device=y.get_device())[y]
		else:
			kk = torch.eye(output_size, dtype=torch.uint8,\
					requires_grad=False)[y]

		c_t = self.c.masked_select(kk).unsqueeze(1)
		self.c = self.c - c_t

		idep_t = self.idep.masked_select(\
					kk.view(self.batch_size,1,output_size)).\
					view(self.batch_size, self.input_size,1)
		self.idep = self.idep-idep_t

		self.concretize()

		return self.u


		
class Symbolic_interval_proj1(Interval):
	'''
	* :attr:`shape` is the input shape of ReLU layers.

	* :attr:`n` is the number of hidden nodes in each layer.

	* :attr:`idep` keeps the input dependencies.

	* :attr:`edep` keeps the error dependency introduced by each
	  overestimated nodes.
	'''
	def __init__(self, lower, upper, proj=None, proj_ind=None, use_cuda=False):
		assert lower.shape[0]==upper.shape[0], "each symbolic"+\
					"should have the same shape"
		
		Interval.__init__(self, lower, upper)
		self.use_cuda = use_cuda
		self.shape = list(self.c.shape[1:])
		self.n = list(self.c[0].reshape(-1).size())[0]
		self.input_size = self.n
		self.batch_size = self.c.shape[0]
		if(self.use_cuda):
			self.idep = torch.eye(self.n, device=\
					self.c.get_device())
		else:
			self.idep = torch.eye(self.n)

		self.edep = []
		self.edep_ind = []

		self.proj_ind = proj_ind

		if(proj>self.input_size):
			warnings.warn("proj is larger than input size")
			self.proj = self.input_size
		else:
			self.proj = proj
		
		if(proj_ind is None):
			idep_ind = np.arange(self.proj)
			proj_ind = np.arange(self.proj, self.input_size)

			self.idep_proj = self.idep[proj_ind].sum(dim=0).unsqueeze(0)
			self.idep = self.idep[idep_ind].unsqueeze(0)
			
			self.idep_proj = self.idep_proj*self.e.\
					view(self.batch_size, self.input_size)
			self.e = self.e.view(self.batch_size, self.input_size)[:, idep_ind]
		else:
			self.idep = self.idep.unsqueeze(0)*\
						self.e.view(self.batch_size,1,self.n)
			#print(self.idep.shape, proj_ind.shape)
			self.idep = self.idep.gather(index=proj_ind.\
						unsqueeze(-1).repeat(1,1,self.n), dim=1)
			#print(self.idep.shape)

			self.idep_proj = (self.idep.sum(dim=1)==0).type_as(self.idep)
			self.idep_proj = self.idep_proj*\
					self.e.view(self.batch_size, self.input_size)
			#print("proj",self.idep_proj.shape)
			

	'''Calculating the upper and lower matrix for symbolic intervals.
	To make concretize easier, convolutional layer nodes will be 
	extended first.
	'''
	def concretize(self):
		self.extend()

		if(self.proj_ind is None):
			e = (self.idep*self.e.view(self.batch_size,\
				self.proj, 1)).abs().sum(dim=1)
		else:
			e = self.idep.abs().sum(dim=1)
		#print("e1", e)
		e = e + self.idep_proj.abs()
		#print("e2", e)
		if(self.edep):
			for i in range(len(self.edep)):
				e = e + self.edep_ind[i].t().mm(self.edep[i].abs())
		#print("e3", e)

		self.l = self.c - e
		self.u = self.c + e

		return self


	'''Extending convolutional layer nodes to a two-dimensional vector.
	'''
	def extend(self):
		self.c = self.c.reshape(self.batch_size, self.n)
		self.idep = self.idep.reshape(-1, self.proj, self.n)
		self.idep_proj = self.idep_proj.reshape(-1, self.n)

		for i in range(len(self.edep)):
			self.edep[i] = self.edep[i].reshape(-1, self.n)


	'''Convert the extended layer back to the shape stored in `shape`.
	'''
	def shrink(self):
		self.c = self.c.reshape(tuple([-1]+self.shape))
		self.idep = self.idep.reshape(tuple([-1]+self.shape))
		self.idep_proj = self.idep_proj.view(tuple([self.batch_size]+self.shape))

		for i in range(len(self.edep)):
			self.edep[i] = self.edep[i].reshape(\
				tuple([-1]+self.shape))


	'''Calculate the wrost case of the analyzed output ranges.
	Return the upper bound of other output dependency minus target's
	output dependency. If the returned value is less than 0, it means
	the worst case provided by interval analysis will never be larger
	than the target label y's. 
	'''
	def worst_case(self, y, output_size):

		assert y.shape[0] == self.l.shape[0] == self.batch_size,\
								"wrong label shape"
		if(self.use_cuda):
			kk = torch.eye(output_size, dtype=torch.uint8,\
				requires_grad=False, device=y.get_device())[y]
		else:
			kk = torch.eye(output_size, dtype=torch.uint8,\
					requires_grad=False)[y]

		c_t = self.c.masked_select(kk).unsqueeze(1)
		self.c = self.c - c_t

		idep_t = self.idep.masked_select(\
					kk.view(self.batch_size,1,output_size)).\
					view(self.batch_size, self.proj,1)
		self.idep = self.idep-idep_t

		idep_proj_t = self.idep_proj.masked_select(kk)
		self.idep_proj = self.idep_proj+idep_proj_t.view(-1,1)
		self.idep_proj = self.idep_proj*(1-kk).type_as(self.idep_proj)

		for i in range(len(self.edep)):
			edep_t = self.edep[i].masked_select((self.edep_ind[i].\
						mm(kk.type_as(self.edep_ind[i]))).type_as(kk)).\
						view(-1,1)
			self.edep[i] = self.edep[i]-edep_t

		self.concretize()

		return self.u 


class Symbolic_interval_proj2(Interval):
	'''
	* :attr:`shape` is the input shape of ReLU layers.

	* :attr:`n` is the number of hidden nodes in each layer.

	* :attr:`idep` keeps the input dependencies.

	* :attr:`edep` keeps the error dependency introduced by each
	  overestimated nodes.
	'''
	def __init__(self, lower, upper, proj=None,\
					proj_ind=None, use_cuda=False):
		assert lower.shape[0]==upper.shape[0], "each symbolic"+\
					"should have the same shape"
		
		Interval.__init__(self, lower, upper)
		self.use_cuda = use_cuda
		self.shape = list(self.c.shape[1:])
		self.n = list(self.c[0].reshape(-1).size())[0]
		self.input_size = self.n
		self.batch_size = self.c.shape[0]
		if(self.use_cuda):
			self.idep = torch.eye(self.n, device=\
					self.c.get_device())
		else:
			self.idep = torch.eye(self.n)

		self.edep = self.e.new_zeros(self.e.shape)

		self.proj_ind = proj_ind
		self.proj = proj

		if(proj_ind is None):
		
			idep_ind = np.arange(self.proj)
			proj_ind = np.arange(self.proj, self.input_size)

			self.idep_proj = self.idep[proj_ind].sum(dim=0).unsqueeze(0)
			self.idep = self.idep[idep_ind].unsqueeze(0)
			
			self.idep_proj = self.idep_proj*self.e.\
					view(self.batch_size, self.input_size)

			self.e = self.e.view(self.batch_size,\
							self.input_size)[:, idep_ind]
		else:
			self.idep = self.idep.unsqueeze(0)*\
						self.e.view(self.batch_size,1,self.n)
			#print(self.idep.shape, proj_ind.shape)
			self.idep = self.idep.gather(index=proj_ind.\
						unsqueeze(-1).repeat(1,1,self.n), dim=1)
			#print(self.idep.shape)

			self.idep_proj = (self.idep.sum(dim=1)==0).type_as(self.idep)
			self.idep_proj = self.idep_proj*\
					self.e.view(self.batch_size, self.input_size)
			#print("proj",self.idep_proj.shape)




	'''Calculating the upper and lower matrix for symbolic intervals.
	To make concretize easier, convolutional layer nodes will be 
	extended first.
	'''
	def concretize(self):
		self.extend()

		if(self.proj_ind is None):
			e = (self.idep*self.e.view(self.batch_size,\
				self.proj, 1)).abs().sum(dim=1)
		else:
			e = self.idep.abs().sum(dim=1)
		#print("e1", e)
		e = e + self.idep_proj.abs()
		#print("e2", e)
		e = e + self.edep.abs()
		#print("e3", e)

		self.l = self.c - e
		self.u = self.c + e

		return self


	'''Extending convolutional layer nodes to a two-dimensional vector.
	'''
	def extend(self):
		self.c = self.c.reshape(self.batch_size, self.n)
		self.idep = self.idep.reshape(-1, self.proj, self.n)
		self.idep_proj = self.idep_proj.reshape(-1, self.n)

		self.edep = self.edep.reshape(self.batch_size, self.n)


	'''Convert the extended layer back to the shape stored in `shape`.
	'''
	def shrink(self):
		self.c = self.c.reshape(tuple([-1]+self.shape))
		self.idep = self.idep.reshape(tuple([-1]+self.shape))
		self.idep_proj = self.idep_proj.view(tuple([self.batch_size]+self.shape))
		self.edep = self.edep.view(tuple([-1]+self.shape))


	'''Calculate the wrost case of the analyzed output ranges.
	Return the upper bound of other output dependency minus target's
	output dependency. If the returned value is less than 0, it means
	the worst case provided by interval analysis will never be larger
	than the target label y's. 
	'''
	def worst_case(self, y, output_size):

		assert y.shape[0] == self.l.shape[0] == self.batch_size,\
								"wrong label shape"
		if(self.use_cuda):
			kk = torch.eye(output_size, dtype=torch.uint8,\
				requires_grad=False, device=y.get_device())[y]
		else:
			kk = torch.eye(output_size, dtype=torch.uint8,\
					requires_grad=False)[y]

		c_t = self.c.masked_select(kk).unsqueeze(1)
		self.c = self.c - c_t

		idep_t = self.idep.masked_select(\
					kk.view(self.batch_size,1,output_size)).\
					view(self.batch_size, self.proj,1)
		self.idep = self.idep-idep_t

		idep_proj_t = self.idep_proj.masked_select(kk)
		self.idep_proj = self.idep_proj+idep_proj_t.view(-1,1)
		self.idep_proj = self.idep_proj*(1-kk).type_as(self.idep_proj)

		edep_t = self.edep.masked_select(kk)
		self.edep = self.edep+edep_t.view(-1,1)
		self.edep = self.edep*(1-kk).type_as(self.edep)

		self.concretize()

		return self.u 



class gen_sym(Symbolic_interval):
	
	def __init__(self, lower, upper, epsilon=[0, 0, 0], norm=["linf", "l2", "l1"], use_cuda=False):
		
		Symbolic_interval.__init__(self, lower, upper, epsilon, norm, use_cuda)
		self.use_cuda = use_cuda
		self.shape = list(self.c.shape[1:])
		self.n = list(self.c[0].reshape(-1).size())[0]
		self.input_size = self.n
		self.batch_size = self.c.shape[0]
		self.epsilon = epsilon
		self.norm = norm
		if(self.use_cuda):
			self.idep = torch.eye(self.n, device=\
					self.c.get_device()).unsqueeze(0)
		else:
			self.idep = torch.eye(self.n).unsqueeze(0)
		self.edep = []
		self.edep_ind = []
		

	def concretize(self):
		self.extend()
		e = None
		for i in range(len(self.norm)):

			if self.norm[i] == "linf":
				e0 = (self.idep*self.e.view(self.batch_size,\
						self.input_size, 1)).abs().sum(dim=1)

			elif self.norm[i] == "l2":
				idep = torch.norm(self.idep, dim=1, keepdim=False)

				e0 = idep*self.epsilon[i]

			elif self.norm[i] == "l1":
				idep = self.idep.abs().max(dim=1, keepdim=False)[0]

				e0 = idep*self.epsilon[i]
			
			if e is None:
				e = e0
			else:
				e = torch.where(e>e0, e, e0)

		if self.edep:
			#print("sym e1", e)
			for i in range(len(self.edep)):
				e = e + self.edep_ind[i].t().mm(self.edep[i].abs())
			#print("sym e2", e)

		self.l = self.c - e
		self.u = self.c + e

		return self
