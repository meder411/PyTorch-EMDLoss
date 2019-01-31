import torch
import torch.nn as nn

import _emd_ext._emd as emd


class EMDFunction(torch.autograd.Function):
	@staticmethod
	def forward(self, xyz1, xyz2):
		cost, match = emd.emd_forward(xyz1, xyz2)
		self.save_for_backward(xyz1, xyz2, match)
		return cost


	@staticmethod
	def backward(self, grad_output):
		xyz1, xyz2, match = self.saved_tensors
		grad_xyz1, grad_xyz2 = emd.emd_backward(xyz1, xyz2, match)
		return grad_xyz1, grad_xyz2




class EMDLoss(nn.Module):
	'''
	Computes the (approximate) Earth Mover's Distance between two point sets. 

	IMPLEMENTATION LIMITATIONS:
	- Double tensors must have <=11 dimensions
	- Float tensors must have <=23 dimensions
	This is due to the use of CUDA shared memory in the computation. This shared memory is limited by the hardware to 48kB.
	'''

	def __init__(self):
		super(EMDLoss, self).__init__()

	def forward(self, xyz1, xyz2):
		'''
		xyz1: B x N x D point set
		xyz2: B x M x D point set
		'''

		assert xyz1.shape[-1] == xyz2.shape[-1], 'Both point sets must have the same dimensionality'
		if xyz1.dtype == torch.float64 and xyz1.shape[-1] > 11:
			error('Tensors of type double can have a maximum of 11 dimensions')
		if xyz1.dtype == torch.float32 and xyz1.shape[-1] > 23:
			error('Tensors of type float can have a maximum of 23 dimensions')

		return EMDFunction.apply(xyz1, xyz2)