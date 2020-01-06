import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

# https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/LossCTC.cpp#L37
# https://github.com/skaae/Lasagne-CTC/blob/master/ctc_cost.py#L162
@torch.jit.script
def ctc_alignment(log_probs, targets, input_lengths, target_lengths, blank : int = 0, reduction : str = 'none', zero : float = float('-inf')):#, add = torch.logsumexp):# add = (lambda t, dim = None: torch.max(t, dim = dim).values), ):#,
	targets_ = torch.cat([targets, targets[:, :1]], dim = -1)
	targets_ = torch.stack([torch.full_like(targets_, blank), targets_], dim = -1).flatten(start_dim = -2)
	B = torch.arange(len(targets), device = input_lengths.device)
	zero_padding = 2
	diff_labels = torch.cat([torch.as_tensor([[False, False]], device = targets.device).expand(len(B), -1), targets_[:, 2:] != targets_[:, :-2]], dim = 1)
	zero = torch.tensor(zero, device = log_probs.device, dtype = log_probs.dtype)
	
	log_alpha = torch.full((len(B), len(log_probs), zero_padding + targets_.shape[-1]), zero, device = log_probs.device, dtype = log_probs.dtype)
	log_alpha[:, 0, zero_padding + 0] = log_probs[0, :, blank]
	log_alpha[:, 0, zero_padding + 1] = log_probs[0, B, targets_[:, 1]]
	log_alpha[:, 1:, zero_padding:] = log_probs.gather(-1, targets_.expand(len(log_probs), -1, -1))[1:].permute(1, 0, 2)
	for t in range(1, len(log_probs)):
		log_alpha[:, t, 2:] += torch.logsumexp(torch.stack([log_alpha[:, t - 1, 2:], log_alpha[:, t - 1, 1:-1], torch.where(diff_labels, log_alpha[:, t - 1, :-2], zero)]), dim = 0)

	l1l2 = log_alpha[B, input_lengths - 1].gather(-1, torch.stack([zero_padding + target_lengths * 2 - 1, zero_padding + target_lengths * 2], dim = -1)) 
	loss = -torch.logsumexp(l1l2, dim = -1)
	
	path = torch.zeros(len(B), len(log_probs), device = log_probs.device, dtype = torch.int64)
	path[B, input_lengths - 1] = zero_padding + 2 * target_lengths - 1 + l1l2.max(dim = -1).indices
	for t in range(len(log_probs) - 1, 1, -1):
		indices = path[:, t]
		indices_ = torch.stack([(indices - 2) * diff_labels[B, (indices - zero_padding).clamp(min = 0)], (indices - 1).clamp(min = 0), indices], dim = -1)
		path[:, t - 1] += (indices - 2 + log_alpha[B, t - 1].gather(-1, indices_).max(dim = -1).indices).clamp(min = 0)
	return torch.zeros_like(log_alpha).scatter_(-1, path.unsqueeze(-1), 1.0)[..., 3::2]

def align_slow(input_lengths, target_lengths, log_alpha, l1l2):
	log_alpha_ = log_alpha[:, :, 2:]
	paths = torch.zeros_like(log_alpha_, dtype = torch.bool)
	for b in range(len(targets)):
		table = log_alpha_[b]
		i = int(input_lengths[b] - 1)
		j = max((table[i, int(target_lengths[b] * 2 + 1 - 1)], int(target_lengths[b] * 2 + 1 - 1)), (table[i, int(target_lengths[b] * 2 - 1)], int(target_lengths[b] * 2 - 1)))[1]
		paths[b, i, j] = True
		while i > 0 and j > 0:
			val = zero
			step = i, j
			if table[i - 1, j] > val:
				step, val = (i - 1, j), table[i - 1, j]
			if table[i - 1, j - 1] > val:
				step, val = (i - 1, j - 1), table[i - 1, j - 1]
			if j > 1 and la3_[b, j] and table[i - 1, j - 2] > val:
				step, val = (i - 1, j - 2), table[i - 1, j - 2]
			
			if step == (i, j):
				break
			i, j = step
			paths[b, i, j] = True
	return paths[:, :, 1::2]

@torch.jit.script
def ctc_loss___(log_probs, targets, input_lengths, target_lengths, blank : int = 0, reduction : str = 'none'):
	targets_ = torch.cat([targets, targets[:, :1]], dim = -1)
	targets_ = torch.stack([torch.full_like(targets_, blank), targets_], dim = -1).flatten(start_dim = -2)
	B = torch.arange(len(targets), device = input_lengths.device)
	
	log_alpha = torch.full((len(targets), len(log_probs), 2 + targets_.shape[-1]), float('-inf'), device = log_probs.device, dtype = log_probs.dtype)
	log_alpha[:, 1:, 2:] = log_probs.gather(-1, targets_.expand(len(log_probs), -1, -1))[1:].permute(1, 0, 2)
	log_alpha[:, 0, 2 + 0] = log_probs[0, :, blank]
	log_alpha[:, 0, 2 + 1] = log_probs[0, B, targets_[:, 1]]
	
	la3_ = torch.cat([torch.as_tensor([[True, True]], device = targets.device).expand(len(targets), -1), targets_[:, 2:] != targets_[:, :-2]], dim = 1)
	neginf = torch.tensor(float('-inf'), device = log_probs.device, dtype = log_probs.dtype)
	for t in range(1, len(log_probs)):
		la3 = log_alpha[:, t - 1, 0:-2]
		la2 = log_alpha[:, t - 1, 1:-1]
		la1 = log_alpha[:, t - 1, 2:]
		log_alpha[:, t, 2:] += torch.logsumexp(torch.stack([la1, la2, torch.where(la3_, la3, neginf)]), dim = 0)

	l1 = log_alpha[B, input_lengths - 1, 2 + target_lengths * 2]
	l2 = log_alpha[B, input_lengths - 1, 2 + target_lengths * 2 - 1]
	return -torch.logsumexp(torch.stack([l1, l2]), dim = 0)


def ctc_loss__(log_probs, targets, input_lengths, target_lengths, blank = 0, reduction = 'none'):
	targets_ = torch.full((targets.shape[0], 2 * targets.shape[-1] + 1), blank, device = targets.device, dtype = targets.dtype)
	temporal_mask = torch.arange(targets.shape[-1], device = input_lengths.device, dtype = input_lengths.dtype).unsqueeze(0) < target_lengths.unsqueeze(1)
	targets_[:, 1::2] = temporal_mask * targets + (~temporal_mask) * targets_[:, 1::2]

	max_target_length = int(target_lengths.max())
	batch_size = targets.shape[0]

	log_alpha = torch.empty(batch_size, log_probs.shape[0], 2 + 2 * max_target_length + 1, device = log_probs.device, dtype = log_probs.dtype)
	neg_log_likelihood = torch.empty(batch_size, device = log_probs.device, dtype = log_probs.dtype)
	lpp  = log_probs.permute(1, 0, 2)
	
	neginf = torch.as_tensor([float('-inf')], device = log_probs.device, dtype = log_probs.dtype)
	log_alpha[:, :3].fill_(neginf.sum())
	two_true = torch.as_tensor([True, True], device = targets.device)
	for b in range(batch_size):
		input_length = input_lengths[b]
		target_length = target_lengths[b]
		log_alpha_a = log_alpha[b]
		log_probs_a = lpp[b]
		get_target_prime = targets_[b,  : 2 * max_target_length + 1]

		log_alpha_a[0, 2 + 0] = log_probs_a[0, blank]
		log_alpha_a[0, 2 + 1] = log_probs_a[0, get_target_prime[1]]
		la3_ = torch.cat([two_true, get_target_prime[2:] != get_target_prime[:-2]])
		
		for t in range(1, input_length):
			la3 = log_alpha_a[t - 1, 0:-2]
			la2 = log_alpha_a[t - 1, 1:-1]
			la1 = log_alpha_a[t - 1, 2:]
			log_alpha_a[t, 2:] = torch.logsumexp(torch.stack([la1, la2, torch.where(la3_, la3, neginf)]), dim = 0) + log_probs_a[t, get_target_prime]

		l1 = log_alpha_a[input_length - 1, 2 + target_length * 2]
		l2 = log_alpha_a[input_length - 1, 2 + target_length * 2 - 1]
		m = torch.max(l1, l2)
		m = 0 if m == neginf else m
		log_likelihood = torch.log(torch.exp(l1 - m) + torch.exp(l2 - m)) + m
		neg_log_likelihood[b] = -log_likelihood

	return neg_log_likelihood

def ctc_loss_(log_probs, targets, input_lengths, target_lengths, blank = 0, reduction = 'none'):
	targets_ = torch.full((targets.shape[0], 2 * targets.shape[-1] + 1), blank, device = targets.device, dtype = targets.dtype)
	temporal_mask = torch.arange(targets.shape[-1], device = input_lengths.device, dtype = input_lengths.dtype).unsqueeze(0) < target_lengths.unsqueeze(1)
	targets_[:, 1::2] = temporal_mask * targets + (~temporal_mask) * targets_[:, 1::2]

	max_target_length = int(target_lengths.max())
	batch_size = targets.shape[0]

	log_alpha = torch.empty(batch_size, log_probs.shape[0], 2 * max_target_length + 1, device = log_probs.device, dtype = log_probs.dtype)
	neg_log_likelihood = torch.empty(batch_size, device = log_probs.device, dtype = log_probs.dtype)
	lpp  = log_probs.permute(1, 0, 2)
	
	neginf = torch.as_tensor([float('-inf')], device = log_probs.device, dtype = log_probs.dtype)
	log_alpha.narrow(1, 0, 1).fill_(neginf.sum())

	for b in range(batch_size):
		input_length = input_lengths[b]
		target_length = target_lengths[b]
		log_alpha_a = log_alpha[b]
		log_probs_a = lpp[b]
		get_target_prime = targets_[b]

		log_alpha_a[0, 0] = log_probs_a[0, blank]
		log_alpha_a[0, 1] = log_probs_a[0, get_target_prime[1]]

		for t in range(1, input_length):
			for s in range(0, 2 * target_length + 1):
				current_target_prime = get_target_prime[s]
				la1 = log_alpha_a[t - 1, s]
				lamax = la1
				if s > 0:
					la2 = log_alpha_a[t - 1, s-1]
					if la2 > lamax:
						lamax = la2
				else: 
					la2 = neginf

				if s > 1 and get_target_prime[s - 2] != current_target_prime:
					la3 = log_alpha_a[t - 1, s-2]
					if la3 > lamax:
						lamax = la3
				else:
					la3 = neginf
			
				if lamax == neginf:
					lamax = 0

				log_alpha_a[t, s] = torch.log(torch.exp(la1 - lamax) + torch.exp(la2 - lamax) + torch.exp(la3 - lamax)) + lamax + log_probs_a[t, current_target_prime]

		l1 = log_alpha_a[input_length - 1, target_length * 2]
		l2 = log_alpha_a[input_length - 1, target_length * 2 - 1]
		m = torch.max(l1, l2)
		m = 0 if m == neginf else m
		log_likelihood = torch.log(torch.exp(l1 - m) + torch.exp(l2 - m)) + m
		neg_log_likelihood[b] = -log_likelihood

	return neg_log_likelihood

if __name__ == '__main__':
	#log_probs = F.log_softmax(torch.rand(1, 2, 2), dim = 1).permute(2, 0, 1).detach().requires_grad_()
	#targets = torch.LongTensor([[1, 1]])
	#input_lengths = torch.LongTensor([2])
	#target_lengths = torch.LongTensor([2])

	torch.manual_seed(1)
	log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().requires_grad_()
	targets = torch.randint(1, 20, (16, 50), dtype=torch.long)
	input_lengths = torch.full((16,), 50, dtype=torch.long)
	target_lengths = torch.randint(49,50,(16,), dtype=torch.long)
	loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank = 0, reduction = 'none')
	loss_ = ctc_loss___(log_probs, targets, input_lengths, target_lengths, blank = 0, reduction = 'none')
	#print(ctc_loss___.graph_for(log_probs, targets, input_lengths, target_lengths, blank = 0, reduction = 'none'))
	print(loss)
	print(loss_)

	P =  ctc_alignment(log_probs, targets, input_lengths, target_lengths, blank = 0, reduction = 'none')
	plt.imshow(P[1].t(), origin = 'lower', aspect = 'auto')
	plt.savefig('data/alignment.jpg')
