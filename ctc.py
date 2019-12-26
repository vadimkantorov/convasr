import torch
import torch.nn.functional as F

def ctc_loss___(log_probs, targets, input_lengths, target_lengths, blank = 0, reduction = 'none'):
	targets_ = torch.full((targets.shape[0], 2 * targets.shape[-1] + 1), blank, device = targets.device, dtype = targets.dtype)
	temporal_mask = torch.arange(targets.shape[-1], device = input_lengths.device, dtype = input_lengths.dtype).unsqueeze(0) < target_lengths.unsqueeze(1)
	targets_[:, 1::2] = temporal_mask * targets + (~temporal_mask) * targets_[:, 1::2]

	max_target_length = int(target_lengths.max())
	max_target_length_ = 2 * max_target_length + 1
	batch_size = targets.shape[0]

	log_alpha = torch.empty(batch_size, log_probs.shape[0], 2 + max_target_length_, device = log_probs.device, dtype = log_probs.dtype)
	neg_log_likelihood = torch.empty(batch_size, device = log_probs.device, dtype = log_probs.dtype)
	
	neginf = torch.as_tensor([float('-inf')], device = log_probs.device, dtype = log_probs.dtype)
	log_alpha[:, :3].fill_(neginf.sum())
	two_true = torch.as_tensor([True, True], device = targets.device)
	
	la3_ = torch.cat([two_true.unsqueeze(0).expand(batch_size, -1), targets_[:, 2:max_target_length_] != targets_[:, :max_target_length_-2]], dim = 1)

	log_alpha[:, 0, 2 + 0] = log_probs[0, :, blank]
	log_alpha[:, 0, 2 + 1] = log_probs[0, torch.arange(batch_size), targets_[:, 1]]

	for t in range(1, len(log_probs)):
		la3 = log_alpha[:, t - 1, 0:-2]
		la2 = log_alpha[:, t - 1, 1:-1]
		la1 = log_alpha[:, t - 1, 2:]
		log_alpha[:, t, 2:] = torch.logsumexp(torch.stack([la1, la2, torch.where(la3_, la3, neginf)]), dim = 0) + log_probs[t].gather(1, targets_[:, : 2 * max_target_length + 1])
		#for b in range(batch_size):
		#	la3 = log_alpha[b, t - 1, 0:-2]
		#	la2 = log_alpha[b, t - 1, 1:-1]
		#	la1 = log_alpha[b, t - 1, 2:]
		#	log_alpha[b, t, 2:] = torch.logsumexp(torch.stack([la1, la2, torch.where(la3_[b], la3, neginf)]), dim = 0) + log_probs[t, b, targets_[b,  : 2 * max_target_length + 1]]

	l1 = log_alpha[:, input_lengths - 1, 2 + target_lengths * 2].diag()
	l2 = log_alpha[:, input_lengths - 1, 2 + target_lengths * 2 - 1].diag()
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

#log_probs = F.log_softmax(torch.rand(1, 2, 2), dim = 1).permute(2, 0, 1).detach().requires_grad_()
#targets = torch.LongTensor([[1, 1]])
#input_lengths = torch.LongTensor([2])
#target_lengths = torch.LongTensor([2])

torch.manual_seed(1)
log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().requires_grad_()
targets = torch.randint(1, 20, (16, 50), dtype=torch.long)
input_lengths = torch.full((16,), 50, dtype=torch.long)
target_lengths = torch.randint(49,50,(16,), dtype=torch.long)
loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction = 'none')
loss_ = ctc_loss___(log_probs, targets, input_lengths, target_lengths, reduction = 'none')

print(loss)
print(loss_)
