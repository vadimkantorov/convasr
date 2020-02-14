import torch
import torch.nn.functional as F

def alignment(log_probs, targets, input_lengths, target_lengths, blank = 0):
	B = torch.arange(len(targets), device = input_lengths.device)
	targets_ = torch.cat([torch.stack([torch.full_like(targets, blank), targets], dim = -1).flatten(start_dim = -2), torch.full_like(targets[:, :1], blank)], dim = -1)
	diff_labels = torch.cat([torch.as_tensor([[False, False]], device = targets.device).expand(len(B), -1), targets_[:, 2:] != targets_[:, :-2]], dim = 1)

	zero, zero_padding = torch.tensor(torch.finfo(log_probs.dtype).min, device = log_probs.device, dtype = log_probs.dtype), 2
	padded_t = zero_padding + targets_.shape[-1]
	log_alpha = torch.full((len(B), padded_t), zero, device = log_probs.device, dtype = log_probs.dtype)
	log_alpha[:, zero_padding + 0] = log_probs[0, :, blank]
	log_alpha[:, zero_padding + 1] = log_probs[0, B, targets_[:, 1]]
	log_alpha_ = torch.zeros((len(log_probs), len(B), padded_t), device = log_probs.device, dtype = torch.uint8)
	for t in range(1, len(log_probs)):
		prev = torch.stack([log_alpha[:, 2:], log_alpha[:, 1:-1], torch.where(diff_labels, log_alpha[:, :-2], zero)])
		log_alpha[:, 2:] = log_probs[t].gather(-1, targets_) + prev.logsumexp(dim = 0)
		log_alpha_[t, :, 2:] = prev.argmax(dim = 0)

	#l1l2 = log_alpha[input_lengths - 1, B].gather(-1, torch.stack([zero_padding + target_lengths * 2 - 1, zero_padding + target_lengths * 2], dim = -1)) 
	l1l2 = log_alpha.gather(-1, torch.stack([zero_padding + target_lengths * 2 - 1, zero_padding + target_lengths * 2], dim = -1)) 
	loss = -torch.logsumexp(l1l2, dim = -1)
	
	path = torch.zeros(len(log_probs), len(B), device = log_alpha.device, dtype = torch.long)
	path[input_lengths - 1, B] = zero_padding + target_lengths * 2 - 1 + l1l2.argmax(dim = -1)
	for t, indices in reversed(list(enumerate(path))[1:]):
		path[t - 1] += indices - log_alpha_[t].gather(-1, indices.unsqueeze(-1)).squeeze(-1)
	return torch.zeros_like(targets_, dtype = torch.long).scatter_(-1, (path.t() - zero_padding).clamp(min = 0), torch.arange(len(path), device = log_alpha.device).expand(len(B), -1))[:, 1::2]
