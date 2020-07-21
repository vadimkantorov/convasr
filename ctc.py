import math
import torch
import torch.nn.functional as F

def alignment(log_probs, targets, input_lengths, target_lengths, blank = 0, pack_backpointers = False):
	B = torch.arange(len(targets), device = input_lengths.device)
	_t_a_r_g_e_t_s_ = torch.cat([torch.stack([torch.full_like(targets, blank), targets], dim = -1).flatten(start_dim = -2), torch.full_like(targets[:, :1], blank)], dim = -1)
	diff_labels = torch.cat([torch.as_tensor([[False, False]], device = targets.device).expand(len(B), -1), _t_a_r_g_e_t_s_[:, 2:] != _t_a_r_g_e_t_s_[:, :-2]], dim = 1)

	zero, zero_padding = torch.tensor(torch.finfo(log_probs.dtype).min, device = log_probs.device, dtype = log_probs.dtype), 2
	padded_t = zero_padding + _t_a_r_g_e_t_s_.shape[-1]
	log_alpha = torch.full((len(B), padded_t), zero, device = log_probs.device, dtype = log_probs.dtype)
	log_alpha[:, zero_padding + 0] = log_probs[0, :, blank]
	log_alpha[:, zero_padding + 1] = log_probs[0, B, _t_a_r_g_e_t_s_[:, 1]]
	
	backpointers_packmask = 0b11
	backpointers_shape = (len(log_probs), len(B), padded_t)
	backpointers = torch.zeros(backpointers_shape if not pack_backpointers else packshape(backpointers_shape, mask = backpointers_packmask)[0], device = log_probs.device, dtype = torch.uint8)
	backpointer = torch.zeros(backpointers_shape[1:], device = log_probs.device, dtype = torch.uint8)

	for t in range(1, len(log_probs)):
		prev = torch.stack([log_alpha[:, 2:], log_alpha[:, 1:-1], torch.where(diff_labels, log_alpha[:, :-2], zero)])
		log_alpha[:, 2:] = log_probs[t].gather(-1, _t_a_r_g_e_t_s_) + prev.logsumexp(dim = 0)
		backpointer[:, 2:] = prev.argmax(dim = 0)
		backpointers[t] = backpointer if not pack_backpointers else packbits(backpointer, mask = backpointers_packmask)

	#l1l2 = log_alpha[input_lengths - 1, B].gather(-1, torch.stack([zero_padding + target_lengths * 2 - 1, zero_padding + target_lengths * 2], dim = -1)) 
	l1l2 = log_alpha.gather(-1, torch.stack([zero_padding + target_lengths * 2 - 1, zero_padding + target_lengths * 2], dim = -1)) 
	
	path = torch.zeros(len(log_probs), len(B), device = log_alpha.device, dtype = torch.long)
	path[input_lengths - 1, B] = zero_padding + target_lengths * 2 - 1 + l1l2.argmax(dim = -1)
	for t, indices in reversed(list(enumerate(path))[1:]):
		backpointer = backpointers[t] if not pack_backpointers else unpackbits(backpointers[t], mask = backpointers_packmask, shape = backpointers_shape[1:])
		path[t - 1] += indices - backpointer.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
	return torch.zeros_like(_t_a_r_g_e_t_s_, dtype = torch.long).scatter_(-1, (path.t() - zero_padding).clamp(min = 0), torch.arange(len(path), device = log_alpha.device).expand(len(B), -1))[:, 1::2]

def tensor_dim_slice(tensor, dim, dim_slice):
	return tensor[(dim if dim >= 0 else dim + tensor.dim()) * (slice(None), ) + (dim_slice, )]

def packshape(shape, dim = -1, mask = 0b00000001, dtype = torch.uint8):
	nbits_element = torch.iinfo(dtype).bits
	nbits = 1 if mask == 0b00000001 else 2 if mask == 0b00000011 else 4 if mask == 0b00001111 else 8 if mask == 0b11111111  else None
	assert nbits is not None and nbits <= nbits_element and nbits_element % nbits == 0
	packed_size = nbits_element // nbits
	shape = list(shape)
	shape[dim] = int(math.ceil(shape[dim] / packed_size))
	return tuple(shape), packed_size, nbits

def packbits(tensor, dim = -1, mask = 0b00000001, out = None, dtype = torch.uint8):
	shape, packed_size, nbits = packshape(tensor.shape, dim = dim, mask = mask, dtype = dtype)
	out = out.zero_() if out is not None else torch.zeros(shape, device = tensor.device, dtype = dtype)
	assert out.shape == tuple(shape)
	for i in range(packed_size):
		sliced_input = tensor_dim_slice(tensor, dim, slice(i, None, packed_size))
		sliced_output = out.narrow(dim, 0, sliced_input.shape[dim])
		compress = sliced_input << (nbits * (packed_size - i - 1))
		torch.bitwise_or(compress, sliced_output, out = sliced_output)
	return out

def unpackbits(tensor, shape, dim = -1, mask = 0b00000001, out = None, dtype = torch.uint8):
	_, packed_size, nbits = packshape(shape, dim = dim, mask = mask, dtype = tensor.dtype)
	out = out.zero_() if out is not None else torch.zeros(shape, device = tensor.device, dtype = dtype)
	assert out.shape == tuple(shape)
	for i in range(packed_size):
		sliced_output = tensor_dim_slice(out, dim, slice(i, None, packed_size))
		sliced_input = tensor.narrow(dim, 0, sliced_output.shape[dim])
		expand = sliced_input >> (nbits * (packed_size - i - 1))
		torch.bitwise_and(expand, mask, out = sliced_output)
	return out
