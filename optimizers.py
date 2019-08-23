import torch

class NoopLR(torch.optim.lr_scheduler._LRScheduler):
	def get_lr(self):
		return [group['lr'] for group in self.optimizer.param_groups]

class PolynomialDecayLR(torch.optim.lr_scheduler._LRScheduler):
	def __init__(self, optimizer, decay_steps, power = 1.0, begin_decay_at = 0, end_lr = 0.0, warmup_steps = 0, last_epoch = -1):
		self.decay_steps = decay_steps
		self.power = power
		self.begin_decay_at = begin_decay_at
		self.end_lr = end_lr
		self.warmup_steps = warmup_steps
		self.init_lr = [group['lr'] for group in optimizer.param_groups]
		super(PolynomialDecayLR, self).__init__(optimizer, last_epoch)

	def get_lr(self):
		global_step = self.last_epoch
		lr = list(map(lambda init_lr: (init_lr * global_step / self.warmup_steps) if self.warmup_steps > 0 and global_step < self.warmup_steps else init_lr, self.init_lr))
		if global_step >= self.begin_decay_at:
			global_step = min(global_step - self.begin_decay_at, self.decay_steps)
			lr = list(map(lambda init_lr: self.end_lr + (init_lr - self.end_lr) * ((self.decay_steps - global_step) / self.decay_steps) ** self.power if global_step < self.decay_steps else self.end_lr, lr))
		return lr

class NovoGrad(torch.optim.Optimizer):
	def __init__(self, params, lr=1.0, betas = (0.95, 0.98), eps=1e-8, weight_decay=0.0, dampening=False):
		defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, dampening=dampening)
		super(NovoGrad, self).__init__(params, defaults)

	def step(self):
		for group in self.param_groups:
			for p in filter(lambda p: p.grad is not None, group['params']):
				state = self.state[p]
				g_2 =  torch.dot(p.grad.data, p.grad.data)
				state['_grads_ema'] = g_2 if '_grads_ema' not in state else state['_grads_ema'] * group['betas'][1] + g_2 * (1. - group['betas'][1])

				grad = (state['_grads_ema'] + group['eps']).rsqrt_().mul_(p.grad.data)
				if group['weight_decay'] > 0:
					grad.add_(group['weight_decay'], p.data)
				if group['dampening']:
					grad *= 1 - group['betas'][0]

				state['momentum_buffer'] = state['momentum_buffer'].mul_(group['betas'][0]).add_(grad) if 'momentum_buffer' in state else grad
				p.data.add_(-group['lr'], state['momentum_buffer'])

def larc(param_groups, larc_eta = 1e-3, larc_mode = 'clip', min_update = 1e-7, eps = 1e-7):
	for group in param_groups:
		for p in filter(lambda p: p.grad is not None, group['params']):
			v_norm = p.data.norm()
			g_norm = p.grad.data.norm()

			if larc_mode == 'clip':
				larc_grad_update = torch.clamp(larc_eta * v_norm / (group['lr'] * (g_norm + eps)), min = min_update, max = 1)
			else:
				larc_grad_update = torch.clamp(larc_eta * v_norm / (g_norm + eps), min = min_update)
			p.grad.data.mul_(larc_grad_update)
