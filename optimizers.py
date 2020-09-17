import torch


def reset_options(optimizer):
	for param_group in optimizer.param_groups:
		param_group.update(optimizer.defaults)


class LRScheduler:
	def __init__(self, optimizer):
		self.optimizer = optimizer

	def step(self, step):
		for group, lr in zip(self.optimizer.param_groups, self.get_lr(step)):
			group['lr'] = lr


class NoopLR(LRScheduler):
	def get_lr(self, step):
		return [group['lr'] for group in self.optimizer.param_groups]


class MultiStepLR(LRScheduler):
	def __init__(self, optimizer, gamma, milestones):
		self.init_lr = [group['lr'] for group in optimizer.param_groups]
		self.gamma = gamma
		self.milestones = milestones
		super().__init__(optimizer)

	def get_lr(self, step):
		gamma_power = ([0] + [i + 1 for i, m in enumerate(self.milestones) if step >= m])[-1]
		return [init_lr * (self.gamma**gamma_power) for init_lr in self.init_lr]


class PolynomialDecayLR(LRScheduler):
	def __init__(self, optimizer, decay_steps, power = 1.0, begin_decay_at = 0, end_lr = 0.0, warmup_steps = 0):
		self.decay_steps = decay_steps
		self.power = power
		self.begin_decay_at = begin_decay_at
		self.end_lr = end_lr
		self.warmup_steps = warmup_steps
		self.init_lr = [group['lr'] for group in optimizer.param_groups]
		super().__init__(optimizer)

	def get_lr(self, step):
		lrs = list(
			map(
				lambda init_lr: (init_lr * step / self.warmup_steps)
				if self.warmup_steps > 0 and step < self.warmup_steps else init_lr,
				self.init_lr
			)
		)
		if step >= self.begin_decay_at:
			step = min(step - self.begin_decay_at, self.decay_steps)
			lrs = list(
				map(
					lambda init_lr: self.end_lr + (init_lr - self.end_lr) *
					((self.decay_steps - step) / self.decay_steps)**self.power
					if step < self.decay_steps else self.end_lr,
					lr
				)
			)
		return lrs


class NovoGrad(torch.optim.Optimizer):
	def __init__(self, params, lr = 1.0, betas = (0.95, 0.98), eps = 1e-8, weight_decay = 0.0, dampening = False):
		super().__init__(
			params, dict(lr = lr, betas = betas, eps = eps, weight_decay = weight_decay, dampening = dampening)
		)

	@torch.no_grad()
	def step(self):
		for group in self.param_groups:
			for p in filter(lambda p: p.grad is not None, group['params']):
				state = self.state[p]
				g_2 = (p.grad**2).sum()
				state['_grads_ema'] = g_2 if '_grads_ema' not in state else state['_grads_ema'] * group['betas'][
					1] + g_2 * (1. - group['betas'][1])

				grad = p.grad / (state['_grads_ema'] + group['eps']).sqrt()
				if group['weight_decay'] > 0:
					grad.add_(p, alpha = group['weight_decay'])
				if group['dampening']:
					grad *= 1 - group['betas'][0]

				state['momentum_buffer'] = state['momentum_buffer'].mul_(
					group['betas'][0]
				).add_(grad) if 'momentum_buffer' in state else grad
				p.add_(state['momentum_buffer'], alpha = -group['lr'])


@torch.no_grad()
def larc_(param_groups, larc_mode = 'clip', eps = 1e-7, min_update = 1e-7, larc_eta = 0.1):  #1e-3):
	for group in param_groups:
		for p in filter(lambda p: p.grad is not None, group['params']):
			v_norm = p.norm()
			g_norm = p.grad.norm()

			if larc_mode == 'clip':
				larc_grad_update = torch.clamp(
					larc_eta * v_norm / (group['lr'] * (g_norm + eps)), min = min_update, max = 1
				)
			else:
				larc_grad_update = torch.clamp(larc_eta * v_norm / (g_norm + eps), min = min_update)
			p.grad.mul_(larc_grad_update)
