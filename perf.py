def exp_moving_average(avg, val, max = 0, K = 50):
	return (1. / K) * min(val, max) + (1 - 1. / K) * avg

class PerformanceMeterDict(dict):
	def __init__(self, *, K = 50, max = 1000, **config):
		self.config = config
		self.K = K
		self.max = max

	def update(self, kwargs, prefix = ''):
		if prefix:
			prefix += '_'

		for name, value in kwargs.items():
			avg_name, max_name, cur_name = prefix + 'avg_' + name, prefix + 'max_' + name, prefix + 'cur_' + name 
			self[avg_name] = exp_moving_average(self.get(avg_name, 0), value, K = self.config.get(name, {}).get('K', self.K), max = self.config.get(name, {}).get('max', self.max))
			self[max_name] = max(self.get(max_name, 0), value)
			self[cur_name] = value
	
	def __missing__(self, key):
		return 0.0

default = PerformanceMeterDict()
update = default.update
