def exp_moving_average(avg, val, max = 0, K = 50):
	if avg is None:
		return val
	return (1. / K) * min(val, max) + (1 - 1. / K) * avg


class PerformanceMeterDict(dict):
	__instance = None

	def __init__(self, *, K = 50, max = 1000, **config):
		self.config = config
		self.K = K
		self.max = max

	@classmethod
	def init_default(cls, *, K = 50, max = 1000, **config):
		cls.__instance = cls(K = K, max = max, **config)

	@classmethod
	def default(cls):
		assert cls.__instance is not None, 'Default performance meter is not initialized, call "perf.init_default" to fix this.'
		return cls.__instance

	@classmethod
	def update_default(cls, kwargs, prefix = ''):
		cls.default().update(kwargs, prefix)

	def update(self, kwargs, prefix = ''):
		if prefix:
			prefix += '_'

		for name, value in kwargs.items():
			avg_name, max_name, cur_name = prefix + 'avg_' + name, prefix + 'max_' + name, prefix + 'cur_' + name
			self[avg_name] = exp_moving_average(self.get(avg_name, None), value, K=self.config.get(name, {}).get('K', self.K), max=self.config.get(name, {}).get('max', self.max))
			self[max_name] = max(self.get(max_name, 0), value)
			self[cur_name] = value

	def __missing__(self, key):
		return 0.0


init_default = PerformanceMeterDict.init_default
default = PerformanceMeterDict.default
update = PerformanceMeterDict.update_default
