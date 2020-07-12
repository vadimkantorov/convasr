import torch

def exp_moving_average(avg, value, K=50):
	return (1. / K) * value + (1 - 1. / K) * avg

class PerformanceMeter:
	def __init__(self):
		self.metrics = dict()
		self.byte_scaler = 1024 ** 3

	def update_metric(self, name, value, subtag = None):
		if subtag is None:
			avg_name = f'performance/{name}_avg'
			max_name = f'performance/{name}_max'
		else:
			avg_name = f'performance/{name}_avg/{subtag}'
			max_name = f'performance/{name}_max/{subtag}'
		old_value = self.metrics.get(avg_name, 0)
		self.metrics[avg_name] = exp_moving_average(old_value, value)

		old_value = self.metrics.get(max_name, 0)
		self.metrics[max_name] = max(old_value, value)

	def update_memory_metrics(self):
		device_count = torch.cuda.device_count()
		total_allocated = 0
		total_reserved = 0
		for i in range(device_count):
			device_stats = torch.cuda.memory_stats(i)

			allocated = device_stats[f'allocated_bytes.all.peak'] / self.byte_scaler
			total_allocated += allocated
			self.update_metric('allocated', allocated, f'cuda:{i}')

			reserved = device_stats[f'reserved_bytes.all.peak'] / self.byte_scaler
			total_reserved += reserved
			self.update_metric('reserved', reserved, f'cuda:{i}')

		self.update_metric('allocated', total_allocated, 'total')
		self.update_metric('reserved', total_reserved, 'total')

	def update_time_metrics(self, time_ms_data, time_ms_fwd, time_ms_bwd, time_ms_model):
		self.update_metric('time_data', time_ms_data)
		self.update_metric('time_forward', time_ms_fwd)
		self.update_metric('time_backward', time_ms_bwd)
		self.update_metric('time_iteration', time_ms_data + time_ms_model)