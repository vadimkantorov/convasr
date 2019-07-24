import random

class SpecAugment(object):
    def __init__(self, n_freq_mask = 2, n_time_mask = 2, width_freq_mask = 6, width_time_mask = 6):
        self.n_time_mask = n_time_mask
        self.n_freq_mask = n_freq_mask
        self.width_time_mask = width_time_mask
        self.width_freq_mask = width_freq_mask

    def __call__(self, spect):
        for idx in range(self.n_freq_mask):
            freq_band = random.randint(0, self.width_freq_mask + 1)
            freq_base = random.randint(0, spect.shape[1] - freq_band)
            spect[freq_base:freq_base+freq_band, :] = 0

        for idx in range(self.n_time_mask):
            time_band = random.randint(0, self.width_time_mask + 1)
            if spect.shape[0] - time_band > 0:
                time_base = random.randint(0, spect.shape[0] - time_band)
                spect[:, time_base:time_base+time_band] = 0

        return spect
