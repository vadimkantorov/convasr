import random

class SpecAugment(object):
    def __init__(self, n_freq_mask = 2, n_time_mask = 2, width_freq_mask = 6, width_time_mask = 6, replace_strategy = None):
        # fb code: https://github.com/facebookresearch/wav2letter/commit/04c3d80bf66fe749466cd427afbcc936fbdec5cd
        # width_freq_mask = 27, width_time_mask = 100, and n_freq_mask/n_time_mask = 2
        # google code: https://github.com/tensorflow/lingvo/blob/master/lingvo/core/spectrum_augmenter.py#L37-L42
        # width_freq_mask = 10 and width_time_mask = 50, and n_freq_mask/n_time_mask = 2

        self.replace_strategy = replace_strategy
        self.n_time_mask = n_time_mask
        self.n_freq_mask = n_freq_mask
        self.width_time_mask = width_time_mask
        self.width_freq_mask = width_freq_mask

    def __call__(self, spect):
        replace_val = spect.mean() if self.replace_strategy == 'mean' else 0

        for idx in range(self.n_freq_mask):
            f = random.randint(0, self.width_freq_mask)
            f0 = random.randint(0, spect.shape[0] - f)
            spect[f0:f0 + f, :] = replace_val

        for idx in range(self.n_time_mask):
            t = random.randint(0, min(self.width_time_mask, spect.shape[1]))
            t0 = random.randint(0, spect.shape[1] - t)
            spect[:, t0:t0 + t] = replace_val

        return spect
