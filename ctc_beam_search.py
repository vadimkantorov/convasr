import numpy as np
import math
import collections
import time

NEG_INF = -float("inf")

def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap

def make_new_beam():
	fn = lambda: (NEG_INF, NEG_INF)
	return collections.defaultdict(fn)


def logsumexp(*args):
	"""
	Stable log sum exp.
	"""
	if all(a == NEG_INF for a in args):
		return NEG_INF
	a_max = max(args)
	lsp = math.log(sum(math.exp(a - a_max)
		for a in args))
	return a_max + lsp


def join_prefix(prefix):
	return ' '.join(prefix)


def decode_labels(prefix, labels):
	if len(prefix) < 0:
		return prefix

	decoded = labels.decode(
			prefix,
			None,
			channel=0,
			replace_blank=True,
			replace_blank_series=True,
			replace_repeat=True,
			replace_space=False,
			speaker=None
	)

	return decoded

# todo
# разобраться с вероятностями.
#


@timing
def decode(probs, beam_size = 10, blank = 0, is_logprobs = True, lm=None, labels=None, min_cutoff=0):
	"""
	Performs inference for the given output probabilities.
	Arguments:
		probs: The output probabilities (e.g. post-softmax) for each
		  time step. Should be an array of shape (time x output dim).
		beam_size (int): Size of the beam to use during inference.
		blank (int): Index of the CTC blank label.
	Returns the output label sequence and the corresponding negative
	log-likelihood estimated by the decoder.
	"""
	S, T = probs.shape

	if not is_logprobs:
		probs = np.log(probs)

	# Elements in the beam are (prefix, (p_blank, p_no_blank))
	# Initialize the beam with the empty sequence, a probability of
	# 1 for ending in blank and zero for ending in non-blank
	# (in log space).
	beam = [(tuple(), (0.0, NEG_INF))]

	for t in range(T):  # Loop over time

		# A default dictionary to store the next step candidates.
		next_beam = make_new_beam()

		for s in range(S):  # Loop over vocab
			p = probs[s, t]

			if p < min_cutoff:
				continue

			# The variables p_b and p_nb are respectively the
			# probabilities for the prefix given that it ends in a
			# blank and does not end in a blank at this time step.
			for prefix, (p_b, p_nb) in beam:  # Loop over beam

				# If we propose a blank the prefix doesn't change.
				# Only the probability of ending in blank gets updated.
				if s == blank:
					n_p_b, n_p_nb = next_beam[prefix]
					n_p_b = logsumexp(n_p_b, p_b + p, p_nb + p)
					next_beam[prefix] = (n_p_b, n_p_nb)
					continue

				# Extend the prefix by the new character s and add it to
				# the beam. Only the probability of not ending in blank
				# gets updated.
				end_t = prefix[-1] if prefix else None
				n_prefix = prefix + (s,)
				n_p_b, n_p_nb = next_beam[n_prefix]
				if s != end_t:
					n_p_nb = logsumexp(n_p_nb, p_b + p, p_nb + p)
				else:
					# We don't include the previous probability of not ending
					# in blank (p_nb) if s is repeated at the end. The CTC
					# algorithm merges characters not separated by a blank.
					n_p_nb = logsumexp(n_p_nb, p_b + p) ### strange case!!!

				# *NB* this would be a good place to include an LM score.

				lm_p = 0
				if t % 1 == 0:
					prefix_list = list(n_prefix)
					decoded_prefix = decode_labels(prefix_list, labels)
					joined_prefix = join_prefix(decoded_prefix)

					if len(joined_prefix) > 0:
						lm_p = lm.log_p(joined_prefix[-12:]) + 2

				next_beam[n_prefix] = (n_p_b + lm_p, n_p_nb + lm_p)

				# If s is repeated at the end we also update the unchanged
				# prefix. This is the merging case.
				if s == end_t:
					n_p_b, n_p_nb = next_beam[prefix]
					n_p_nb = logsumexp(n_p_nb, p_nb + p)
					next_beam[prefix] = (n_p_b, n_p_nb)

		# Sort and trim the beam before moving on to the
		# next time-step.
		beam = sorted(next_beam.items(),
				key=lambda x: logsumexp(*x[1]),
				reverse=True)
		beam = beam[:beam_size]

	best = beam[0]
	return best[0], -logsumexp(*best[1]), beam


if __name__ == "__main__":
	np.random.seed(4)

	time = 50
	output_dim = 20

	probs = np.random.rand(output_dim, time)
	probs = probs / np.sum(probs, axis=1, keepdims=True)

	labels, score = decode(probs, is_logprobs=False)
	print("Score {:.3f}".format(score))