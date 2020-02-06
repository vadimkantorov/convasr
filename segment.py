
def resegment(c, r, h, rh, max_segment_seconds):
	rh_ = lambda rh, i, w, first, last: [(k, u) for k, u in enumerate(rh) if (first or i is None or u['begin'] >= rh[i]['end']) and (last or u['end'] <= w['end'])]
	rhk = [r, h].index(rh)
	i = [None, None]
	for j, w in enumerate(rh):
		if j == len(rh) - 1 or w['end'] - rh[i[rhk] or 0]['end'] > max_segment_seconds:
			first_last = dict(first = i[rhk] is None, last = j == len(rh) - 1)
			r_ = rh_(r, i[0], rh[j], **first_last); rk, r_ = zip(*r_) if r_ else ([i[0]], [])
			h_ = rh_(h, i[1], rh[j], **first_last); hk, h_ = zip(*h_) if h_ else ([i[1]], [])
			i = (rk[-1], hk[-1])
			yield [c, list(r_), list(h_)]

