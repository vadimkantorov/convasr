import os
import sys
import time
import json
import random
import itertools
import subprocess


def expjson(
	root_dir,
	experiment_id,
	epoch = None,
	iteration = None,
	columns = {},
	meta = {},
	tag = '',
	name = None,
	git_revision = True,
	git_http = None
):
	if git_revision is True:
		try:
			git_revision, git_comment = map(lambda b: b.decode('utf-8'), subprocess.check_output(['git', 'log', '--format=%h%x00%s', '--no-decorate', '-1']).split(b'\x00'))
		except:
			git_revision, git_comment = 'error', 'error'
	else:
		git_revision, git_comment = ''

	obj = dict(
		experiment_id = experiment_id,
		iteration = f'epoch{epoch:02d}_iter{iteration:07d}' if epoch is not None and iteration is not None else 'test',
		columns = columns,
		time = int(time.time()),
		meta = meta,
		git_revision = git_revision,
		git_comment = git_comment,
		git_http = git_http.replace('%h', git_revision) if git_http else None,
		tag = tag
	)

	json_dir = os.path.join(root_dir, 'events')
	os.makedirs(json_dir, exist_ok = True)
	name = f'{int(time.time())}.{random.randint(10, 99)}.json' if name is None else name
	json_path = os.path.join(json_dir, name)
	json.dump(obj, open(json_path, 'w'), sort_keys = True, indent = 2, ensure_ascii = False)
	return json_path


def exphtml(root_dir, html_dir = 'public', strftime = '%Y-%m-%d %H:%M:%S', repeat = 5, timeout = 5):
	json_dir = os.path.join(root_dir, 'events')
	html_dir = os.path.join(root_dir, html_dir)
	os.makedirs(html_dir, exist_ok = True)
	html_path = os.path.join(html_dir, 'index.html')
	generated_time = time.strftime(strftime, time.gmtime())

	def json_load(path):
		try:
			j = json.load(open(path))
			j['path'] = path
			j['meta'], j['tag'], j['iteration'], j['git_http'], j['git_revision'], j['git_comment'] = j.get('meta', {}), j.get('tag') or 'default', j.get('iteration', ''), j.get('git_http', ''), j.get('git_revision', ''), j.get('git_comment', '')
			return j
		except:
			return {}

	groupby = lambda items, key, key_sorted = None: [(list(g), k) for k, g in itertools.groupby(sorted(items, key = key_sorted or key), key = key)]
	list_map = lambda *args: list(map(*args))
	map0 = lambda func, items: [(func(elem0), *_) for elem0, *_ in items]
	strip_hidden = lambda s: s.lstrip('.')
	hide = lambda s: '.' + strip_hidden(s)

	random_key = lambda: random.randint(1, int(1e9))
	events = list(filter(None, (json_load(os.path.join(json_dir, json_file)) for json_file in os.listdir(json_dir))))
	by_experiment_id = lambda e: e['experiment_id']
	by_tag = lambda e: e['tag']
	by_time = lambda e: e['time']
	by_iteration = lambda e: (e['iteration'], e['time'])
	by_max_event_time = lambda exp: max(map(by_time, exp[0]))
	columns_union = lambda experiments: set(
		sum((list_map(strip_hidden, e['columns']) for events in experiments for e in events), [])
	)
	fields_union = lambda experiments: set(
		sum((list_map(strip_hidden, c) for events in experiments for e in events for c in e['columns'].values()), [])
	)
	tags_union = lambda experiments: set(e['tag'] for events in experiments for e in events)
	last_event_by_column = lambda events, c: ([dict(columns = {})] + [e for e in events for c_ in [map(strip_hidden, e['columns'])] if c in c_ or hide(c) in c_])[-1]
	last_event_by_field = lambda events, f: ([dict(columns = {})] + [e for e in events for f_ in [sum(map(list, e['columns'].values()), [])] if f in f_ or hide(f) in f_])[-1]

	experiments, experiments_id = zip(*sorted(map0(lambda events: sorted(events, key = by_iteration), groupby(events, by_experiment_id)), key = by_max_event_time, reverse = True))

	columns = sorted(columns_union(experiments))
	fields = sorted(fields_union(experiments))
	tags = sorted(tags_union(experiments))

	experiment_recent = experiments[0]
	columns_recent = columns_union([experiment_recent[-1:]])
	fields_recent = fields_union([experiment_recent[-1:]])

	columns_checked = {c: c in columns_recent and hide(c) not in columns_recent for c in columns}
	fields_checked = {f: f in fields_recent and hide(f) not in fields_recent for f in fields}

	with open(html_path, 'w') as html:
		html.write(f'<html><head><title>Results @ {generated_time}</title>')
		html.write(
			'''
			<meta charset="utf-8" />
			<meta http-equiv="refresh" content="600" />
			<script src="https://cdn.jsdelivr.net/npm/vega@5.8.1"></script>
			<script src="https://cdn.jsdelivr.net/npm/vega-lite@4.0.0-beta.12"></script>
			<script src="https://cdn.jsdelivr.net/npm/vega-embed@6.1.0"></script>
			<style>
				.nowrap {white-space:nowrap}
				.m0 {margin:0px}
				.textleft {text-align:left}
				.mr {margin-right:3px}
				.mt {margin-top: 20px}
				.sepright {border-right: 1px solid black}
				.git {background:lightblue}
				.flyout {background:lightgray}
				.odd {background:darkgray}
				.vt {vertical-align:top}
			</style>
		'''
		)
		html.write('</head>')
		html.write('<body onload="onload()">')
		html.write(
			'''
		<script>
			var toggle = selector => Array.from(document.querySelectorAll(`${selector}`)).map(e => {e.hidden = !e.hidden});

			function onload()
			{
			}
		</script>'''
		)
		html.write(f'<h1>Generated at {generated_time}</h1>')

		def render_expand(name, selector, prepend = '', expand_char = '±'):
			return f'''<a href="#" onclick='{prepend} toggle("{selector}"); return false'>{name}{expand_char}</a>'''

		def render_header_line(name, names, checked):
			return f'<tr><th class="textleft">{name}</th><td><label class="nowrap"><input type="search" name="{name}" placeholder="Filter"></label></td><td>' + (
				''.join(
					f'<label class="nowrap"><input type="checkbox" onchange="toggle(event.target.value)" value=".{name}{hash(c)}" name="{c}" {"checked" if checked is True or checked[c] else ""} />{c}</label>'
					for c in names
				) if checked is not False else ''
			) + '</td></tr>'

		def render_value(val):
			if isinstance(val, int) or isinstance(val, float):
				return '{:.04f}'.format(val)
			elif isinstance(val, dict):
				return '<pre>' + json.dumps(val, sort_keys = True, indent = 2, ensure_ascii = False) + '</pre>'
			elif isinstance(val, list):
				return '<table style="width:100%"><tr>' + ''.join(
					f'<th>{f}</th>' for f in val[0]
				) + '</tr>' + '\n'.join(
					'<tr>' + ''.join(f'<td>{render_value(f)}</td>' for f in row) + '</tr>' for row in val[1:]
				) + '</table>' if len(val) > 0 else ''
			else:
				return str(val)

		def render_cell(event, column, field, append = []):
			val = event['columns'].get(column, {}).get(field, '')
			if isinstance(val, dict):
				key = f'cell{random_key()}'
				prepend = f'(!document.getElementById("{key}_").classList.contains("vega-embed")) && vegaEmbed("#{key}_", JSON.parse(document.querySelector("#{key}_ pre").innerText));' if isinstance(
					val.get('value'), dict
				) and 'vega' in val['value'].get('$schema', '') else ''
				cell = render_expand(val.get('name', field), f'#{key}', prepend)
				append.append(
					f'<tr hidden id="{key}" class="flyout"><td></td><td class="vt">{render_expand(column + " > " + field, "#" + key)}</td><td colspan="100" id="{key}_">{render_value(val["value"])}</td></tr>'
				)
			else:
				cell = render_value(val)
			return cell

		hidden = lambda checked: "hidden" if not checked else ""

		def render_experiment(events, experiment_id):
			generated_time = time.strftime(strftime, time.localtime(events[-1]['time']))
			key_exp = f'experiments{hash(experiment_id)}'
			res = f'''<tr class="mt"><td><strong>tag</strong></td><td title="{generated_time}"><strong>{render_expand(experiment_id, '.' + key_exp)}</strong></td>''' + ''.join(
				f'<td {hidden(columns_checked[c])} class="columns{hash(c)}"><strong>{c}</strong></td>' for c in columns
			) + '</tr>'
			for i, (events_by_tag, tag) in enumerate(groupby(events, by_tag, by_time)):
				for j, e in enumerate(events_by_tag):
					generated_time = time.strftime(strftime, time.localtime(e['time']))
					meta = json.dumps(e['meta'], sort_keys = True, indent = 2, ensure_ascii = False)
					key_meta = f'meta{random_key()}'
					key_tag = f'tags{hash(tag)}'
					show = j == 0 or j == len(events_by_tag) - 1
					odd = i % 1 == 1
					res += f'''<tr class="{'odd' if odd else ''} {key_tag} {key_exp}" {'hidden' if not show else ''}><td>{tag}</td><td title="{generated_time}" class="sepright">{render_expand(e["iteration"], "." + key_meta)}</td>'''
					append = []
					res += ''.join(
						f'<td {hidden(columns_checked[c])} class="columns{hash(c)}">' + ''.join(
							f'<span title="{f}" class="mr fields{hash(f)}" {hidden(fields_checked[f])}>{render_cell(e, c, f, append)}</span>'
							for f in fields
						) + '</td>'
						for c in columns
					)
					res += '</tr>'
					res += ''.join(append)
					res += '<tr class="git {key_meta}" hidden><td></td><td><a href="{git_http}">commit: @{git_revision}</a></td><td colspan="100">message: {git_comment}</td></tr>\n'.format(
						key_meta = key_meta, **e
					)
					res += f'<tr class="flyout {key_meta}" hidden><td colspan="100"><pre>{meta}</pre></td></tr>' if e[
						'meta'] else ''
			return res

		html.write('<form action="."><table width="100%">')
		html.write(render_header_line('fields', fields, fields_checked))
		html.write(render_header_line('columns', columns, columns_checked))
		html.write(render_header_line('experiments', experiments_id, False))
		html.write(render_header_line('tags', tags, True))
		html.write('<tr><td></td><td><input type="submit" value="Filter" style="width:100%" /></td></tr>')
		html.write(
			'</table></form><hr/><table cellpadding="2px" cellspacing="0">' +
			''.join(map(render_experiment, experiments, experiments_id)) + '</table></body></html>'
		)

		try:
			subprocess.check_call(['git', 'add', '-A'], cwd = root_dir)
			subprocess.check_call(['git', 'commit', '-a', '--allow-empty-message', '-m', ''], cwd = root_dir)
			for i in range(repeat):
				try:
					subprocess.check_call(['git', 'pull'], cwd = root_dir)
					subprocess.check_call(['git', 'push'], cwd = root_dir)
					break
				except:
					print(sys.exc_info())
		except:
			print(sys.exc_info())


if __name__ == '__main__':
	exphtml(sys.argv[1])
