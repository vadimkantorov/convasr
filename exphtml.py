import os
import sys
import time
import json
import random
import itertools
import subprocess

def expjson(root_dir, experiment_id, epoch = None, iteration = None, columns = {}, meta = {}, tag = '', name = None, git_revision = True, git_http = None):
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
	
	json_dir = os.path.join(root_dir, 'json')
	os.makedirs(json_dir, exist_ok = True)
	name = f'{int(time.time())}.{random.randint(10, 99)}.json' if name is None else name
	json_path = os.path.join(json_dir, name)
	json.dump(obj, open(json_path, 'w'), sort_keys = True, indent = 2, ensure_ascii = False)

def exphtml(root_dir, html_dir = 'public', strftime = '%Y-%m-%d %H:%M:%S', repeat = 5, timeout = 5):
	json_dir = os.path.join(root_dir, 'json')
	html_dir = os.path.join(root_dir, html_dir)
	os.makedirs(html_dir, exist_ok = True)
	html_path = os.path.join(html_dir, 'index.html')
	generated_time = time.strftime(strftime, time.gmtime())

	def json_load(path):
		try:
			j = json.load(open(path))
			j['path'] = path
			j['git_http'], j['git_revision'], j['git_comment'] = j.get('git_http', ''), j.get('git_revision', ''), j.get('git_comment', '')
			return j
		except:
			return {}

	groupby = lambda items, key: [(list(g), k) for k, g in itertools.groupby(sorted(items, key = key), key = key)]
	list_map = lambda *args: list(map(*args))
	map0 = lambda func, items: [(func(elem0), *_) for elem0, *_ in items]
	strip_hidden = lambda s: s.lstrip('.')
	hide = lambda s: '.' + strip_hidden(s)

	events = list(filter(None, (json_load(os.path.join(json_dir, json_file)) for json_file in os.listdir(json_dir))))
	by_experiment_id = lambda e: e['experiment_id']
	by_tag = lambda e: e['tag']
	by_time = lambda e: e['time']
	by_iteration = lambda e: (e['iteration'], e['time'])
	by_max_event_time = lambda exp: max(map(by_time, exp[0]))
	columns_union = lambda experiments: set(sum((list_map(strip_hidden, e['columns']) for events, *_ in experiments for e in events), []))
	fields_union = lambda experiments: set(sum((list_map(strip_hidden, c) for events, *_ in experiments for e in events for c in e['columns'].values()), []))
	last_event_by_column = lambda events, c: [e for e in events if c in map(strip_hidden, e['columns'])][-1]
	last_event_by_field = lambda events, f: [e for e in events if f in sum(map(list, e['columns'].values()), [])][-1]
	
	experiments = sorted(map0(lambda events: sorted(events, key = by_iteration), groupby(events, by_experiment_id)), key = by_max_event_time, reverse = True)
	experiments_ids = [experiment_id for events, experiment_id in experiments]

	columns = sorted(columns_union(experiments))
	fields = sorted(fields_union(experiments))

	experiment_recent = experiments[0]
	columns_recent = columns_union([experiment_recent])
	fields_recent = columns_union([experiment_recent])

	columns_hidden = {c : c not in columns_recent or hide(c) in last_event_by_column(experiment_recent[0], c) for c in columns}
	fields_hidden =  {f : f not in fields_recent or hide(f) in last_event_by_field(experiment_recent[0], f) for f in fields}

	with open(html_path, 'w') as html:
		html.write('<html>')
		html.write('<head>')
		html.write(f'<title>Results @ {generated_time}</title>')
		html.write('''
			<meta http-equiv="refresh" content="600" />
			<script src="https://cdn.jsdelivr.net/npm/vega@5.8.1"></script>
			<script src="https://cdn.jsdelivr.net/npm/vega-lite@4.0.0-beta.12"></script>
			<script src="https://cdn.jsdelivr.net/npm/vega-embed@6.1.0"></script>
			<style>
				.nowrap {white-space:nowrap}
				.m0 {margin:0px}
				.textleft {text-align:left}
			</style>
		''')
		html.write('</head>')
		html.write('<body onload="onload()">')
		html.write('''
		<script>
			var toggle = className => Array.from(document.querySelectorAll(`.${className}`)).map(e => {e.hidden = !e.hidden});

			function onload()
			{
				const hash = window.location.hash.replace('#', '');
				const parts = hash.length > 0 ? hash.split(';') : [];
				
				parts
					.map(p => p.split('='))
					.map(([prefix, selector]) =>
					{
						if(selector)
						{
							Array.from(document.querySelectorAll(`input[value^=${prefix}]:not([name*=${selector}])`)).map(checkbox => checkbox.click());
							document.getElementById(`filter_${prefix}`).value = selector;
						}
					});
			}

			function filter(event)
			{
				if (event.keyCode === 13)
				{
					event.preventDefault();
					event.target.nextSibling.click();
				}
			}

			function filter_(event)
			{
				const filter_field = document.getElementById('filter_field').value, filter_col = document.getElementById('filter_col').value, filter_exp = document.getElementById('filter_exp').value;
				window.location.hash = `field=${filter_field};col=${filter_col};exp=${filter_exp} `.replace('field=;', '').replace('col=;', '').replace('exp= ', '').trimEnd();
				
				window.location.reload();
				event.preventDefault();
				return false;
			}
		</script>''')
		html.write(f'<h1>Generated at {generated_time}</h1>')
		html.write('<table width="100%">')
	
		def render_header_line(name, names, hidden):
			return f'<tr><th class="textleft">{name}</th><td><div id="searchbox"><form action="." class="m0"><label class="nowrap"><input id="filter_{name}" type="text" name="search" placeholder="Filter" onkeyup="return filter(event)"><button onclick="return filter_(event)">Filter</button></label></form></div></td><td>' + (''.join(f'<label class="nowrap"><input type="checkbox" name="{c}" value="{name}{hash(c)}" {"checked" if not hidden[c] else ""} onchange="toggle(event.target.value)" />{c}</label>' for c in names) if hidden is not True else '') + '</td></tr>'

		def render_experiment(events, experiment_id):
			generated_time = time.strftime(strftime, time.localtime(events[-1]['time']))
			return f'''<tr><td title="{generated_time}" onclick="toggle('{experiment_id}.hidden')"><strong>{experiment_id}</strong></td>''' + ''.join(f'<td class="col{hash(c)}"><strong>{c}</strong></td>' for c in columns) + '</tr>'

		def render_cell():
			return '{:.04f}'.format(o) if isinstance(o, float) else '''<a href="_blank" onclick="toggle('{key}'); return false">{name}</a>'''.format(key = key, **o) if isinstance(o, dict) else str(o)

		html.write(render_header_line('fields', fields, fields_hidden))
		html.write(render_header_line('columns', columns, columns_hidden))
		html.write(render_header_line('experiments', experiments_ids, True))
		# TODO: add tags, change hidden to checked
		html.write('</table><hr/>')

		#key = lambda experiment_id, iteration = '', column = '', field = '': f'flyout{experiment_id}_{iteration}_{column}_{field}'.replace('.', '_')
		
		html.write('<table cellpadding="2px" cellspacing="0">')
		for experiment in experiments:
			html.write(render_experiment(*experiment))
		#	idx = set([0, len(jsons) - 1] + [i for i, j in enumerate(jsons) if 'iter' not in j['iteration']])

		#	for i, j in enumerate(jsons):
		#		generated_time = time.strftime(strftime, time.localtime(j['time']))
		#		hidden = 'hidden' if i not in idx else ''
		#		meta_key = f'meta{hash(experiment_id + str(j["iteration"]))}'
		#		experiment_key = f'data-experiment-id="{experiment_id}"'
		#		iteration = j.get('iteration', '')

		#		meta = json.dumps(j['meta'], sort_keys = True, indent = 2, ensure_ascii = False) if j.get('meta') else None
		#		html.write(f'<tr {experiment_key} class="{hidden} {experiment_id}" {hidden}>')
		#		html.write(f'''<td onclick="toggle('{meta_key}')" title="{generated_time}" style="border-right: 1px solid black">{iteration}</td>''')
		#		html.write(''.join(f'<td class="col{hash(c)}">' + ''.join(f'<span title="{f}" style="margin-right:3px" {"hidden" if f != field else ""} class="field{hash(f)}">{fmt(j["columns"].get(c, {}).get(f, ""), key = key(experiment_id, iteration, c, f))}</span>' for f in fields) + '</td>' for c in columns))
		#		html.write('</tr>\n')

		#		for c in columns:
		#			for f in fields:
		#				val = j['columns'].get(c, {}).get(f, '')
		#				if isinstance(val, dict):
		#					flyout_key = key(experiment_id, iteration, c, f)
		#					html.write('<tr {experiment_key} hidden class="{flyout_key}" style="background-color:lightcoral"><td colspan="100"><pre>{flyout}</pre></td></tr>\n'.format(flyout = val['flyout'], flyout_key = flyout_key, experiment_key = experiment_key))
		#		
		#		html.write('<tr {experiment_key} hidden class="{meta_key}" style="background-color:lightblue"><td><a href="{git_http}">@{git_revision}</a></td><td colspan="100">{git_comment}</td></tr>\n'.format(meta_key = meta_key, experiment_key = experiment_key, **j))
		#		html.write(f'<tr {experiment_key} hidden class="{meta_key}" style="background-color:lightgray"><td colspan="100"><pre>{meta}</pre></td></tr>\n' if meta else '')

		# 	html.write('<tr><td>&nbsp;</td> </tr>')
		
		html.write('</table></body></html>')

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
