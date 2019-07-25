import os
import json
import argparse
import base64

parser = argparse.ArgumentParser()
parser.add_argument('--transcripts', default = 'data/transcripts.json')
parser.add_argument('--vis')
args = parser.parse_args()

ref_tra = list(sorted(json.load(open(args.transcripts)), key = lambda j: j['cer']))

vis = open(args.vis or args.transcripts + '.html' , 'w')
vis.write(f'<html><meta charset="utf-8"><body><h1>{args.transcripts}</h1><table style="border-collapse:collapse"><thead><tr><th>cer</th><th>filename</th><th>audio</th><th><div>reference</div><div>transcript</div></th></tr></thead><tbody>')

for i, (reference, transcript, filename, cer) in enumerate(list(map(j.get, ['reference', 'transcript', 'filename', 'cer'])) for j in ref_tra):
    encoded = base64.b64encode(open(filename, 'rb').read()).decode('utf-8').replace('\n', '')
    vis.write(f'<tr><td style="border-right: 2px black solid">{cer:.02%}</td> <td style="font-size:xx-small">{os.path.basename(filename)}</td> <td><audio controls src="data:audio/wav;base64,{encoded}"/></td><td><div><b>{reference}</b></div><div>{transcript}</div></td></tr>\n')

vis.write('</tbody></table></body></html>')
