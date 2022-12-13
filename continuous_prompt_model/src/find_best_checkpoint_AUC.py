import argparse
import os
import json


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='../log_files/')
parser.add_argument('--prefix', type=str, default='levyholt_reparse_raw_')
parser.add_argument('--out_fn', type=str, default='../log_files_aggr/levyholt_reparse_raw_aggr.txt')
parser.add_argument('--ckpt_root', type=str, default='../checkpoints/')
parser.add_argument('--ckpt_path', type=str, default=None)
parser.add_argument('--keep_all', action='store_true')

args = parser.parse_args()

results_by_config_id = {}

files = os.listdir(args.root)
files.sort()

out_fp = open(args.out_fn, 'w', encoding='utf8')

for fn in files:
	if not fn.startswith(args.prefix):
		continue
	postfix = fn[len(args.prefix):]
	config_id = postfix.split('_')[0]
	with open(os.path.join(args.root, fn), 'r', encoding='utf8') as fp:
		result_flag = False
		test_flag = False  # after this flag has been set to True, the results are test set and should not be considered here!
		result_bucket = {'AUCBSLN': None, 'AUCHALF': None, 'AUCREL': None, 'F1': None, 'Precision': None, 'Recall': None, 'theta': None, 'val_loss': None}

		for line in fp:
			if line.rstrip('\n') == '=== Best TEST Performance ====':
				test_flag = True
			elif line.rstrip('\n') == 'DATALOADER:0 TEST RESULTS':
				assert result_flag is False
				if not test_flag:
					result_flag = True
			elif line.rstrip('\n') == '--------------------------------------------------------------------------------':
				result_flag = False
			elif result_flag is False or test_flag is True:
				pass
			elif result_flag is True:
				try:
					if "'AUCBSLN': " in line:
						assert result_bucket['AUCBSLN'] is None
						result_bucket['AUCBSLN'] = float(line.split(':')[1].split(',')[0].split('}')[0])
					elif "'AUCHALF': " in line:
						assert result_bucket['AUCHALF'] is None
						result_bucket['AUCHALF'] = float(line.split(':')[1].split(',')[0].split('}')[0])
					elif "'AUCREL': " in line:
						assert result_bucket['AUCREL'] is None
						result_bucket['AUCREL'] = float(line.split(':')[1].split(',')[0].split('}')[0])
					elif "'theta': " in line:
						assert result_bucket['theta'] is None
						result_bucket['theta'] = float(line.split(':')[1].split(',')[0].split('}')[0])
					elif "'best F1': " in line or "'F1': " in line:
						assert result_bucket['F1'] is None
						result_bucket['F1'] = float(line.split(':')[1].split(',')[0].split('}')[0])
					elif "'Precision': " in line:
						assert result_bucket['Precision'] is None
						result_bucket['Precision'] = float(line.split(':')[1].split(',')[0].split('}')[0])
					elif "'Recall': " in line:
						assert result_bucket['Recall'] is None
						result_bucket['Recall'] = float(line.split(':')[1].split(',')[0].split('}')[0])
					elif "'val_loss': " in line:
						assert result_bucket['val_loss'] is None
						result_bucket['val_loss'] = float(line.split(':')[1].split(',')[0].split('}')[0])
					else:
						print(line)
						raise AssertionError
				except ValueError as e:
					print(line)
					print(fn)
					print(e)
					raise
			else:
				raise AssertionError
		for key in result_bucket:
			if result_bucket[key] is None:
				print(f"fn: {fn}")
				print(result_bucket)
				if key not in ['AUCREL']:
					raise AssertionError

		if config_id in results_by_config_id:
			print(fn)
			raise AssertionError
		results_by_config_id[config_id] = result_bucket

results_by_config_id_auc = {k: v for (k, v) in sorted(results_by_config_id.items(), key=lambda x: x[1]['AUCHALF'], reverse=True)}
# results_by_config_id_auchalf = {k: v for (k, v) in sorted(results_by_config_id.items(), key=lambda x: x[1]['AUCHALF'], reverse=True)}
# results_by_config_id_f1 = {k: v for (k, v) in sorted(results_by_config_id.items(), key=lambda x: x[1]['F1'], reverse=True)}

# max_f1_confid = None
# for k in results_by_config_id_f1:
# 	max_f1_confid = k
# 	break
# max_auchalf_confid = None
# for k in results_by_config_id_auchalf:
# 	max_auchalf_confid = k
# 	break
top5_auc_confid = []
for k in results_by_config_id_auc:
	top5_auc_confid.append(k)
	if len(top5_auc_confid) >= 5:
		break

if args.ckpt_path is not None:
	print(os.path.join(args.ckpt_root, args.ckpt_path))

for k_idx, k in enumerate(results_by_config_id_auc):
	print(f"config id: {k}")
	print(f"Results: ")
	print(results_by_config_id[k])
	print("")
	if k not in top5_auc_confid and not args.keep_all:
		if args.ckpt_path is not None:
			ckpt_path = os.path.join(args.ckpt_root, args.ckpt_path) % int(k)
			for f in os.listdir(ckpt_path):
				if f.endswith('ckpt'):
					os.remove(os.path.join(ckpt_path, f))

	print(f"config id: {k}", file=out_fp)
	print(f"Results: ", file=out_fp)
	print(results_by_config_id[k], file=out_fp)
	print("", file=out_fp)

print(f"max_auc_confid: {top5_auc_confid}")
# print(f"max_auchalf_confid: {max_auchalf_confid}")
# print(f"max_f1_confid: {max_f1_confid}")
print(f"max_auc_confid: {top5_auc_confid}", file=out_fp)
# print(f"max_auchalf_confid: {max_auchalf_confid}", file=out_fp)
# print(f"max_f1_confid: {max_f1_confid}", file=out_fp)

out_fp.close()
