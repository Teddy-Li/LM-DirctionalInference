import argparse
import os
import json


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='../log_files/', help='root directory of log files')
parser.add_argument('--prefix', type=str, default='levyholt_reparse_raw_', help='prefix of log files for the current configuration')
parser.add_argument('--out_fn', type=str, default='../log_files_aggr/levyholt_reparse_raw_aggr.txt', help='output file name')
parser.add_argument('--ckpt_root', type=str, default='../checkpoints/', help='Root directory of all model checkpoints.')
parser.add_argument('--ckpt_path', type=str, default=None, help='The path to all checkpoints from the current config at test.')
parser.add_argument('--keep_all', action='store_true', help='Whether to keep the checkpoints of all the models, or delete all but the best.')

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
		result_bucket = {'AUC': None, 'F1': None, 'Precision': None, 'Recall': None, 'val_loss': None}
		# print(f"First line!")
		for line in fp:
			if line.rstrip('\n') == '=== Best TEST Performance ====':
				test_flag = True
			elif line.rstrip('\n') == 'TEST RESULTS':
				assert result_flag is False
				if not test_flag:
					result_flag = True
			elif line.rstrip('\n') == '--------------------------------------------------------------------------------':
				result_flag = False
			elif result_flag is False or test_flag is True:
				pass
			elif result_flag is True:
				# print(line)
				try:
					if "'AUC': tensor" in line:
						pass
					elif "'AUCHALF': tensor" in line:
						auc = float(line.split('(')[1].split(')')[0].split(',')[0])
						if result_bucket['AUC'] is not None and auc != result_bucket['AUC']:
							print(fn)
							print(result_bucket)
							print(line)
							raise AssertionError
						result_bucket['AUC'] = auc
					elif "'AUCREL': tensor" in line:
						pass
					elif "'theta': tensor" in line:
						pass
					elif "'best F1': tensor" in line or "'F1': tensor" in line:
						f1 = float(line.split('(')[1].split(')')[0].split(',')[0])
						assert result_bucket['F1'] is None or f1 == result_bucket['F1']
						result_bucket['F1'] = f1
					elif "'Precision': tensor" in line:
						pr = float(line.split('(')[1].split(')')[0].split(',')[0])
						assert result_bucket['Precision'] is None or pr == result_bucket['Precision']
						result_bucket['Precision'] = pr
					elif "'Recall': tensor" in line:
						rec = float(line.split('(')[1].split(')')[0].split(',')[0])
						assert result_bucket['Recall'] is None or rec == result_bucket['Recall']
						result_bucket['Recall'] = rec
					elif "'val_loss': tensor" in line:
						vl = float(line.split('(')[1].split(')')[0].split(',')[0])
						assert result_bucket['val_loss'] is None or result_bucket['val_loss'] == vl
						result_bucket['val_loss'] = vl
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
				raise AssertionError

		if config_id in results_by_config_id:
			print(fn)
			raise AssertionError
		results_by_config_id[config_id] = result_bucket
		# print("")

results_by_config_id_auc = {k: v for (k, v) in sorted(results_by_config_id.items(), key=lambda x: x[1]['AUC'], reverse=True)}
# results_by_config_id_auchalf = {k: v for (k, v) in sorted(results_by_config_id.items(), key=lambda x: x[1]['AUCHALF'], reverse=True)}
results_by_config_id_f1 = {k: v for (k, v) in sorted(results_by_config_id.items(), key=lambda x: x[1]['F1'], reverse=True)}

max_f1_confid = None
for k in results_by_config_id_f1:
	max_f1_confid = k
	break
# max_auchalf_confid = None
# for k in results_by_config_id_auchalf:
# 	max_auchalf_confid = k
# 	break
max_auc_confid = None
for k in results_by_config_id_auc:
	max_auc_confid = k
	break

if args.ckpt_path is not None:
	print(os.path.join(args.ckpt_root, args.ckpt_path))

for k_idx, k in enumerate(results_by_config_id):
	print(f"config id: {k}")
	print(f"Results: ")
	print(results_by_config_id[k])
	print("")
	if k not in [max_f1_confid, max_auc_confid] and not args.keep_all:
		if args.ckpt_path is not None:
			ckpt_path = os.path.join(args.ckpt_root, args.ckpt_path) % int(k)
			for f in os.listdir(ckpt_path):
				if f.endswith('ckpt'):
					os.remove(os.path.join(ckpt_path, f))

	print(f"config id: {k}", file=out_fp)
	print(f"Results: ", file=out_fp)
	print(results_by_config_id[k], file=out_fp)
	print("", file=out_fp)

print(f"max_auc_confid: {max_auc_confid}")
# print(f"max_auchalf_confid: {max_auchalf_confid}")
print(f"max_f1_confid: {max_f1_confid}")
print(f"max_auc_confid: {max_auc_confid}", file=out_fp)
# print(f"max_auchalf_confid: {max_auchalf_confid}", file=out_fp)
print(f"max_f1_confid: {max_f1_confid}", file=out_fp)

out_fp.close()
