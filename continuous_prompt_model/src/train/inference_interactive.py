import argparse
import pytorch_lightning as pl
import sys
sys.path.append('src')
from models.multnat_model import MultNatModel
from data.levy_holt import LevyHolt, LABEL_KEY, SENT_KEY, ANTI_KEY
import torch
import os


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint', type=str, required=True)
	parser.add_argument('--res_root', type=str, required=True)
	parser.add_argument('--num_patts', type=int, required=True)
	parser.add_argument('--num_toks_per_patt', type=int, required=True)
	parser.add_argument('--lang', type=str, required=True)

	args = parser.parse_args()
	assert args.lang in ['EN', 'ZH']

	model = MultNatModel.load_from_checkpoint(args.checkpoint)
	dataset = LevyHolt(txt_file=None, num_patterns=args.num_patts, num_tokens_per_pattern=args.num_toks_per_patt,
					   only_sep=False, use_antipatterns=True, training=False, pattern_chunk_size=5)
	pairs = [('I have a cat', 'I own a cat'), ('I own a cat', 'I have a cat')]

	batch_instances = []
	for prem, hypo in pairs:
		instance = dataset.create_instances((prem,'',''), (hypo,'',''), False, language=args.lang)

		assert len(instance) == 1
		instance = instance[0]
		batch_instances.append((instance[SENT_KEY], instance[ANTI_KEY], instance[LABEL_KEY]))

	batch_instances = model.collate(batch_instances)
	res_dict = model.validation_step(batch_instances, 0)
	assert torch.all((res_dict['scores'] > -1) & (res_dict['scores'] < 1))
	res_dict['scores'] = (res_dict['scores']+1)/2

	print(res_dict)

	# model.set_minimum_precision(0.21910)  # 2831 / 12921
	# model.set_pr_rec_curve_path(os.path.join(args.res_root, 'en_levyholt_test_pr_rec.txt'))
	# model.set_score_outfile(os.path.join(args.res_root, 'en_levyholt_test_Y.txt'))