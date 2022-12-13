import argparse
import pytorch_lightning as pl
import sys
from src.models.multnat_model import MultNatModel
from src.data.levy_holt import LevyHoltPattern, LABEL_KEY, SENT_KEY, ANTI_KEY
import torch
import os



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint', type=str, required=True)
	parser.add_argument('--res_root', type=str, default=None)
	parser.add_argument('--lang', type=str, required=True)

	args = parser.parse_args()
	assert args.lang in ['EN', 'ZH']

	model = MultNatModel.load_from_checkpoint(args.checkpoint)
	dataset = LevyHoltPattern(txt_file=None, pattern_file=None, antipattern_file=None, best_k_patterns=None,
					   		  pattern_chunk_size=5, training=False, curated_auto=False, is_directional=True)
	pairs = [('I have a cat', 'I own a cat'), ('I own a cat', 'I have a cat')]

	batch_instances = []
	for prem, hypo in pairs:
		instance = dataset.create_instances((prem,'',''), (hypo,'',''), False, language=args.lang)

		assert len(instance) == 1
		instance = instance[0]
		batch_instances.append((instance[SENT_KEY], instance[ANTI_KEY], instance[LABEL_KEY]))

	batch_instances = model.collate(batch_instances)
	res_dict = model.validation_step(batch_instances, 0)
	assert torch.all((res_dict['scores'] > 0) & (res_dict['scores'] < 1))

	print(res_dict)

	# model.set_minimum_precision(0.21910)  # 2831 / 12921
	# model.set_pr_rec_curve_path(os.path.join(args.res_root, 'en_levyholt_test_pr_rec.txt'))
	# model.set_score_outfile(os.path.join(args.res_root, 'en_levyholt_test_Y.txt'))