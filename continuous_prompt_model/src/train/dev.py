import argparse
import pytorch_lightning as pl
import sys
sys.path.append('src')
from models.multnat_model import MultNatModel
from utils import add_dataset_specific_args, load_custom_data
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--classification_threshold', type=float, default=None)
    parser.add_argument('--gpus', type=int, default=1)
    add_dataset_specific_args(parser)
    args = parser.parse_args()

    model = MultNatModel.load_from_checkpoint(args.checkpoint)

    if args.classification_threshold is not None:
        model.set_classification_threshold(args.classification_threshold)

    ckpt_root = '/'.join(args.checkpoint.split('/')[:-1])

    print(f"Testing on: '../datasets/data_en_levy_holt/levy_holt/dev.txt'")
    model.set_minimum_precision(0.19763)  # 217 / 1098
    model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_levyholt_dev_pr_rec.txt'))
    model.set_score_outfile(os.path.join(ckpt_root, 'en_levyholt_dev_Y.txt'))
    dataloader = load_custom_data(args, model, '../datasets/data_en_levy_holt/levy_holt/dev.txt')
    trainer = pl.Trainer(gpus=args.gpus, logger=False)
    trainer.test(model, test_dataloaders=dataloader)

    del dataloader
    del trainer

    print(f"Testing on: '../datasets/data_en_sherliic/levy_holt/dev.txt'")
    model.set_minimum_precision(0.34)  # 68 / 200
    model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_sherliic_dev_pr_rec.txt'))
    model.set_score_outfile(os.path.join(ckpt_root, 'en_sherliic_dev_Y.txt'))
    dataloader = load_custom_data(args, model, '../datasets/data_en_sherliic/levy_holt/dev.txt')
    trainer = pl.Trainer(gpus=args.gpus, logger=False)
    trainer.test(model, test_dataloaders=dataloader)

    del dataloader
    del trainer

    print(f"Testing on: '../datasets/data_levyholt_reparse_reparse/levy_holt/dev.txt'")
    model.set_minimum_precision(0.19763)  # 217 / 1098
    model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'zh_levyholt_rep_dev_pr_rec.txt'))
    model.set_score_outfile(os.path.join(ckpt_root, 'zh_levyholt_rep_dev_Y.txt'))
    dataloader = load_custom_data(args, model, '../datasets/data_levyholt_reparse_reparse/levy_holt/dev.txt')
    trainer = pl.Trainer(gpus=args.gpus, logger=False)
    trainer.test(model, test_dataloaders=dataloader)

    del dataloader
    del trainer

    print(f"Testing on: '../datasets/data_levyholt_raw_raw/levy_holt/dev.txt'")
    model.set_minimum_precision(0.19763)  # 217 / 1098
    model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'zh_levyholt_raw_dev_pr_rec.txt'))
    model.set_score_outfile(os.path.join(ckpt_root, 'zh_levyholt_raw_dev_Y.txt'))
    dataloader = load_custom_data(args, model, '../datasets/data_levyholt_raw_raw/levy_holt/dev.txt')
    trainer = pl.Trainer(gpus=args.gpus, logger=False)
    trainer.test(model, test_dataloaders=dataloader)

    del dataloader
    del trainer

    print(f"Testing on: '../datasets/data_sherliic_raw_raw/levy_holt/dev.txt'")
    model.set_minimum_precision(0.34)
    model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'zh_sherliic_raw_dev_pr_rec.txt'))
    model.set_score_outfile(os.path.join(ckpt_root, 'zh_sherliic_raw_dev_Y.txt'))
    dataloader = load_custom_data(args, model, '../datasets/data_sherliic_raw_raw/levy_holt/dev.txt')
    trainer = pl.Trainer(gpus=args.gpus, logger=False)
    trainer.test(model, test_dataloaders=dataloader)

    del dataloader
    del trainer

    print(f"Testing on: '../datasets/data_endir_levyholt/levy_holt/dev.txt'")
    model.set_minimum_precision(0.5)
    model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_levyholt_dir_dev_pr_rec.txt'))
    model.set_score_outfile(os.path.join(ckpt_root, 'en_levyholt_dir_dev_Y.txt'))
    dataloader = load_custom_data(args, model, '../datasets/data_endir_levyholt/levy_holt/dev.txt')
    trainer = pl.Trainer(gpus=args.gpus, logger=False)
    trainer.test(model, test_dataloaders=dataloader)

    del dataloader
    del trainer

    print(f"Testing on: '../datasets/data_zhdir_levyholt_raw/levy_holt/dev.txt'")
    model.set_minimum_precision(0.5)
    model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'zh_levyholt_raw_dir_dev_pr_rec.txt'))
    model.set_score_outfile(os.path.join(ckpt_root, 'zh_levyholt_raw_dir_dev_Y.txt'))
    dataloader = load_custom_data(args, model, '../datasets/data_zhdir_levyholt_raw/levy_holt/dev.txt')
    trainer = pl.Trainer(gpus=args.gpus, logger=False)
    trainer.test(model, test_dataloaders=dataloader)

    del dataloader
    del trainer

    print(f"Testing on: '../datasets/data_zhdir_levyholt_rep/levy_holt/dev.txt'")
    model.set_minimum_precision(0.5)
    model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'zh_levyholt_rep_dir_dev_pr_rec.txt'))
    model.set_score_outfile(os.path.join(ckpt_root, 'zh_levyholt_rep_dir_dev_Y.txt'))
    dataloader = load_custom_data(args, model, '../datasets/data_zhdir_levyholt_rep/levy_holt/dev.txt')
    trainer = pl.Trainer(gpus=args.gpus, logger=False)
    trainer.test(model, test_dataloaders=dataloader)