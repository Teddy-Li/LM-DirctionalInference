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
    parser.add_argument('--range', type=str, default='all', help='[all / en / zh / ant / dirlbl / symlbl / dirsym / dirsym_posi]')
    add_dataset_specific_args(parser)
    args = parser.parse_args()

    if len(args.range) == 0:
        args.range = 'all'
    assert args.range in ['all', 'en', 'zh', 'ant', 'dirlbl', 'symlbl', 'dirsym', 'dirsym_posi', 'dirsym_negi']

    model = MultNatModel.load_from_checkpoint(args.checkpoint)

    if args.classification_threshold is not None:
        model.set_classification_threshold(args.classification_threshold)

    ckpt_root = '/'.join(args.checkpoint.split('/')[:-1])

    if args.range in ['all', 'en']:
        print(f"Testing on: '../datasets/data_en_levy_holt/levy_holt/test.txt'")
        model.set_minimum_precision(0.21910)  # 2831 / 12921
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_levyholt_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_levyholt_test_Y.txt'))
        dataloader = load_custom_data(args, model, '../datasets/data_en_levy_holt/levy_holt/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['all', 'en']:
        print(f"Testing on: '../datasets/data_en_sherliic/levy_holt/test.txt'")
        model.set_minimum_precision(0.33255)  # 994 / 2989
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_sherliic_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_sherliic_test_Y.txt'))
        dataloader = load_custom_data(args, model, '../datasets/data_en_sherliic/levy_holt/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['all', 'zh']:
        print(f"Testing on: '../datasets/data_levyholt_reparse_reparse/levy_holt/test.txt'")
        model.set_minimum_precision(0.21910)  # 2831 / 12921
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'zh_levyholt_rep_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'zh_levyholt_rep_test_Y.txt'))
        dataloader = load_custom_data(args, model, '../datasets/data_levyholt_reparse_reparse/levy_holt/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['all', 'zh']:
        print(f"Testing on: '../datasets/data_levyholt_raw_raw/levy_holt/test.txt'")
        model.set_minimum_precision(0.21910)  # 2831 / 12921
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'zh_levyholt_raw_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'zh_levyholt_raw_test_Y.txt'))
        dataloader = load_custom_data(args, model, '../datasets/data_levyholt_raw_raw/levy_holt/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['all', 'zh']:
        print(f"Testing on: '../datasets/data_sherliic_raw_raw/levy_holt/test.txt'")
        model.set_minimum_precision(0.33255)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'zh_sherliic_raw_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'zh_sherliic_raw_test_Y.txt'))
        dataloader = load_custom_data(args, model, '../datasets/data_sherliic_raw_raw/levy_holt/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['all', 'en']:
        print(f"Testing on: '../datasets/data_endir_levyholt/levy_holt/test.txt'")
        model.set_minimum_precision(0.5)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_levyholt_dir_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_levyholt_dir_test_Y.txt'))
        dataloader = load_custom_data(args, model, '../datasets/data_endir_levyholt/levy_holt/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['all', 'zh']:
        print(f"Testing on: '../datasets/data_zhdir_levyholt_raw/levy_holt/test.txt'")
        model.set_minimum_precision(0.5)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'zh_levyholt_raw_dir_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'zh_levyholt_raw_dir_test_Y.txt'))
        dataloader = load_custom_data(args, model, '../datasets/data_zhdir_levyholt_raw/levy_holt/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['all', 'zh']:
        print(f"Testing on: '../datasets/data_zhdir_levyholt_rep/levy_holt/test.txt'")
        model.set_minimum_precision(0.5)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'zh_levyholt_rep_dir_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'zh_levyholt_rep_dir_test_Y.txt'))
        dataloader = load_custom_data(args, model, '../datasets/data_zhdir_levyholt_rep/levy_holt/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)


        del dataloader
        del trainer

    if args.range in ['all', 'en']:
        print(f"Testing on: '../datasets/data_levyholt_directionals/test_en_sym.txt'")
        model.set_minimum_precision(0.1741)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_levyholt_sym_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_levyholt_sym_test_Y.txt'))
        dataloader = load_custom_data(args, model, '../datasets/data_levyholt_directionals/test_en_sym.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['all', 'zh']:
        print(f"Testing on: '../datasets/data_levyholt_directionals/test_zh_sym_raw.txt'")
        model.set_minimum_precision(0.1741)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'zh_levyholt_raw_sym_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'zh_levyholt_raw_sym_test_Y.txt'))
        dataloader = load_custom_data(args, model, '../datasets/data_levyholt_directionals/test_zh_sym_raw.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['all', 'zh']:
        print(f"Testing on: '../datasets/data_levyholt_directionals/test_zh_sym_rep.txt'")
        model.set_minimum_precision(0.1741)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'zh_levyholt_rep_sym_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'zh_levyholt_rep_sym_test_Y.txt'))
        dataloader = load_custom_data(args, model, '../datasets/data_levyholt_directionals/test_zh_sym_rep.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['all', 'en', 'ant']:
        print(f"Testing on: '../datasets/data_en_antonym/full/test.txt'")
        model.set_minimum_precision(0.4817)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_antonym_full_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_antonym_full_test_Y.txt'))
        dataloader = load_custom_data(args, model, '../datasets/data_en_antonym/full/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['all', 'en', 'ant']:
        print(f"Testing on: '../datasets/data_en_antonym/base/test.txt'")
        model.set_minimum_precision(0.4487)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_antonym_base_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_antonym_base_test_Y.txt'))
        dataloader = load_custom_data(args, model, '../datasets/data_en_antonym/base/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['all', 'en', 'ant']:
        print(f"Testing on: '../datasets/data_en_antonym/dir/test.txt'")
        model.set_minimum_precision(0.5)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_antonym_dir_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_antonym_dir_test_Y.txt'))
        dataloader = load_custom_data(args, model, '../datasets/data_en_antonym/dir/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['all', 'dirlbl']:
        print(f"Testing on: '../datasets/data_en_levy_holt/dirlabels/test.txt'")
        model.set_minimum_precision(0.0726)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_levyholt_dirlabels_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_levyholt_dirlabels_test_Y.txt'))
        dataloader = load_custom_data(args, model, '../datasets/data_en_levy_holt/dirlabels/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['all', 'symlbl']:
        print(f"Testing on: '../datasets/data_en_levy_holt/symlabels/test.txt'")
        model.set_minimum_precision(0.1348)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_levyholt_symlabels_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_levyholt_symlabels_test_Y.txt'))
        dataloader = load_custom_data(args, model, '../datasets/data_en_levy_holt/symlabels/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['dirsym']:
        print(f"Testing on: '../datasets/data_en_levy_holt/dirsym/test.txt'")
        model.set_minimum_precision(0.5160)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_levyholt_dirsym_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_levyholt_dirsym_test_Y.txt'))
        dataloader = load_custom_data(args, model, '../datasets/data_en_levy_holt/dirsym/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['dirsym_posi']:
        print(f"Testing on: '../datasets/data_en_levy_holt/dirsym_posi/test.txt'")
        model.set_minimum_precision(0.3500)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_levyholt_dirsym_posi_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_levyholt_dirsym_posi_test_Y.txt'))
        dataloader = load_custom_data(args, model, '../datasets/data_en_levy_holt/dirsym_posi/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['dirsym_negi']:
        print(f"Testing on: '../datasets/data_en_levy_holt/dirsym_negi/test.txt'")
        model.set_minimum_precision(0.091)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_levyholt_dirsym_negi_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_levyholt_dirsym_negi_test_Y.txt'))
        dataloader = load_custom_data(args, model, '../datasets/data_en_levy_holt/dirsym_negi/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['dirsym_cross']:
        print(f"Testing on: '../datasets/data_en_levy_holt/dirnegi_symposi/test.txt'")
        model.set_minimum_precision(0.6849)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_levyholt_symposi_dirnegi_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_levyholt_symposi_dirnegi_test_Y.txt'))
        dataloader = load_custom_data(args, model, '../datasets/data_en_levy_holt/dirnegi_symposi/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer


