import argparse
import pytorch_lightning as pl
import os
import sys
sys.path.append('src')
from models.nli_model import NLIModel
from models.multnat_model import MultNatModel
from train.utils import add_dataset_specific_args, load_custom_data

str2model = {
    "NLI": NLIModel,
    "MultNat": MultNatModel
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=str2model.keys())
    parser.add_argument('checkpoint')
    parser.add_argument('--classification_threshold', type=float, default=None)
    parser.add_argument('--gpus', type=int, nargs='+', default=[])
    parser.add_argument('--range', type=str, default='all', help='[all / en / zh / ant / dirlbl / symlbl / dirsym / dirsym_posi]')
    parser.add_argument('--is_directional', action='store_true')
    add_dataset_specific_args(parser)
    args = parser.parse_args()
    print(args)

    if len(args.range) == 0:
        args.range = 'all'
    assert args.range in ['all', 'en', 'zh', 'zhraw', 'ant', 'dirlbl', 'symlbl', 'dirsym', 'dirsym_posi', 'dirsym_negi',
                          'oneposi_allnegi', 'dirsym_cross', 'except_unrelated', 'hypo_only', 'zh_hypoonly', 'hypo_only_dirsym',
                          'hypo_only_dirsym_cross', 'hypo_only_oneposi_allnegi', 'hypo_only_random', 'qaeval_hyponly_namearg',
                          'qaeval_hyponly_typearg', 'qaeval_hyponly_nick', 'qaeval_hyponly_namearg_freqmap',
                          'qaeval_hyponly_typearg_freqmap', 'clue_qaeval_hyponly_namearg', 'clue_qaeval_hyponly_typearg',
                          'clue_qaeval_hyponly_typearg_freqmap', 'clue_qaeval_hyponly_namearg_freqmap', 'nc_qaeval_all',
                          'clue_qaeval_all', 'qaeval_hyponly_namearg_radical', 'qaeval_hyponly_typearg_radical']

    cls = str2model[args.model]
    model = cls.load_from_checkpoint(args.checkpoint)
    print(f"Model loaded.")

    if args.classification_threshold is not None:
        model.set_classification_threshold(args.classification_threshold)

    ckpt_root = '/'.join(args.checkpoint.split('/')[:-1])

    if args.range in ['all', 'en']:
        print(f"Testing on: '../datasets/data_en_levy_holt/orig/test.txt'")
        model.set_minimum_precision(0.21910)  # 2831 / 12921
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_levyholt_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_levyholt_test_Y.txt'))
        dataloader = load_custom_data(args, args.model, model, '../datasets/data_en_levy_holt/orig/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['all', 'en']:
        print(f"Testing on: '../datasets/data_en_sherliic/orig/test.txt'")
        model.set_minimum_precision(0.33255)  # 994 / 2989
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_sherliic_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_sherliic_test_Y.txt'))
        dataloader = load_custom_data(args, args.model, model, '../datasets/data_en_sherliic/orig/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['all', 'en']:
        print(f"Testing on: '../datasets/data_en_levy_holt/dirposi_dirnegi/test.txt'")
        model.set_minimum_precision(0.5)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_levyholt_dir_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_levyholt_dir_test_Y.txt'))
        dataloader = load_custom_data(args, args.model, model, '../datasets/data_en_levy_holt/dirposi_dirnegi/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['all', 'en']:
        print(f"Testing on: '../datasets/data_en_levy_holt/symposi_symnegi/test.txt'")
        model.set_minimum_precision(0.1741)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_levyholt_sym_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_levyholt_sym_test_Y.txt'))
        dataloader = load_custom_data(args, args.model, model, '../datasets/data_en_levy_holt/symposi_symnegi/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['dirsym_posi', 'en']:
        print(f"Testing on: '../datasets/data_en_levy_holt/symposi_dirposi/test.txt'")
        model.set_minimum_precision(0.3500)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_levyholt_symposi_dirposi_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_levyholt_symposi_dirposi_test_Y.txt'))
        dataloader = load_custom_data(args, args.model, model, '../datasets/data_en_levy_holt/symposi_dirposi/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['dirsym_negi', 'en']:
        print(f"Testing on: '../datasets/data_en_levy_holt/dirnegi_symnegi/test.txt'")
        model.set_minimum_precision(0.091)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_levyholt_dirnegi_symnegi_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_levyholt_dirnegi_symnegi_test_Y.txt'))
        dataloader = load_custom_data(args, args.model, model, '../datasets/data_en_levy_holt/dirnegi_symnegi/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['dirsym_cross', 'en']:
        print(f"Testing on: '../datasets/data_en_levy_holt/dirposi_symnegi/test.txt'")
        model.set_minimum_precision(0.0884)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_levyholt_dirposi_symnegi_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_levyholt_dirposi_symnegi_test_Y.txt'))
        dataloader = load_custom_data(args, args.model, model, '../datasets/data_en_levy_holt/dirposi_symnegi/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['dirsym_cross', 'en']:
        print(f"Testing on: '../datasets/data_en_levy_holt/symposi_dirnegi/test.txt'")
        model.set_minimum_precision(0.6849)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_levyholt_symposi_dirnegi_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_levyholt_symposi_dirnegi_test_Y.txt'))
        dataloader = load_custom_data(args, args.model, model, '../datasets/data_en_levy_holt/symposi_dirnegi/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['hypo_only']:
        print(f"Testing on: '../datasets/data_en_levy_holt/dirposi_dirnegi_hypo_only/test.txt'")
        model.set_minimum_precision(0.5)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_dirposi_dirnegi_hypo_only_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_dirposi_dirnegi_hypo_only_test_Y.txt'))
        dataloader = load_custom_data(args, args.model, model, '../datasets/data_en_levy_holt/dirposi_dirnegi_hypo_only/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['hypo_only']:
        print(f"Testing on: '../datasets/data_en_levy_holt/orig_hypo_only/test.txt'")
        model.set_minimum_precision(0.2191)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_levyholt_orig_hypo_only_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_levyholt_orig_hypo_only_test_Y.txt'))
        dataloader = load_custom_data(args, args.model, model, '../datasets/data_en_levy_holt/orig_hypo_only/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['hypo_only']:
        print(f"Testing on: '../datasets/data_en_levy_holt/symposi_symnegi_hypo_only/test.txt'")
        model.set_minimum_precision(0.1741)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_symposi_symnegi_hypo_only_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_symposi_symnegi_hypo_only_test_Y.txt'))
        dataloader = load_custom_data(args, args.model, model, '../datasets/data_en_levy_holt/symposi_symnegi_hypo_only/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['hypo_only']:
        print(f"Testing on: '../datasets/data_en_levy_holt/symposi_dirposi_hypo_only/test.txt'")
        model.set_minimum_precision(0.3500)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_symposi_dirposi_hypo_only_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_symposi_dirposi_hypo_only_test_Y.txt'))
        dataloader = load_custom_data(args, args.model, model, '../datasets/data_en_levy_holt/symposi_dirposi_hypo_only/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['hypo_only']:
        print(f"Testing on: '../datasets/data_en_levy_holt/symnegi_dirnegi_hypo_only/test.txt'")
        model.set_minimum_precision(0.091)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_symnegi_dirnegi_hypo_only_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_symnegi_dirnegi_hypo_only_test_Y.txt'))
        dataloader = load_custom_data(args, args.model, model, '../datasets/data_en_levy_holt/symnegi_dirnegi_hypo_only/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['hypo_only']:
        print(f"Testing on: '../datasets/data_en_levy_holt/dirposi_symnegi_hypo_only/test.txt'")
        model.set_minimum_precision(0.0884)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_dirposi_symnegi_hypo_only_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_dirposi_symnegi_hypo_only_test_Y.txt'))
        dataloader = load_custom_data(args, args.model, model, '../datasets/data_en_levy_holt/dirposi_symnegi_hypo_only/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['hypo_only']:
        print(f"Testing on: '../datasets/data_en_levy_holt/symposi_dirnegi_hypo_only/test.txt'")
        model.set_minimum_precision(0.6849)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_symposi_dirnegi_hypo_only_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_symposi_dirnegi_hypo_only_test_Y.txt'))
        dataloader = load_custom_data(args, args.model, model, '../datasets/data_en_levy_holt/symposi_dirnegi_hypo_only/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['qaeval_hyponly_namearg', 'nc_qaeval_all']:
        print(f"Testing on: '../datasets/data_qaeval_15_30_0_30_0_0/hypoonly_namearg_lhsize/test.txt'")
        model.set_minimum_precision(0.346723)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_qaeval_15_30_0_30_0_0_hyponly_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_qaeval_15_30_0_30_0_0_hyponly_test_Y.txt'))
        dataloader = load_custom_data(args, args.model, model,
                                      '../datasets/data_qaeval_15_30_0_30_0_0/hypoonly_namearg_lhsize/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['qaeval_hyponly_namearg_freqmap', 'nc_qaeval_all']:
        print(f"Testing on: '../datasets/data_qaeval_15_30_0_30_0_freqmap/hypoonly_namearg_lhsize/test.txt'")
        model.set_minimum_precision(0.3627)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_qaeval_15_30_0_30_0_freqmap_hyponly_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_qaeval_15_30_0_30_0_freqmap_hyponly_test_Y.txt'))
        dataloader = load_custom_data(args, args.model, model,
                                      '../datasets/data_qaeval_15_30_0_30_0_freqmap/hypoonly_namearg_lhsize/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['qaeval_hyponly_namearg_freqmap', 'nc_qaeval_all']:
        print(f"Testing on: '../datasets/data_qaeval_15_30_0_30_0_10_freqmap/hypoonly_namearg_lhsize/test.txt'")
        model.set_minimum_precision(0.3835664)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_qaeval_15_30_0_30_0_10_freqmap_hyponly_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_qaeval_15_30_0_30_0_10_freqmap_hyponly_test_Y.txt'))
        dataloader = load_custom_data(args, args.model, model,
                                      '../datasets/data_qaeval_15_30_0_30_0_10_freqmap/hypoonly_namearg_lhsize/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['qaeval_hyponly_namearg_radical', 'nc_qaeval_all']:
        print(f"Testing on: '../datasets/data_qaeval_15_30_1000_30_0_0/hypoonly_namearg_lhsize/test.txt'")
        model.set_minimum_precision(0.36505)
        model.set_pr_rec_curve_path(
            os.path.join(ckpt_root, 'en_qaeval_15_30_1000_30_0_0_hyponly_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_qaeval_15_30_1000_30_0_0_hyponly_test_Y.txt'))
        dataloader = load_custom_data(args, args.model, model,
                                      '../datasets/data_qaeval_15_30_1000_30_0_0/hypoonly_namearg_lhsize/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['qaeval_hyponly_typearg', 'nc_qaeval_all']:
        print(f"Testing on: '../datasets/data_qaeval_15_30_0_30_0_0/hypoonly_typearg_lhsize/test.txt'")
        model.set_minimum_precision(0.3365)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_qaeval_15_30_0_30_0_0_hyponly_typearg_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_qaeval_15_30_0_30_0_0_hyponly_typearg_test_Y.txt'))
        dataloader = load_custom_data(args, args.model, model, '../datasets/data_qaeval_15_30_0_30_0_0/hypoonly_typearg_lhsize/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['qaeval_hyponly_nick', 'nc_qaeval_all']:
        print(f"Testing on: '../datasets/data_qaeval_nick/hypoonly_typearg_lhsize/test.txt'")
        model.set_minimum_precision(0.5425)
        model.set_pr_rec_curve_path(
            os.path.join(ckpt_root, 'en_qaeval_nick_hyponly_typearg_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_qaeval_nick_hyponly_typearg_test_Y.txt'))
        dataloader = load_custom_data(args, args.model, model, '../datasets/data_qaeval_nick/hypoonly_typearg_lhsize/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['qaeval_hyponly_typearg_freqmap', 'nc_qaeval_all']:
        print(f"Testing on: '../datasets/data_qaeval_15_30_0_30_0_freqmap/hypoonly_typearg_lhsize/test.txt'")
        model.set_minimum_precision(0.3581)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_qaeval_15_30_0_30_0_freqmap_hyponly_typearg_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_qaeval_15_30_0_30_0_freqmap_hyponly_typearg_test_Y.txt'))
        dataloader = load_custom_data(args, args.model, model, '../datasets/data_qaeval_15_30_0_30_0_freqmap/hypoonly_typearg_lhsize/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['qaeval_hyponly_typearg_freqmap', 'nc_qaeval_all']:
        print(f"Testing on: '../datasets/data_qaeval_15_30_0_30_0_10_freqmap/hypoonly_typearg_lhsize/test.txt'")
        model.set_minimum_precision(0.3898)
        model.set_pr_rec_curve_path(os.path.join(ckpt_root, 'en_qaeval_15_30_0_30_0_10_freqmap_hyponly_typearg_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'en_qaeval_15_30_0_30_0_10_freqmap_hyponly_typearg_test_Y.txt'))
        dataloader = load_custom_data(args, args.model, model, '../datasets/data_qaeval_15_30_0_30_0_10_freqmap/hypoonly_typearg_lhsize/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['clue_qaeval_hyponly_namearg', 'clue_qaeval_all']:
        print(f"Testing on: '../datasets/data_qaeval_clue_15_30_0_30_0_0/hypoonly_namearg_lhsize/test.txt'")
        model.set_minimum_precision(0.3607)
        model.set_pr_rec_curve_path(
            os.path.join(ckpt_root, 'zh_qaeval_15_30_0_30_0_0_hyponly_namearg_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'zh_qaeval_15_30_0_30_0_0_hyponly_namearg_test_Y.txt'))
        dataloader = load_custom_data(args, args.model, model,
                                      '../datasets/data_qaeval_clue_15_30_0_30_0_0/hypoonly_namearg_lhsize/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['clue_qaeval_hyponly_namearg_freqmap', 'clue_qaeval_all']:
        print(f"Testing on: '../datasets/data_qaeval_clue_15_30_0_30_0_0_freqmap/hypoonly_namearg_lhsize/test.txt'")
        model.set_minimum_precision(0.3737)
        model.set_pr_rec_curve_path(
            os.path.join(ckpt_root, 'zh_qaeval_15_30_0_30_0_0_freqmap_hyponly_namearg_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'zh_qaeval_15_30_0_30_0_0_freqmap_hyponly_namearg_test_Y.txt'))
        dataloader = load_custom_data(args, args.model, model,
                                      '../datasets/data_qaeval_clue_15_30_0_30_0_0_freqmap/hypoonly_namearg_lhsize/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['clue_qaeval_hyponly_namearg_freqmap', 'clue_qaeval_all']:
        print(f"Testing on: '../datasets/data_qaeval_clue_15_30_0_30_0_10_freqmap/hypoonly_namearg_lhsize/test.txt'")
        model.set_minimum_precision(0.3902)
        model.set_pr_rec_curve_path(
            os.path.join(ckpt_root, 'zh_qaeval_15_30_0_30_0_10_freqmap_hyponly_namearg_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'zh_qaeval_15_30_0_30_0_10_freqmap_hyponly_namearg_test_Y.txt'))
        dataloader = load_custom_data(args, args.model, model,
                                      '../datasets/data_qaeval_clue_15_30_0_30_0_10_freqmap/hypoonly_namearg_lhsize/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['clue_qaeval_hyponly_typearg', 'clue_qaeval_all']:
        print(f"Testing on: '../datasets/data_qaeval_clue_15_30_0_30_0_0/hypoonly_typearg_lhsize/test.txt'")
        model.set_minimum_precision(0.3605)
        model.set_pr_rec_curve_path(
            os.path.join(ckpt_root, 'zh_qaeval_15_30_0_30_0_0_hyponly_typearg_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'zh_qaeval_15_30_0_30_0_0_hyponly_typearg_test_Y.txt'))
        dataloader = load_custom_data(args, args.model, model,
                                      '../datasets/data_qaeval_clue_15_30_0_30_0_0/hypoonly_typearg_lhsize/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['clue_qaeval_hyponly_typearg_freqmap', 'clue_qaeval_all']:
        print(f"Testing on: '../datasets/data_qaeval_clue_15_30_0_30_0_0_freqmap/hypoonly_typearg_lhsize/test.txt'")
        model.set_minimum_precision(0.3777)
        model.set_pr_rec_curve_path(
            os.path.join(ckpt_root, 'zh_qaeval_15_30_0_30_0_0_freqmap_hyponly_typearg_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'zh_qaeval_15_30_0_30_0_0_freqmap_hyponly_typearg_test_Y.txt'))
        dataloader = load_custom_data(args, args.model, model,
                                      '../datasets/data_qaeval_clue_15_30_0_30_0_0_freqmap/hypoonly_typearg_lhsize/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer

    if args.range in ['clue_qaeval_hyponly_typearg_freqmap', 'clue_qaeval_all']:
        print(f"Testing on: '../datasets/data_qaeval_clue_15_30_0_30_0_10_freqmap/hypoonly_typearg_lhsize/test.txt'")
        model.set_minimum_precision(0.3888)
        model.set_pr_rec_curve_path(
            os.path.join(ckpt_root, 'zh_qaeval_15_30_0_30_0_10_freqmap_hyponly_typearg_test_pr_rec.txt'))
        model.set_score_outfile(os.path.join(ckpt_root, 'zh_qaeval_15_30_0_30_0_10_freqmap_hyponly_typearg_test_Y.txt'))
        dataloader = load_custom_data(args, args.model, model,
                                      '../datasets/data_qaeval_clue_15_30_0_30_0_10_freqmap/hypoonly_typearg_lhsize/test.txt')
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        trainer.test(model, test_dataloaders=dataloader)

        del dataloader
        del trainer
