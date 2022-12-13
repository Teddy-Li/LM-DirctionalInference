from typing import Callable
import argparse
from pathlib import Path
import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers.test_tube import TestTubeLogger
from pytorch_lightning import LightningModule
import sys

sys.path.append('src')
from data.sherliic import SherliicSentences, SherliicPattern
from data.levy_holt import LevyHoltSentences, LevyHoltPattern


def add_generic_args(parser) -> None:
    parser.add_argument(
        "--checkpoint_dir",
        default=None,
        type=str,
        required=True,
        help="The checkpoint directory where the checkpoints will be written.",
    )

    parser.add_argument('--experiment_name', required=True, default='default')

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )

    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O2",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--max_grad_norm", dest="gradient_clip_val",
                        default=1.0, type=float, help="Max gradient norm")
    # parser.add_argument("--do_train", action="store_true",
    #                     help="Whether to run training.")
    # parser.add_argument("--do_predict", action="store_true",
    #                     help="Whether to run predictions on the test set.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        dest="accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument('--seed', type=int, default=47110815)
    parser.add_argument('--gpus', type=int, nargs='+', default=[])

    parser.add_argument('--fixed_lm', action='store_true')
    parser.add_argument('--is_directional', action='store_true')
    parser.add_argument('--lm_prefix', type=str, default='roberta')
    parser.add_argument('--data_name', type=str, default='levy_holt')
    parser.add_argument('--ckpts_per_epoch', type=int, default=1)


def generic_train(
        model_cls: Callable[[argparse.Namespace], LightningModule],
        args: argparse.Namespace
):
    pl.seed_everything(args.seed)

    # init model
    model = model_cls(args)

    cpdir = Path(model.hparams.checkpoint_dir)
    cpdir.mkdir(exist_ok=True)
    cdir = Path(os.path.join(
        model.hparams.checkpoint_dir, args.experiment_name))
    cdir.mkdir(exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=cdir, prefix="checkpoint",
        monitor="AUCHALF", mode="max", save_top_k=1
    )
    logger = TestTubeLogger('tt_logs', name=args.experiment_name)

    train_params = {}

    # TODO: remove with PyTorch 1.6 since pl uses native amp
    if args.fp16:
        train_params["precision"] = 16
        train_params["amp_level"] = args.fp16_opt_level

    if args.gpus is not None and len(args.gpus) > 1:
        train_params["distributed_backend"] = "ddp"

    trainer = pl.Trainer.from_argparse_args(
        args,
        weights_summary=None,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        deterministic=True,
        val_check_interval=(1.0 / args.ckpts_per_epoch),
        **train_params
    )

    trainer.fit(model)

    return trainer, model


str2dataset = {
    "NLI": (SherliicSentences, LevyHoltSentences),
    "MultNat": (SherliicPattern, LevyHoltPattern)
}


def add_dataset_specific_args(parser: argparse.ArgumentParser):
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--levy_holt', action='store_true')
    parser.add_argument('--pattern_file', default=None)
    parser.add_argument('--antipattern_file', default=None)
    parser.add_argument('--best_k_patterns', type=int, default=None)
    parser.add_argument('--curated_auto', action='store_true')


def load_custom_data(args: argparse.Namespace, model_str: str,
                     model: pl.LightningModule, dataset: str = None):
    if args.dataset is not None or dataset is not None:
        if args.dataset is not None and dataset is None:
            dataset = args.dataset
        print(f"dataset: {dataset}")

        data_cls = str2dataset[args.model]
        kwargs = {}
        print(f"args.is_directional: {args.is_directional}")
        kwargs['is_directional'] = args.is_directional

        if args.levy_holt:
            data_cls = data_cls[1]
        else:
            data_cls = data_cls[0]

            if model_str == "MultNat":
                kwargs['pattern_idx'] = -1
                kwargs['antipattern_idx'] = -1
            if args.pattern_file is None:
                kwargs['with_examples'] = True
            if args.curated_auto:
                kwargs['with_examples'] = True
                kwargs['curated_auto'] = True

        if args.pattern_file is not None:
            kwargs['pattern_file'] = args.pattern_file
            kwargs['antipattern_file'] = args.antipattern_file
            kwargs['best_k_patterns'] = args.best_k_patterns

        data = data_cls(dataset, **kwargs)
        dataloader = DataLoader(
            data,
            batch_size=model.hparams.eval_batch_size,
            num_workers=model.hparams.num_workers,
            collate_fn=model.collate,
            pin_memory=True
        )
    else:
        dataloader = None

    return dataloader
