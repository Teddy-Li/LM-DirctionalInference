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
from data.sherliic import Sherliic
from data.levy_holt import LevyHolt


def add_generic_args(parser) -> None:
    parser.add_argument(
        "--checkpoint_dir",
        default=None,
        type=str,
        required=True,
        help="The checkpoint directory where the checkpoints will be written.",
    )

    parser.add_argument('--experiment_name', required=True, default='default')

    parser.add_argument("--max_grad_norm", dest="gradient_clip_val",
                        default=1.0, type=float, help="Max gradient norm")

    parser.add_argument(
        "--gradient_accumulation_steps",
        dest="accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument('--seed', type=int, default=47110815)
    parser.add_argument('--gpus', type=int, default=1)

    # parser.add_argument('--lm_prefix', type=str, default='roberta')
    parser.add_argument('--data_name', type=str, default='levy_holt')
    parser.add_argument('--clear_cache_cycle', type=int, default=20)
    parser.add_argument('--ckpts_per_epoch', type=int, default=2)


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
        dirpath=cdir, filename='{epoch}-{val_loss:.2f}-{AUC:.2f}',
        monitor="AUCHALF", mode="max", save_top_k=1
    )
    logger = TestTubeLogger('tt_logs', name=args.experiment_name)

    trainer = pl.Trainer.from_argparse_args(
        args,
        weights_summary=None,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        deterministic=True,
        val_check_interval=(1.0/args.ckpts_per_epoch),
        distributed_backend="ddp" if (args.gpus is not None and args.gpus > 1) else None
    )

    trainer.fit(model)

    return trainer, model


def add_dataset_specific_args(parser: argparse.ArgumentParser):
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--levy_holt', action='store_true')
    parser.add_argument("--use_antipatterns", action='store_true')
    parser.add_argument("--num_patterns", default=1, type=int)
    parser.add_argument("--num_tokens_per_pattern", default=1, type=int)
    parser.add_argument("--only_sep", action='store_true')
    parser.add_argument("--pattern_chunk_size", default=5, type=int)


def load_custom_data(args: argparse.Namespace, model: pl.LightningModule, dataset: str = None) -> DataLoader:
    if args.dataset is not None or dataset is not None:
        if args.dataset is not None and dataset is None:
            dataset = args.dataset
        print(f"dataset: {dataset}")

        if args.levy_holt:
            data_cls = LevyHolt
        else:
            data_cls = Sherliic

        kwargs = dict(
            use_antipatterns=args.use_antipatterns,
            num_patterns=args.num_patterns,
            num_tokens_per_pattern=args.num_tokens_per_pattern,
            only_sep=args.only_sep,
            pattern_chunk_size=args.pattern_chunk_size
        )

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
