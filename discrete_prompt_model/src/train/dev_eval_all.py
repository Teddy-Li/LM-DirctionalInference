import argparse
import pytorch_lightning as pl
import sys
import os
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
    parser.add_argument('checkpoint_root')
    parser.add_argument('--classification_threshold', type=float, default=None)
    parser.add_argument('--gpus', type=int, nargs='+', default=[])
    add_dataset_specific_args(parser)
    args = parser.parse_args()

    cls = str2model[args.model]

    fns = os.listdir(args.checkpoint_root)
    fns.sort()
    best_F1 = 0.0
    best_fn = None

    for fn in fns:
        full_fn = os.path.join(args.checkpoint_root, fn)
        print(fn)
        if os.path.isdir(full_fn):
            ckpt_fns = os.listdir(full_fn)
            ckpt_fns = [os.path.join(full_fn, x) for x in ckpt_fns]
            ckpt_fns.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            ckpt_fn = None
            for f in ckpt_fns:
                if f.endswith('.ckpt'):
                    ckpt_fn = f
                    break
            print(ckpt_fn)
            if ckpt_fn is None:
                print(f"No valid checkpoint found for: {full_fn}")
                continue

            model = cls.load_from_checkpoint(ckpt_fn)
            if args.classification_threshold is not None:
                model.set_classification_threshold(args.classification_threshold)

            dataloader = load_custom_data(args, args.model, model)

            trainer = pl.Trainer(gpus=args.gpus, logger=False)
            result = trainer.test(model, test_dataloaders=dataloader)
            print(result)
            if result['F1'] > best_F1:
                best_F1 = result['F1']
                best_fn = ckpt_fn

    print(f"Best dev set F1: {best_F1}; best fn: {best_fn}.")
