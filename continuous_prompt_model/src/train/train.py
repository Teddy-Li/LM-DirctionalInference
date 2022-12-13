import argparse
import sys
sys.path.append('src')
from models.multnat_model import MultNatModel
from utils import add_generic_args, generic_train
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_args(parser)
    MultNatModel.add_model_specific_args(parser)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.gpus = None
    print(f"gradient accumulation step: {args.accumulate_grad_batches};")
    print(f"learning rate: {args.learning_rate};")
    print(f"weights decay: {args.weight_decay}")
    print(f"Number of patterns: {args.num_patterns}")
    print(f"Number of tokens per pattern: {args.num_tokens_per_pattern}")

    trainer, model = generic_train(MultNatModel, args)

    print("=== Best VAL Performance ====")
    trainer.test(test_dataloaders=model.val_dataloader())
