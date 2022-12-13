from pytorch_lightning.metrics.functional import auc
import torch


def compute_auc(precisions: torch.FloatTensor, recalls: torch.FloatTensor,
                filter_threshold: float = 0.5) -> torch.FloatTensor:
    xs, ys = [], []
    for p, r in zip(precisions, recalls):
        if p >= filter_threshold:
            xs.append(r)
            ys.append(p)

    return auc(
        torch.cat([x.unsqueeze(0) for x in xs], 0),
        torch.cat([y.unsqueeze(0) for y in ys], 0)
    )


def find_best_F1(precisions, recalls, thresholds):
    def calc_f1(p, r):
        return 2 * (p * r) /(p + r)

    best_f1 = 0.0
    best_thres = None
    best_prec = None
    best_rec = None

    for p, r, t in zip(precisions, recalls, thresholds):
        cur_f1 = calc_f1(p, r)
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_thres = t
            best_prec = p
            best_rec = r
    if best_thres is None:
        print(f"Attention! No f1 score is found to be non-zero!")
    return best_f1, best_prec, best_rec, best_thres