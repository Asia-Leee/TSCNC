import torch.nn as nn
import math
import torch
def freeze_vars(model, var_name, freeze_bn=False):
    """
    freeze vars. If freeze_bn then only freeze batch_norm params.
    """

    assert var_name in ["weight", "bias", "popup_scores","weights"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if not isinstance(v, (nn.BatchNorm2d, nn.BatchNorm2d)) or freeze_bn:
                if getattr(v, var_name) is not None:
                    getattr(v, var_name).requires_grad = False


def unfreeze_vars(model, var_name):
    assert var_name in ["weight", "bias", "popup_scores","weights"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if getattr(v, var_name) is not None:
                getattr(v, var_name).requires_grad = True
def initialize_scaled_score(model):
    print(
        "Initialization relevance score proportional to weight magnitudes (OVERWRITING SOURCE NET SCORES)"
    )
    for m in model.modules():

        if hasattr(m, "popup_scores") and hasattr(m,"weight"):
            n = nn.init._calculate_correct_fan(m.popup_scores, "fan_in")

            m.popup_scores.data = (
                math.sqrt(6 / n) * m.weight.data / torch.max(torch.abs(m.weight.data))
            )
        elif hasattr(m,"popup_scores") and hasattr(m,"weights"):
            n = nn.init._calculate_correct_fan(m.popup_scores, "fan_in")

            m.popup_scores.data = (
                    math.sqrt(6 / n) * m.weights.data / torch.max(torch.abs(m.weights.data))
            )

def prepare_model(model, args):
    """
        1. Set model pruning rate
        2. Set gradients base of training mode.  设置训练模式的梯度基础
    """




    if args.isprune:
        print(f"#################### Pruning network ####################")
        print(f"===>>  gradient for weights: None  | training importance scores only")
        unfreeze_vars(model, "popup_scores")
        freeze_vars(model, "weight", freeze_bn=False)
        freeze_vars(model,"weights",freeze_bn=False)
        freeze_vars(model, "bias", freeze_bn=False)



    elif args.isfinetune:
        print(f"#################### Fine-tuning network ####################")
        print(
            f"===>>  gradient for importance_scores: None  | fine-tuning important weigths only"
        )
        freeze_vars(model, "popup_scores", args.freeze_bn)
        unfreeze_vars(model, "weight")
        unfreeze_vars(model, "bias")

    else:
        assert False, f"{args.exp_mode} mode is not supported"

    initialize_scaled_score(model)

def show_gradients(model):
    for i, v in model.named_parameters():
        print(f"variable = {i}, Gradient requires_grad = {v.requires_grad}")