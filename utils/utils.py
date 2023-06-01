import numpy as np
import shutil
import math
import os
import yaml
import sys
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
import torch.nn.functional as F
from torch.nn.parallel._functions import Broadcast
from torch.nn.parallel import scatter, parallel_apply, gather
from functools import partial
from nested_dict import nested_dict
from models.L0WideResNet.l0_layers import L0Conv2d
from models.L0WideResNet.base_layers import MAPConv2d, MAPDense
prng = np.random.RandomState(1)
torch.manual_seed(1)

def count_zero_number(model):
    number_zero = 0
    number_model = 0
    for m in model.modules():
        if isinstance(m, MAPDense) or isinstance(m, MAPConv2d):
            zero_matrix = torch.zeros_like(m.weight)
            number_zero += torch.sum(torch.where(torch.eq(m.weight, 0), 1, 0))
            number_model += m.weight.numel()
        elif isinstance(m, L0Conv2d):
            zero_matrix = torch.zeros_like(m.weights)
            number_zero += torch.sum(torch.where(torch.eq(m.weights, 0), 1, 0))
            number_model += m.weights.numel()
    sparsity = number_zero / number_model
    return sparsity



def change_random_seed(seed):
    global prng
    prng = np.random.RandomState(seed)
    torch.manual_seed(seed)


def to_one_hot(x, n_cats=10):
    y = np.zeros((x.shape[0], n_cats))
    y[np.arange(x.shape[0]), x] = 1
    return y.astype(np.float32)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def get_flat_fts(in_size, fts):
    dummy_input = torch.ones(1, *in_size)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
    f = fts(torch.autograd.Variable(dummy_input))
    print('conv_out_size: {}'.format(f.size()))
    return int(np.prod(f.size()[1:]))


def adjust_learning_rate(optimizer, epoch, lr=0.1, lr_decay_ratio=0.1, epoch_drop=(), writer=None):
    """Simple learning rate drop according to the provided parameters"""
    optim_factor = 0
    for i, ep in enumerate(epoch_drop):
        if epoch > ep:
            optim_factor = i + 1
    lr = lr * lr_decay_ratio ** optim_factor

    # log to TensorBoard
    if writer is not None:
        writer.add_scalar('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def subnet_to_dense(subnet_dict, p):
    """
        Convert a subnet state dict (with subnet layers) to dense i.e., which can be directly
        loaded in network with dense layers.
    """
    dense = {}

    # load dense variables
    for (k, v) in subnet_dict.items():
        if "popup_scores" not in k:
            dense[k] = v

    # update dense variables
    for (k, v) in subnet_dict.items():
        if "popup_scores" in k:
            s = torch.abs(subnet_dict[k])

            out = s.clone()
            _, idx = s.flatten().sort()
            j = int((1 - p) * s.numel())

            flat_out = out.flatten()
            flat_out[idx[:j]] = 0
            flat_out[idx[j:]] = 1
            if k.replace("popup_scores", "weight") in dense.keys():
                dense[k.replace("popup_scores", "weight")] = (
                    subnet_dict[k.replace("popup_scores", "weight")] * out
                )
            else:
                dense[k.replace("popup_scores", "weights")] = (
                        subnet_dict[k.replace("popup_scores", "weights")] * out
                )
    return dense
def save_checkpoint(state, checkpoint_path, filename='checkpoint.pth.tar',isprune=False,sparsity=1.,save_dense=False):
    """Saves checkpoint to disk"""
    if not isprune:
        directory = "runs/%s/" % checkpoint_path
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + filename
        torch.save(state, filename)
    else:
        directory="runs/prune_checkpoint/%s/" % checkpoint_path
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename=directory+filename
        torch.save(state, filename)

        if save_dense:
            state["state_dict"] = subnet_to_dense(state["state_dict"], sparsity)
            torch.save(
                subnet_to_dense(state, sparsity),  os.path.join(directory, "checkpoint_best_adv_dense.pth.tar"))


def net_with_popupscores_to_dense(net_with_popup_scores_dict, p):
    """
        Convert a subnet state dict (with subnet layers) to dense i.e., which can be directly
        loaded in network with dense layers.
    """
    dense = {}

    # load dense variables
    for (k, v) in net_with_popup_scores_dict.items():
        if "popup_scores" not in k:
            dense[k] = v

    # update dense variables
    for (k, v) in net_with_popup_scores_dict.items():
        if "popup_scores" in k:
            s = torch.abs(net_with_popup_scores_dict[k])

            out = s.clone()
            _, idx = s.flatten().sort()
            j = int((1 - p) * s.numel())

            flat_out = out.flatten()
            flat_out[idx[:j]] = 0
            flat_out[idx[j:]] = 1
            if k.replace("popup_scores", "weight") in dense :
                dense[k.replace("popup_scores", "weight")] = (
                    net_with_popup_scores_dict[k.replace("popup_scores", "weight")] * out
                )
            elif k.replace("popup_scores", "weights") in dense:
                dense[k.replace("popup_scores", "weights")] = (
                        net_with_popup_scores_dict[k.replace("popup_scores", "weights")] * out
                )
    return dense
def save_prune_checkpoint(state, is_best, name,save_dense, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/prune_checkpoint/%s/" % name
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/prune_checkpoint/%s/' % name + 'model_best.pth.tar')

    if save_dense:
        state["state_dict"] = net_with_popupscores_to_dense(state["state_dict"], 0.1)
        filename_dense=directory+'checkpoint_dense.pth.tar'
        torch.save(
            net_with_popupscores_to_dense(state, 0.1),
            filename_dense)
        if is_best:
            shutil.copyfile(filename_dense,'runs/prune_checkpoint/%s/' % name + 'model_best_dense.pth.tar')


def cast(params, dtype='float'):
    if isinstance(params, dict):
        return {k: cast(v, dtype) for k,v in params.items()}
    else:
        return getattr(params.cuda() if torch.cuda.is_available() else params, dtype)()


def conv_params(ni, no, k=1):
    return kaiming_normal_(torch.Tensor(no, ni, k, k))


def linear_params(ni, no):
    return {'weight': kaiming_normal_(torch.Tensor(no, ni)), 'bias': torch.zeros(no)}


def bnparams(n):
    return {'weight': torch.rand(n),
            'bias': torch.zeros(n),
            'running_mean': torch.zeros(n),
            'running_var': torch.ones(n)}


def data_parallel(f, input, params, mode, device_ids, output_device=None):
    assert isinstance(device_ids, list)
    if output_device is None:
        output_device = device_ids[0]

    if len(device_ids) == 1:
        return f(input, params, mode)

    params_all = Broadcast.apply(device_ids, *params.values())
    params_replicas = [{k: params_all[i + j*len(params)] for i, k in enumerate(params.keys())}
                       for j in range(len(device_ids))]

    replicas = [partial(f, params=p, mode=mode)
                for p in params_replicas]
    inputs = scatter([input], device_ids)
    outputs = parallel_apply(replicas, inputs)
    return gather(outputs, output_device)


def flatten(params):
    return {'.'.join(k): v for k, v in nested_dict(params).items_flat() if v is not None}


def batch_norm(x, params, base, mode):
    return F.batch_norm(x, weight=params[base + '.weight'],
                        bias=params[base + '.bias'],
                        running_mean=params[base + '.running_mean'],
                        running_var=params[base + '.running_var'],
                        training=mode)


def print_tensor_dict(params):
    kmax = max(len(key) for key in params.keys())
    for i, (key, v) in enumerate(params.items()):
        print(str(i).ljust(5), key.ljust(kmax + 3), str(tuple(v.shape)).ljust(23), torch.typename(v), v.requires_grad)


def set_requires_grad_except_bn_(params):
    for k, v in params.items():
       if not k.endswith('running_mean') and not k.endswith('running_var'):
            v.requires_grad = True



def freeze_vars(model, var_name, freeze_bn=False):
    """
    freeze vars. If freeze_bn then only freeze batch_norm params.
    """

    assert var_name in ["weight", "bias", "popup_scores","weights","qz_loga"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if not isinstance(v, (nn.BatchNorm2d, nn.BatchNorm2d)) or freeze_bn:
                if getattr(v, var_name) is not None:
                    getattr(v, var_name).requires_grad = False


def unfreeze_vars(model, var_name):
    assert var_name in ["weight", "bias", "popup_scores"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if getattr(v, var_name) is not None:
                getattr(v, var_name).requires_grad = True
def show_gradients(model):
    for i, v in model.named_parameters():
        print(f"variable = {i}, Gradient requires_grad = {v.requires_grad}")
def initialize_scaled_score(model):
    print(
        "Initialization relevance score proportional to weight magnitudes (OVERWRITING SOURCE NET SCORES)"
    )
    for m in model.modules():
        if hasattr(m, "popup_scores"):
            n = torch.nn.init._calculate_correct_fan(m.popup_scores, "fan_in")
            # Close to kaiming unifrom init
            m.popup_scores.data = (
                math.sqrt(6 / n) * m.weight.data / torch.max(torch.abs(m.weight.data))
            ) if hasattr(m,'weight') else (
                math.sqrt(6 / n) * m.weights.data / torch.max(torch.abs(m.weights.data))
            )

def prepare_model(model, args):
    """
        1. Set model pruning rate
        2. Set gradients base of training mode.
    """




    # elif args.exp_mode == "prune":
    unfreeze_vars(model, "popup_scores")
    freeze_vars(model, "weight", args.freeze_bn)
    freeze_vars(model, "weights", args.freeze_bn)
    freeze_vars(model, "qz_loga", args.freeze_bn)
    freeze_vars(model, "bias", args.freeze_bn)
    initialize_scaled_score(model)

def parse_configs_file(args):
    # get commands from command line
    override_args = argv_to_vars(sys.argv)

    # load yaml file
    yaml_txt = open(args.configs).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML config from {args.configs}")
    args.__dict__.update(loaded_yaml)
def argv_to_vars(argv):
    var_names = []
    for arg in argv:
        if arg.startswith("-") and arg_to_varname(arg) != "config":
            var_names.append(arg_to_varname(arg))

    return var_names
def arg_to_varname(st: str):
    st = trim_preceding_hyphens(st)
    st = st.replace("-", "_")

    return st.split("=")[0]
def trim_preceding_hyphens(st):
    i = 0
    while st[i] == "-":
        i += 1

    return st[i:]

