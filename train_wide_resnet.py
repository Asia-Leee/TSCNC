import torch.backends.cudnn as cudnn
from models.L0WideResNet.L0WideResNet import L0WideResNet
from dataloader.dataloaders import *
from torch.optim import lr_scheduler

from eval.test_method import *
from train.train_method import train
from loss.trade_loss import trades_loss
from loss.cross_entropy_loss import Cross_Entropy_Loss
from configs.args import  parse_args
from eval import test_method
from utils.utils import *

best_acc1_benign = 0
best_acc1_adv=0
writer = None
time_acc = [(0, 0, 0)]
total_steps = 0
exp_flops, exp_l0 = [], []


def main():
    global args, best_acc1_benign, best_acc1_adv,writer, time_acc, total_steps, exp_flops, exp_l0
    args = parse_args()

    args.configs = "configs/l0_WRN_28_10.yml"

    parse_configs_file(args)
    log_dir_net = args.save_path
    args.save_path += '_{}_{}'.format(args.depth, args.width)
    print('model:', args.save_path)

    if args.tensorboard:
        # used for logging to TensorBoard
        from tensorboardX import SummaryWriter
        directory = 'logs/{}/{}'.format(log_dir_net, args.save_path)
        if os.path.exists(directory):
            shutil.rmtree(directory)
            os.makedirs(directory)
        else:
            os.makedirs(directory)
        writer = SummaryWriter(directory)

    # Data loading code
    if args.dataset=='c10':
        train_loader, val_loader, num_classes = cifar10(args.data_path,augment=args.augment, batch_size=args.batch_size)
    elif args.dataset=='c100':
        train_loader, val_loader, num_classes = cifar100(args.data_path, augment=args.augment,
                                                        batch_size=args.batch_size)
    elif args.dataset=='imagenet_1k':
        train_loader,val_loader,num_classes=imagnenet_1k_torch(data_path=args.data_path,batch_size=args.batch_size,seed=args.seed)
    elif args.dataset=='tinyimagenet_base':
        train_loader,val_loader,num_classes=tiny_imagenet_base(data_path=args.data_path,batch_size=args.batch_size,seed=args.seed)
    elif args.dataset=='tinyimagenet_adv':
        train_loader,val_loader,num_classes=tiny_imagenet_adv(data_path=args.data_path,batch_size=args.batch_size,seed=args.seed)
    # create model
    model = L0WideResNet(args.depth, num_classes, widen_factor=args.width, droprate_init=args.droprate_init,
                         N=args.N, beta_ema=args.beta_ema, weight_decay=args.weight_decay, local_rep=args.local_rep,
                         lamba=args.lamba, temperature=args.temp,iscloss=args.iscloss,a=args.a,lambdda=args.lambdda,args=args)

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    if args.multi_gpu:
        model = torch.nn.DataParallel(model).cuda()
    else:
        if torch.cuda.is_available():
            model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, nesterov=True)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            # best_prec1_adv=checkpoint['best_prec1_adv']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            total_steps = checkpoint['total_steps']
            time_acc = checkpoint['time_acc']
            exp_flops = checkpoint['exp_flops']
            exp_l0 = checkpoint['exp_l0']
            if checkpoint['beta_ema'] > 0:
                if not args.multi_gpu:
                    model.beta_ema = checkpoint['beta_ema']
                    model.avg_param = checkpoint['avg_params']
                    model.steps_ema = checkpoint['steps_ema']
                else:
                    model.module.beta_ema = checkpoint['beta_ema']
                    model.module.avg_param = checkpoint['avg_params']
                    model.module.steps_ema = checkpoint['steps_ema']

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            total_steps, exp_flops, exp_l0 = 0, [], []

    #orune a sparse-trained model
    if args.isprune:
        if os.path.isfile(args.checkpoint_path):
            print("=> loading checkpoint '{}'".format(args.checkpoint_path))
            checkpoint = torch.load(args.checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if checkpoint['beta_ema'] > 0:
                if not args.multi_gpu:
                    model.beta_ema = checkpoint['beta_ema']
                    model.avg_param = checkpoint['avg_params']
                    model.steps_ema = checkpoint['steps_ema']
                else:
                    model.module.beta_ema = checkpoint['beta_ema']
                    model.module.avg_param = checkpoint['avg_params']
                    model.module.steps_ema = checkpoint['steps_ema']

            if args.multi_gpu:
                model.module.load_ema_params()
            else:
                model.load_ema_params()

            prepare_model(model, args)
            show_gradients(model)
    cudnn.benchmark = True


    device = torch.device('cuda:0')
    def loss_function(output, input_var,target_var, model,device,optimizer,iscloss,isadvtraining):

        if isadvtraining:
            loss=trades_loss(output,model,x_natural=input_var,y=target_var,device=device,optimizer=optimizer,
                         step_size=0.0069,epsilon=8.0/255,perturb_steps=10,beta=6.0,clip_min=0,clip_max=1.0,)
        else:
            loss = Cross_Entropy_Loss(output, target_var)

        if iscloss:
            reg ,closs= model.regularization() if not args.multi_gpu else model.module.regularization()
            total_loss = loss + reg+ closs
            if torch.cuda.is_available():
                total_loss = total_loss.cuda()
            return total_loss

        else: #nocloss
            reg,closs=model.regularization() if not args.multi_gpu else model.module.regularization()
            total_loss = loss + reg
            if torch.cuda.is_available():
                total_loss = total_loss.cuda()
            return total_loss

    lr_schedule = lr_scheduler.MultiStepLR(optimizer, milestones=args.epoch_drop, gamma=args.lr_decay_ratio)
    start_time=time.time()

    #start training
    for epoch in range(args.start_epoch, args.epochs):
        time_glob = time.time()
        # train for one epoch
        if args.multi_gpu and (args.dataset=='imagenet_1k' or args.dataset=='tinyimagenet'):
            train_loader.sampler.set_epoch(epoch)
        acc1_train = train(train_loader, model, loss_function, optimizer, lr_schedule,
                         epoch,device,writer,total_steps,exp_flops,exp_l0,args.iscloss,args)
        # test for one epoch
        validate_method=getattr(test_method,args.test_method)
        acc1_benign,acc1_adv=validate_method(val_loader, model, loss_function,optimizer, epoch,device,writer,args.iscloss,args)

        time_ep = time.time() - time_glob
        time_acc.append((time_ep + time_acc[-1][0], acc1_train, acc1_benign))

        # remember best prec@1 and save checkpoint
        is_best_benign = acc1_benign > best_acc1_benign
        is_best_adv=acc1_adv>best_acc1_adv
        best_acc1_benign = max(acc1_benign, best_acc1_benign)
        best_acc1_adv=max(acc1_adv,best_acc1_adv)


        state = {
            'epoch': epoch ,
            'state_dict': model.state_dict(),
            'acc1_benign':acc1_benign,
            'acc1_adv': acc1_adv,
            'optimizer': optimizer.state_dict(),
            'total_steps': total_steps,
            'time_acc': time_acc,
            'exp_flops': exp_flops,
            'exp_l0': exp_l0
        }
        if not args.multi_gpu:
            state['beta_ema'] = model.beta_ema
            if model.beta_ema > 0:
                state['avg_params'] = model.avg_param
                state['steps_ema'] = model.steps_ema
        else:
            state['beta_ema'] = model.module.beta_ema
            if model.module.beta_ema > 0:
                state['avg_params'] = model.module.avg_param
                state['steps_ema'] = model.module.steps_ema

        save_checkpoint(state,checkpoint_path=args.save_path,filename='checkpoint.pth.tar',isprune=args.isprune,sparsity=args.sparsity,)

        if is_best_adv and args.isadvtraining:
            save_checkpoint(state,checkpoint_path=args.save_path,filename='checkpoint_best_adv.pth.tar',isprune=args.isprune,sparsity=args.sparsity,save_dense=True)

        elif is_best_benign and not args.isadvtraining:
            save_checkpoint(state, checkpoint_path=args.save_path, filename='checkpoint_best_benign.pth.tar',isprune=args.isprune)

    end_time=time.time()
    print('Total time:',(end_time-start_time)/3600,'h')
    if args.tensorboard:
        writer.close()

if __name__ == '__main__':
    main()
