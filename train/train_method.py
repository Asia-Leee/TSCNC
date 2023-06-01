
import time
import torch
from utils.utils import save_checkpoint, AverageMeter, accuracy

def train(train_loader, model, criterion, optimizer, lr_schedule, epoch,device,writer,total_steps,exp_flops,exp_l0,iscloss,args):
    """Train for one epoch on the training set"""
    # global total_steps, exp_flops, exp_l0, args, writer
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # switch to train mode
    model.train()

    lr_schedule.step(epoch=epoch)
    if writer is not None:
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

    end = time.time()
    start_time=time.time()
    for i, (input_, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        total_steps += 1
        if torch.cuda.is_available():
            target = target.cuda(non_blocking=True)
            input_ = input_.cuda()
        input_var = torch.autograd.Variable(input_)
        target_var = torch.autograd.Variable(target)

        # compute output
        if args.amp:
            with torch.cuda.amp.autocast():   #amp  training
                output = model(input_var)
                loss = criterion(output,input_var, target_var, model,device,optimizer,iscloss,args.isadvtraining)
        else:
            output = model(input_var)
            loss = criterion(output, input_var, target_var, model, device, optimizer, iscloss,args.isadvtraining)
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data, input_.size(0))
        top1.update(prec1, input_.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # clamp the parameters
        layers = model.layers if not args.multi_gpu else model.module.layers
        for k, layer in enumerate(layers):
            layer.constrain_parameters()

        e_fl, e_l0 = model.get_exp_flops_l0() if not args.multi_gpu else \
            model.module.get_exp_flops_l0()
        exp_flops.append(e_fl)
        exp_l0.append(e_l0)
        if writer is not None:
            writer.add_scalar('stats_comp/exp_flops', e_fl, total_steps)
            writer.add_scalar('stats_comp/exp_l0', e_l0, total_steps)


        if not args.isprune:
            if not args.multi_gpu:
                if model.beta_ema > 0.:
                    model.update_ema()
            else:
                if model.module.beta_ema > 0.:
                    model.module.update_ema()


        batch_time.update(time.time() - end)
        end = time.time()


        if i % args.print_freq == 0 or i==len(train_loader)-1 :
            end_time=time.time()
            minibatch_time=end_time-start_time
            print(' Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))
            print('minibatch_time',minibatch_time/60,'minute')
            # return top1.avg, closs
    # log to TensorBoard
    if writer is not None:
        writer.add_scalar('train/loss', losses.avg, epoch)
        writer.add_scalar('train/err', top1.avg, epoch)

    return top1.avg