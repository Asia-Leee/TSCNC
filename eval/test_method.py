import time
import torch
from utils.utils import  AverageMeter, accuracy
from eval.attack_method import pgd_whitebox
from eval.attack_method import attack_loader
#pgd-10 test
def PGD_10(val_loader, model, criterion, optimizer, epoch, device, writer,closs,args):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top1_adv = AverageMeter()

    # switch to evaluate mode
    model.eval()
    # model_yuanshi=copy.deepcopy(model)
    if not args.isprune:
        if not args.multi_gpu:
            if model.beta_ema > 0: #设置为0就不会更改test的参数
                old_params = model.get_params()
                model.load_ema_params()
        else:
            if model.module.beta_ema > 0:
                old_params = model.module.get_params()
                model.module.load_ema_params()

    end = time.time()
    for i, (input_, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            target = target.cuda(non_blocking=True)
            input_ = input_.cuda()
        input_var = torch.autograd.Variable(input_, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        output = model(input_var)


        loss = criterion(output, input_var,target_var, model,device,optimizer,closs)
        model.eval()
        losses.update(loss.data, input_.size(0))

        prec1 = accuracy(output.data, target, topk=(1,))[0]

        top1.update(prec1, input_.size(0))


        images_adv = pgd_whitebox(
            model=model,
            x=input_var,
            y=target_var,
            device=device,
            epsilon=8.0 / 255,
            num_steps=30,
            step_size=0.0023,
            clip_min=0,
            clip_max=1.0,
            is_random=True,
        )
        output_adv = model(images_adv)
        acc_adv = accuracy(output_adv, target, )
        top1_adv.update(acc_adv[0], input_.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'acc_benign@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'acc_adv {top1_adv.val:.3f} ({top1_adv.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top1_adv=top1_adv))
            # return top1.avg, top1_adv.avg

    print(' * Err@1 {top1.avg:.3f}\t'
          '* top_adv_acc {top1_adv.avg:.3f}'.format(top1=top1, top1_adv=top1_adv))

    if not args.isprune:
        if not args.multi_gpu:
            if model.beta_ema > 0:
                model.load_params(old_params)
        else:
            if model.module.beta_ema > 0:
                model.module.load_params(old_params)

    # log to TensorBoard
    if writer is not None:
        writer.add_scalar('val/loss', losses.avg, epoch)
        writer.add_scalar('val/err', top1.avg, epoch)
        layers = model.layers if not args.multi_gpu else model.module.layers
        for k, layer in enumerate(layers):
            if hasattr(layer, 'qz_loga'):
                mode_z = layer.sample_z(1, sample=0).view(-1)
                writer.add_histogram('mode_z/layer{}'.format(k), mode_z.cpu().data.numpy(), epoch)
    return top1.avg, top1_adv.avg

#pgd-20-test
def PGD_20(val_loader, model, criterion, optimizer, epoch, device, writer,closs,args):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top1_adv = AverageMeter()

    # switch to evaluate mode
    model.eval()
    # model_yuanshi=copy.deepcopy(model)
    if not args.isprune:
        if not args.multi_gpu:
            if model.beta_ema > 0:
                old_params = model.get_params()
                model.load_ema_params()
        else:
            if model.module.beta_ema > 0:
                old_params = model.module.get_params()
                model.module.load_ema_params()

    end = time.time()
    for i, (input_, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            target = target.cuda(non_blocking=True)
            input_ = input_.cuda()
        input_var = torch.autograd.Variable(input_, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        output = model(input_var)


        # loss = criterion(output, input_var,target_var, model,device,optimizer,closs)
        # losses.update(loss.data, input_.size(0))

        prec1 = accuracy(output.data, target, topk=(1,))[0]

        top1.update(prec1, input_.size(0))


        images_adv = pgd_whitebox(
            model=model,
            x=input_var,
            y=target_var,
            device=device,
            epsilon=8.0 / 255,
            num_steps=20,
            step_size=2.0 / 255,
            clip_min=0,
            clip_max=1.0,
            is_random=True,
        )
        output_adv = model(images_adv)
        acc_adv = accuracy(output_adv, target, )
        top1_adv.update(acc_adv[0], input_.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Err@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'acc_adv {top1_adv.val:.3f} ({top1_adv.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top1_adv=top1_adv))
            # return top1.avg, top1_adv.avg

    print(' * Err@1 {top1.avg:.3f}\t'
          '* top_adv_acc {top1_adv.avg:.3f}'.format(top1=top1, top1_adv=top1_adv))

    if not args.isprune:
        if not args.multi_gpu:
            if model.beta_ema > 0:
                model.load_params(old_params)
        else:
            if model.module.beta_ema > 0:
                model.module.load_params(old_params)

    # log to TensorBoard
    if writer is not None:
        writer.add_scalar('val/loss', losses.avg, epoch)
        writer.add_scalar('val/err', top1.avg, epoch)
        layers = model.layers if not args.multi_gpu else model.module.layers
        for k, layer in enumerate(layers):
            if hasattr(layer, 'qz_loga'):
                mode_z = layer.sample_z(1, sample=0).view(-1)
                writer.add_histogram('mode_z/layer{}'.format(k), mode_z.cpu().data.numpy(), epoch)
    return top1.avg, top1_adv.avg
def PGD_50(val_loader, model, criterion, optimizer, epoch, device, writer,closs,args):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top1_adv = AverageMeter()

    # switch to evaluate mode
    model.eval()
    # model_yuanshi=copy.deepcopy(model)
    if not args.isprune:
        if not args.multi_gpu:
            if model.beta_ema > 0:
                old_params = model.get_params()
                model.load_ema_params()
        else:
            if model.module.beta_ema > 0:
                old_params = model.module.get_params()
                model.module.load_ema_params()

    end = time.time()
    for i, (input_, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            target = target.cuda(non_blocking=True)
            input_ = input_.cuda()
        input_var = torch.autograd.Variable(input_, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        output = model(input_var)


        # loss = criterion(output, input_var,target_var, model,device,optimizer,closs)
        # losses.update(loss.data, input_.size(0))

        prec1 = accuracy(output.data, target, topk=(1,))[0]

        top1.update(prec1, input_.size(0))


        images_adv = pgd_whitebox(
            model=model,
            x=input_var,
            y=target_var,
            device=device,
            epsilon=8.0 / 255,
            num_steps=50,
            step_size=2.0 / 255,
            clip_min=0,
            clip_max=1.0,
            is_random=True,
        )
        output_adv = model(images_adv)
        acc_adv = accuracy(output_adv, target, )
        top1_adv.update(acc_adv[0], input_.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Err@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'acc_adv {top1_adv.val:.3f} ({top1_adv.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top1_adv=top1_adv))
            # return top1.avg, top1_adv.avg

    print(' * Err@1 {top1.avg:.3f}\t'
          '* top_adv_acc {top1_adv.avg:.3f}'.format(top1=top1, top1_adv=top1_adv))

    if not args.isprune:
        if not args.multi_gpu:
            if model.beta_ema > 0:
                model.load_params(old_params)
        else:
            if model.module.beta_ema > 0:
                model.module.load_params(old_params)

    # log to TensorBoard
    if writer is not None:
        writer.add_scalar('val/loss', losses.avg, epoch)
        writer.add_scalar('val/err', top1.avg, epoch)
        layers = model.layers if not args.multi_gpu else model.module.layers
        for k, layer in enumerate(layers):
            if hasattr(layer, 'qz_loga'):
                mode_z = layer.sample_z(1, sample=0).view(-1)
                writer.add_histogram('mode_z/layer{}'.format(k), mode_z.cpu().data.numpy(), epoch)
    return top1.avg, top1_adv.avg

#base-test
def base(val_loader, model, criterion, optimizer, epoch, device, writer,closs,args):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    if not args.multi_gpu:
        if model.beta_ema > 0:
            old_params = model.get_params()
            model.load_ema_params()
    else:
        if model.module.beta_ema > 0:
            old_params = model.module.get_params()
            model.module.load_ema_params()

    end = time.time()
    for i, (input_, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            target = target.cuda(non_blocking=True)
            input_ = input_.cuda()
        input_var = torch.autograd.Variable(input_, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        output = model(input_var)

        prec1 = accuracy(output.data, target_var, topk=(1,))[0]
        top1.update(prec1, input_.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i==len(val_loader)-1:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Err@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))
            # return top1.avg,torch.tensor(0)  #测试用

    print(' * Err@1 {top1.avg:.3f}'.format(top1=top1))

    if not args.multi_gpu:
        if model.beta_ema > 0:
            model.load_params(old_params)
    else:
        if model.module.beta_ema > 0:
            model.module.load_params(old_params)

    # log to TensorBoard
    if writer is not None:
        writer.add_scalar('val/loss', losses.avg, epoch)
        writer.add_scalar('val/err', top1.avg, epoch)
        layers = model.layers if not args.multi_gpu else model.module.layers
        for k, layer in enumerate(layers):
            if hasattr(layer, 'qz_loga'):
                mode_z = layer.sample_z(1, sample=0).view(-1)
                writer.add_histogram('mode_z/layer{}'.format(k), mode_z.cpu().data.numpy(), epoch)

    return top1.avg,torch.tensor(0)

def APGD(val_loader, model, criterion, optimizer, epoch, device, writer,closs,args):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top1_adv = AverageMeter()

    # switch to evaluate mode
    model.eval()
    # model_yuanshi=copy.deepcopy(model)
    if not args.isprune:
        if not args.multi_gpu:
            if model.beta_ema > 0:
                old_params = model.get_params()
                model.load_ema_params()
        else:
            if model.module.beta_ema > 0:
                old_params = model.module.get_params()
                model.module.load_ema_params()

    end = time.time()
    for i, (input_, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            target = target.cuda(non_blocking=True)
            input_ = input_.cuda()
        input_var = torch.autograd.Variable(input_, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        output = model(input_var)


        # loss = criterion(output, input_var,target_var, model,device,optimizer,closs)
        # losses.update(loss.data, input_.size(0))
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        top1.update(prec1, input_.size(0))



        attack=attack_loader(model=model,attack_method='APGD',epsilon=8/255,num_steps=10,step_size=2/255,num_class=100)

        images_adv = attack(input_var,target_var)
        output_adv = model(images_adv)
        acc_adv = accuracy(output_adv, target, )
        top1_adv.update(acc_adv[0], input_.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Err@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'acc_adv {top1_adv.val:.3f} ({top1_adv.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top1_adv=top1_adv))
            # return top1.avg, top1_adv.avg

    print(' * Err@1 {top1.avg:.3f}\t'
          '* top_adv_acc {top1_adv.avg:.3f}'.format(top1=top1, top1_adv=top1_adv))

    if not args.isprune:
        if not args.multi_gpu:
            if model.beta_ema > 0:
                model.load_params(old_params)
        else:
            if model.module.beta_ema > 0:
                model.module.load_params(old_params)

    # log to TensorBoard
    if writer is not None:
        writer.add_scalar('val/loss', losses.avg, epoch)
        writer.add_scalar('val/err', top1.avg, epoch)
        layers = model.layers if not args.multi_gpu else model.module.layers
        for k, layer in enumerate(layers):
            if hasattr(layer, 'qz_loga'):
                mode_z = layer.sample_z(1, sample=0).view(-1)
                writer.add_histogram('mode_z/layer{}'.format(k), mode_z.cpu().data.numpy(), epoch)
    return top1.avg, top1_adv.avg

def AA(val_loader, model, criterion, optimizer, epoch, device, writer,closs,args):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top1_adv = AverageMeter()

    # switch to evaluate mode
    model.eval()
    # model_yuanshi=copy.deepcopy(model)
    if not args.isprune:
        if not args.multi_gpu:
            if model.beta_ema > 0:
                old_params = model.get_params()
                model.load_ema_params()
        else:
            if model.module.beta_ema > 0:
                old_params = model.module.get_params()
                model.module.load_ema_params()

    end = time.time()
    for i, (input_, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            target = target.cuda(non_blocking=True)
            input_ = input_.cuda()
        input_var = torch.autograd.Variable(input_, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        output = model(input_var)


        # loss = criterion(output, input_var,target_var, model,device,optimizer,closs)
        # losses.update(loss.data, input_.size(0))
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        top1.update(prec1, input_.size(0))



        attack=attack_loader(model=model,attack_method='AA',epsilon=8/255,num_steps=10,step_size=2/255,num_class=100)
        images_adv = attack(input_var,target_var)
        output_adv = model(images_adv)
        acc_adv = accuracy(output_adv, target, )
        top1_adv.update(acc_adv[0], input_.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Err@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'acc_adv {top1_adv.val:.3f} ({top1_adv.avg:.3f})'.format(
            i, len(val_loader), batch_time=batch_time, loss=losses,
            top1=top1, top1_adv=top1_adv))
        # return top1.avg, top1_adv.avg

    print(' * Err@1 {top1.avg:.3f}\t'
          '* top_adv_acc {top1_adv.avg:.3f}'.format(top1=top1, top1_adv=top1_adv))

    if not args.isprune:
        if not args.multi_gpu:
            if model.beta_ema > 0:
                model.load_params(old_params)
        else:
            if model.module.beta_ema > 0:
                model.module.load_params(old_params)

    # log to TensorBoard
    if writer is not None:
        writer.add_scalar('val/loss', losses.avg, epoch)
        writer.add_scalar('val/err', top1.avg, epoch)
        layers = model.layers if not args.multi_gpu else model.module.layers
        for k, layer in enumerate(layers):
            if hasattr(layer, 'qz_loga'):
                mode_z = layer.sample_z(1, sample=0).view(-1)
                writer.add_histogram('mode_z/layer{}'.format(k), mode_z.cpu().data.numpy(), epoch)
    return top1.avg, top1_adv.avg

def FAB(val_loader, model, criterion, optimizer, epoch, device, writer,closs,args):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top1_adv = AverageMeter()

    # switch to evaluate mode
    model.eval()
    # model_yuanshi=copy.deepcopy(model)
    if not args.isprune:
        if not args.multi_gpu:
            if model.beta_ema > 0:
                old_params = model.get_params()
                model.load_ema_params()
        else:
            if model.module.beta_ema > 0:
                old_params = model.module.get_params()
                model.module.load_ema_params()

    end = time.time()
    for i, (input_, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            target = target.cuda(non_blocking=True)
            input_ = input_.cuda()
        input_var = torch.autograd.Variable(input_, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        output = model(input_var)


        # loss = criterion(output, input_var,target_var, model,device,optimizer,closs)
        # losses.update(loss.data, input_.size(0))
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        top1.update(prec1, input_.size(0))



        attack=attack_loader(model=model,attack_method='FAB',epsilon=8/255,num_steps=10,step_size=2/255,num_class=10)
        images_adv = attack(input_var,target_var)
        output_adv = model(images_adv)
        acc_adv = accuracy(output_adv, target, )
        top1_adv.update(acc_adv[0], input_.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Err@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'acc_adv {top1_adv.val:.3f} ({top1_adv.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top1_adv=top1_adv))
            # return top1.avg, top1_adv.avg

    print(' * Err@1 {top1.avg:.3f}\t'
          '* top_adv_acc {top1_adv.avg:.3f}'.format(top1=top1, top1_adv=top1_adv))

    if not args.isprune:
        if not args.multi_gpu:
            if model.beta_ema > 0:
                model.load_params(old_params)
        else:
            if model.module.beta_ema > 0:
                model.module.load_params(old_params)

    # log to TensorBoard
    if writer is not None:
        writer.add_scalar('val/loss', losses.avg, epoch)
        writer.add_scalar('val/err', top1.avg, epoch)
        layers = model.layers if not args.multi_gpu else model.module.layers
        for k, layer in enumerate(layers):
            if hasattr(layer, 'qz_loga'):
                mode_z = layer.sample_z(1, sample=0).view(-1)
                writer.add_histogram('mode_z/layer{}'.format(k), mode_z.cpu().data.numpy(), epoch)
    return top1.avg, top1_adv.avg

def Square(val_loader, model, criterion, optimizer, epoch, device, writer,closs,args):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top1_adv = AverageMeter()

    # switch to evaluate mode
    model.eval()
    # model_yuanshi=copy.deepcopy(model)
    if not args.isprune:
        if not args.multi_gpu:
            if model.beta_ema > 0:
                old_params = model.get_params()
                model.load_ema_params()
        else:
            if model.module.beta_ema > 0:
                old_params = model.module.get_params()
                model.module.load_ema_params()

    end = time.time()
    for i, (input_, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            target = target.cuda(non_blocking=True)
            input_ = input_.cuda()
        input_var = torch.autograd.Variable(input_, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        output = model(input_var)


        # loss = criterion(output, input_var,target_var, model,device,optimizer,closs)
        # losses.update(loss.data, input_.size(0))
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        top1.update(prec1, input_.size(0))



        attack=attack_loader(model=model,attack_method='Square',epsilon=8/255,num_steps=10,step_size=2/255,num_class=10)
        images_adv = attack(input_var,target_var)
        output_adv = model(images_adv)
        acc_adv = accuracy(output_adv, target, )
        top1_adv.update(acc_adv[0], input_.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Err@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'acc_adv {top1_adv.val:.3f} ({top1_adv.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top1_adv=top1_adv))
            # return top1.avg, top1_adv.avg

    print(' * Err@1 {top1.avg:.3f}\t'
          '* top_adv_acc {top1_adv.avg:.3f}'.format(top1=top1, top1_adv=top1_adv))

    if not args.isprune:
        if not args.multi_gpu:
            if model.beta_ema > 0:
                model.load_params(old_params)
        else:
            if model.module.beta_ema > 0:
                model.module.load_params(old_params)

    # log to TensorBoard
    if writer is not None:
        writer.add_scalar('val/loss', losses.avg, epoch)
        writer.add_scalar('val/err', top1.avg, epoch)
        layers = model.layers if not args.multi_gpu else model.module.layers
        for k, layer in enumerate(layers):
            if hasattr(layer, 'qz_loga'):
                mode_z = layer.sample_z(1, sample=0).view(-1)
                writer.add_histogram('mode_z/layer{}'.format(k), mode_z.cpu().data.numpy(), epoch)
    return top1.avg, top1_adv.avg

def CW_inf(val_loader, model, criterion, optimizer, epoch, device, writer,closs,args):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top1_adv = AverageMeter()

    # switch to evaluate mode
    model.eval()
    # model_yuanshi=copy.deepcopy(model)
    if not args.isprune:
        if not args.multi_gpu:
            if model.beta_ema > 0:
                old_params = model.get_params()
                model.load_ema_params()
        else:
            if model.module.beta_ema > 0:
                old_params = model.module.get_params()
                model.module.load_ema_params()

    end = time.time()
    for i, (input_, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            target = target.cuda(non_blocking=True)
            input_ = input_.cuda()
        input_var = torch.autograd.Variable(input_, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        output = model(input_var)


        # loss = criterion(output, input_var,target_var, model,device,optimizer,closs)
        # losses.update(loss.data, input_.size(0))
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        top1.update(prec1, input_.size(0))



        attack=attack_loader(model=model,attack_method='CW_inf',epsilon=8/255,num_steps=10,step_size=2/255,num_class=100)
        images_adv = attack(input_var,target_var)
        output_adv = model(images_adv)
        acc_adv = accuracy(output_adv, target, )
        top1_adv.update(acc_adv[0], input_.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Err@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'acc_adv {top1_adv.val:.3f} ({top1_adv.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top1_adv=top1_adv))
            # return top1.avg, top1_adv.avg

    print(' * Err@1 {top1.avg:.3f}\t'
          '* top_adv_acc {top1_adv.avg:.3f}'.format(top1=top1, top1_adv=top1_adv))

    if not args.isprune:
        if not args.multi_gpu:
            if model.beta_ema > 0:
                model.load_params(old_params)
        else:
            if model.module.beta_ema > 0:
                model.module.load_params(old_params)

    # log to TensorBoard
    if writer is not None:
        writer.add_scalar('val/loss', losses.avg, epoch)
        writer.add_scalar('val/err', top1.avg, epoch)
        layers = model.layers if not args.multi_gpu else model.module.layers
        for k, layer in enumerate(layers):
            if hasattr(layer, 'qz_loga'):
                mode_z = layer.sample_z(1, sample=0).view(-1)
                writer.add_histogram('mode_z/layer{}'.format(k), mode_z.cpu().data.numpy(), epoch)
    return top1.avg, top1_adv.avg