
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchattacks
from torchattacks.attack import Attack
def pgd_whitebox(
    model,
    x,
    y,
    device,
    epsilon,
    num_steps,
    step_size,
    clip_min,
    clip_max,
    is_random=True,
):

    x_pgd = Variable(x.data, requires_grad=True)
    if is_random:
        random_noise = (
            torch.FloatTensor(x_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        )
        x_pgd = Variable(x_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([x_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(x_pgd), y)
        loss.backward()
        eta = step_size * x_pgd.grad.data.sign()
        x_pgd = Variable(x_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(x_pgd.data - x.data, -epsilon, epsilon)
        x_pgd = Variable(x.data + eta, requires_grad=True)
        x_pgd = Variable(torch.clamp(x_pgd, clip_min, clip_max), requires_grad=True)

    return x_pgd

class CW_Linf(Attack):

    def __init__(self, model, eps, c=0.1, kappa=0, steps=1000, lr=0.01):
        super().__init__("CW_Linf", model)
        self.eps = eps
        self.alpha = eps/steps * 2.3
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        adv_images = images.clone().detach()


        # Starting at a uniformly random point
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()


        for step in range(self.steps):

            adv_images.requires_grad = True

            outputs = self.model(adv_images)
            f_loss = self.f(outputs, labels).sum()
            cost = f_loss

            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            adv_images = adv_images.detach() - self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

    def tanh_space(self, x):
        return 1/2*(torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        return self.atanh(x*2-1)

    def atanh(self, x):
        return 0.5*torch.log((1+x)/(1-x))

    # f-function in the paper
    def f(self, outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())

        if self.targeted:
            return torch.clamp((i-j), min=-self.kappa)
        else:
            return torch.clamp((j-i), min=-self.kappa)



def attack_loader(model,attack_method,epsilon,num_steps,step_size,num_class):
    if attack_method=='PGD-10':
        return torchattacks.PGD(model=model,eps=epsilon,alpha=step_size,steps=num_steps,random_start=True)
    elif attack_method=='PGD-20':
        return torchattacks.PGD(model=model, eps=epsilon, alpha=step_size, steps=num_steps, random_start=True)
    elif attack_method=='PGD-50':
        return torchattacks.PGD(model=model, eps=epsilon, alpha=step_size, steps=num_steps, random_start=True)
    elif attack_method=='AA':
        return  torchattacks.AutoAttack(model=model,eps=epsilon,n_classes=num_class)
    elif attack_method=='APGD':
        return torchattacks.APGD(model=model,eps=epsilon,loss='ce',steps=30)
    elif attack_method=='CW_inf':
        return CW_Linf(model=model,eps=epsilon,lr=0.1,steps=30)
    elif attack_method=='FAB':
        return torchattacks.FAB(model=model,eps=epsilon,multi_targeted=True,n_classes=num_class,n_restarts=1)
    elif attack_method=='Square':
        return torchattacks.Square(model=model,eps=epsilon,n_queries=5000,n_restarts=1)


