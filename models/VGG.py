import torch
import torch.nn as nn
from models.L0WideResNet.l0_layers import L0Conv2d
from models.L0WideResNet.base_layers import MAPConv2d, MAPDense
from copy import deepcopy

_AFFINE = True
# _AFFINE = False

defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}
class VGG(nn.Module):
    def __init__(self, dataset='cifar10', depth=16, mean=None, std=None, init_weights=True,droprate_init=0.3, cfg=None,num_class=10,
                 args=None,lambdda=1.,a=1,N=50000,weight_decay=5e-4,beta_ema=0.99,lamba=0.01,iscloss=False,temperature=2./3.,local_rep=False):
        super(VGG, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]
        self.mean = mean.view(1, -1, 1, 1)
        self.std = std.view(1, -1, 1, 1)
        self.lambdda=lambdda
        self.lamba=lamba
        self.a=a
        self.N=N
        self.weight_decay = N * weight_decay
        self.beta_ema=beta_ema
        self.iscloss=iscloss
        self.args = args
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.feature = self.make_layers(cfg, True,droprate_init,temperature=temperature,local_rep=local_rep)
        self.dataset = dataset
        self.classifier = MAPDense(cfg[-1], num_class,args=args,weight_decay=self.weight_decay,)
        self.layers, self.bn_params = [], []
        for m in self.modules():
            if isinstance(m, MAPDense) or isinstance(m, MAPConv2d) or isinstance(m, L0Conv2d):
                self.layers.append(m)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                self.bn_params += [m.weight, m.bias]
        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

        print('Using weight decay: {}'.format(self.weight_decay))



        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False,droprate_init=0.5,temperature=2./3.,local_rep=False):
        layers = []
        in_channels = 3
        for i,v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if i==0:
                    MAPConv2dfirst=MAPConv2d(in_channels,v,kernel_size=3,stride=1,padding=1,bias=False,weight_decay=self.weight_decay,args=self.args)
                    if batch_norm:
                        layers += [MAPConv2dfirst, nn.BatchNorm2d(v, affine=_AFFINE), nn.ReLU(inplace=True)]
                    else:
                        layers += [MAPConv2dfirst, nn.ReLU(inplace=True)]
                else:
                    conv2d = L0Conv2d(in_channels, v, kernel_size=3,stride=1, padding=1, bias=False,droprate_init=droprate_init,weight_decay=self.weight_decay,
                                      local_rep=local_rep,lamba=self.lamba,temperature=temperature,a=self.a,args=self.args)
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(v, affine=_AFFINE), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x, pop=False, inter=False):
        # Feature visualization
        if not inter:
            # x = (x-self.mean) / self.std
            x = self.feature(x)
            if pop:
                return x


        # if self.dataset == 'tiny':
        #     x = nn.AvgPool2d(4)(x)
        # else:
        #     x = nn.AvgPool2d(2)(x)
        # x=self.avgpool(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            # if isinstance(m, L0Conv2d):
            #     n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            #     m.weights.data.normal_(0, math.sqrt(2. / n))
            #     if m.bias is not None:
            #         m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1.0)
                    m.bias.data.zero_()
            # elif isinstance(m, MAPDense):
            #     m.weight.data.normal_(0, 0.01)
            #     m.bias.data.zero_()
    def regularization(self):
        regularization = 0.
        Loss_conditiona=0.
        # det=0.
        for layer in self.layers:
            regularization += - (self.lambdda/ self.N) * layer.regularization()
            if isinstance(layer,MAPDense):
                Loss_conditiona+=(torch.log(1e-5 +torch.det(torch.matmul(layer.weight.T, layer.weight)/(layer.weight.shape[1])))) ** 2
                # det=torch.det(torch.matmul(layer.weight.T, layer.weight)/(layer.weight.shape[1]/5.))
        for bnw in self.bn_params:
            if self.weight_decay > 0:
                regularization += (self.weight_decay / self.N) * .5 * torch.sum(bnw.pow(2))  # todo 这还有个正则项呢
        if torch.cuda.is_available():
            regularization = regularization.cuda()
            Loss_conditiona = Loss_conditiona.cuda()
        return regularization,Loss_conditiona











    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0., 0.
        for layer in self.layers:
            try:
                e_fl, e_l0 = layer.count_expected_flops_and_l0()
                expected_flops += e_fl
                expected_l0    += e_l0
            except:
                pass
        return expected_flops, expected_l0

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self,*args):

        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema**self.steps_ema))




    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params