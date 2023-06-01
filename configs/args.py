import argparse


def parse_args():
    parser=argparse.ArgumentParser(description='pytorch training')
    parser.add_argument("--configs", type=str, default="", help="configs file", )
    parser.add_argument('--epochs', default=200, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--print-freq', '-p', default=100, type=int,
                        help='print frequency (default: 100)')
    parser.add_argument('--depth', default=28, type=int,
                        help='total depth of the network (default: 28)')
    parser.add_argument('--N',default=50000,type=int,help='the length of training dataset')
    parser.add_argument('--width', default=10, type=int,
                        help='total width of the network (default: 10)')
    parser.add_argument('--droprate_init', default=0.3, type=float,
                        help='dropout probability (default: 0.3)')
    parser.add_argument('--no-augment', dest='augment', action='store_false',
                        help='whether to use standard augmentation (default: True)')
    parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                        help='To not use bottleneck block')
    parser.add_argument('--save_path', default='L0WideResNet', type=str,
                        help='name of experiment')
    parser.add_argument('--no-tensorboard', dest='tensorboard', action='store_false',
                        help='whether to use tensorboard (default: True)')
    parser.add_argument('--multi_gpu', default=True)
    parser.add_argument('--lamba', type=float, default=0.001,
                        help='Coefficient for the L0WideResNet term.')
    parser.add_argument('--beta_ema', type=float, default=0.99)
    parser.add_argument('--lr_decay_ratio', type=float, default=0.2)
    parser.add_argument('--dataset', choices=['c10', 'c100'], default='c10')
    parser.add_argument('--dataset_path',default='',type=str)
    parser.add_argument('--local_rep', action='store_true')
    parser.add_argument('--epoch_drop', nargs='*', type=int, default=(60, 120, 160))
    parser.add_argument('--temp', type=float, default=2. / 3.)
    #resume
    parser.add_argument('--resume', default=False, type=str,
                        help='path to latest checkpoint (default: none)')
    #prune
    parser.add_argument('--sparsity',default=0.99,help='the sparsity of models ')

    #train hyperparamater
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        help='mini-batch size (default: 128)')
    parser.add_argument('--iscloss', default=True, help='whether to use condition number (True or False)')
    parser.add_argument('--a', default=None, help='the hyper-paramter of transformed L1')
    parser.add_argument('--lambdda', default=10, help='the ture coefficient of L0WideResNet penalty')
    parser.add_argument('--isadvtraining', default=True, help='whether to use adv training')
    parser.add_argument('--isprune',default=False,help='whether to use prune popup_scores')
    parser.add_argument('--isfinetune', default=False, help='whether to finetune network')
    parser.add_argument('--amp',default=True,help='autocast training')

    #test hyperparameter
    parser.add_argument('--test_method',default='PGD-10',choices='base,PGD-10,PGD-20,AA,SA,FGSM',help='attack methods')


    parser.set_defaults(bottleneck=True)
    parser.set_defaults(augment=True)
    parser.set_defaults(tensorboard=False)



    return parser.parse_args()



