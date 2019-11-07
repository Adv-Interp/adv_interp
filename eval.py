from __future__ import print_function
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import sys
import datetime

from tqdm import tqdm
from networks import *
import utils
from utils import softCrossEntropy, CWLoss, Attack_PGD

parser = argparse.ArgumentParser(
    description='Adversarial Interpolation Training')

parser.register('type', 'bool', utils.str2bool)

parser.add_argument('--resume',
                    '-r',
                    action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--attack', default=True, type='bool', help='attack')
parser.add_argument('--model_dir', type=str, help='model path')
parser.add_argument('--init_model_pass',
                    default='-1',
                    type=str,
                    help='init model pass')

parser.add_argument('--attack_method',
                    default='pgd',
                    type=str,
                    help='adv_mode (natural, pdg or cw)')
parser.add_argument('--attack_method_list', type=str)

parser.add_argument('--log_step', default=7, type=int, help='log_step')

parser.add_argument('--num_classes', default=10, type=int, help='num classes')
parser.add_argument('--batch_size_test',
                    default=100,
                    type=int,
                    help='batch size for testing')
parser.add_argument('--image_size', default=32, type=int, help='image size')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transform_test)

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=args.batch_size_test,
                                         shuffle=False,
                                         num_workers=2)

print('======= WideResenet 28-10 ========')
basic_net = WideResNet(depth=28, num_classes=args.num_classes, widen_factor=10)

basic_net = basic_net.to(device)
if device == 'cuda':
    basic_net = torch.nn.DataParallel(basic_net)
    cudnn.benchmark = True

# load parameters

if args.resume and args.init_model_pass != '-1':
    # Load checkpoint.
    print('==> Resuming from saved checkpoint..')
    f_path_latest = os.path.join(args.model_dir, 'latest')
    f_path = os.path.join(args.model_dir,
                          ('checkpoint-%s' % args.init_model_pass))
    if not os.path.isdir(args.model_dir):
        print('train from scratch: no checkpoint directory or file found')
    elif args.init_model_pass == 'latest' and os.path.isfile(f_path_latest):
        checkpoint = torch.load(f_path_latest)
        basic_net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        print('resuming from epoch %s in latest' % start_epoch)
    elif os.path.isfile(f_path):
        checkpoint = torch.load(f_path)
        basic_net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        print('resuming from epoch %s' % start_epoch)
    elif not os.path.isfile(f_path) or not os.path.isfile(f_path_latest):
        print('train from scratch: no checkpoint directory or file found')

# configs
config_natural = {'train': False, 'attack': False}

config_fgsm = {
    'train': False,
    'v_min': -1.0,
    'v_max': 1.0,
    'epsilon': 8.0 / 255 * 2,
    'num_steps': 1,
    'step_size': 8.0 / 255 * 2,
    'random_start': True
}

config_pgd = {
    'train': False,
    'v_min': -1.0,
    'v_max': 1.0,
    'epsilon': 8.0 / 255 * 2,
    'num_steps': 20,
    'step_size': 2.0 / 255 * 2,
    'random_start': True,
    'loss_func': torch.nn.CrossEntropyLoss(reduction='none')
}

config_cw = {
    'train': False,
    'v_min': -1.0,
    'v_max': 1.0,
    'epsilon': 8.0 / 255 * 2,
    'num_steps': 20,
    'step_size': 2.0 / 255 * 2,
    'random_start': True,
    'loss_func': CWLoss(args.num_classes)
}


def test_main(epoch, net):
    net.eval()
    test_loss = 0.0
    correct = 0.0
    total = 0.0

    iterator = tqdm(testloader, ncols=0, leave=False)
    for batch_idx, (inputs, targets) in enumerate(iterator):
        start_time = time.time()
        inputs, targets = inputs.to(device), targets.to(device)

        pert_inputs = inputs.detach()

        outputs, _ = net(pert_inputs, targets)

        duration = time.time() - start_time

        _, predicted = outputs.max(1)
        batch_size = targets.size(0)
        total += batch_size
        correct_num = predicted.eq(targets).sum().item()
        correct += correct_num
        iterator.set_description(
            str(predicted.eq(targets).sum().item() / targets.size(0)))

        if batch_idx % args.log_step == 0:
            print(
                "Step %d, Duration %.2f, Current-batch-acc %.2f, Avg-acc %.2f"
                % (batch_idx, duration, 100. * correct_num / batch_size,
                   100. * correct / total))

    acc = 100. * correct / total
    print('Test accuracy: %.2f' % (acc))
    return acc


attack_list = args.attack_method_list.split('-')
attack_num = len(attack_list)

for attack_idx in range(attack_num):

    args.attack_method = attack_list[attack_idx]

    if args.attack_method == 'natural':
        print('======Evaluation using Natural Images======')
        net = Attack_PGD(basic_net, config_natural)
    elif args.attack_method.upper() == 'FGSM':
        print('======Evaluation under FGSM Attack======')
        net = Attack_PGD(basic_net, config_fgsm)
    elif args.attack_method.upper() == 'PGD':
        print('======Evaluation under PGD Attack======')
        net = Attack_PGD(basic_net, config_pgd)
    elif args.attack_method.upper() == 'CW':
        print('======Evaluation under CW Attack======')
        net = Attack_PGD(basic_net, config_cw)
    else:
        raise Exception(
            'Should be a valid attack method. The specified attack method is: {}'
            .format(args.attack_method))

    criterion = nn.CrossEntropyLoss(reduction='mean')

    test_main(0, net)
