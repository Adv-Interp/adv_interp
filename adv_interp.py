''' Adversarial Interpolation '''

from utils import cos_dist
import copy
import pickle
import torch
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable


def adv_interp(inputs,
               y,
               base_net,
               num_classes,
               epsilon=8,
               epsilon_y=0.5,
               v_min=0,
               v_max=255):
    # x: image batch with shape [batch_size, c, h, w]
    # y: one-hot label batch with shape [batch_size, num_classes]
    net = copy.deepcopy(base_net)
    x = inputs.clone()

    inv_index = torch.arange(x.size(0) - 1, -1, -1).long()
    x_prime = x[inv_index, :, :, :].detach()
    y_prime = y[inv_index, :]
    x_init = x.detach() + torch.zeros_like(x).uniform_(-epsilon, epsilon)

    x_init.requires_grad_()
    zero_gradients(x_init)
    if x_init.grad is not None:
        x_init.grad.data.fill_(0)
    net.eval()

    fea_b = net(x_init, mode='feature')
    fea_t = net(x_prime, mode='feature')

    loss_adv = cos_dist(fea_b, fea_t)
    net.zero_grad()
    loss_adv = loss_adv.mean()
    loss_adv.backward(retain_graph=True)

    x_tilde = x_init.data - epsilon * torch.sign(x_init.grad.data)

    x_tilde = torch.min(torch.max(x_tilde, inputs - epsilon), inputs + epsilon)
    x_tilde = torch.clamp(x_tilde, v_min, v_max)

    y_bar_prime = (1 - y_prime) / (num_classes - 1.0)
    y_tilde = (1 - epsilon_y) * y + epsilon_y * y_bar_prime

    return x_tilde.detach(), y_tilde.detach()