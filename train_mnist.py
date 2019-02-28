"""
Train MNIST
"""
import torch
from torchvision.datasets import mnist
from torchvision import transforms
from torch.utils.data import DataLoader
from ep_mlp import EPMLP
from fp_solver import FixedStepSolver
from tensorboardX import SummaryWriter
from time import time

# ARGS
BATCH_SIZE = 128
HIDDEN_SIZES = [500]
STEP_SIZE = 0.5
MAX_STEPS = 50
LR = 0.01
LOGGING_STEPS = 5
DEVICE = 'cuda'
EPOCHS = 5

# GLOBAL stuff
WRITER = SummaryWriter('./logs')


class RunningAvg:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def reset(self):
        self.sum = 0
        self.count = 0

    def record(self, val, num):
        self.sum += val * num
        self.count += num

    def get_avg(self):
        return self.sum / self.count if self.count > 0 else 0.


class OneHot(object):
    def __init__(self, num_class):
        self.num_class = num_class

    def __call__(self, label):
        oh_vec = torch.Tensor(self.num_class).zero_()
        oh_vec[label] = 1.
        return oh_vec


def get_data_loaders():
    img_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.view(-1))])
    train_dset = mnist.MNIST(root='./mnist_data', train=True,
                             download=True,
                             transform=img_transform,
                             target_transform=OneHot(10))
    val_dset = mnist.MNIST(root='./mnist_data', train=False,
                           download=True,
                           transform=img_transform,
                           target_transform=OneHot(10))
    train_loader = DataLoader(train_dset, batch_size=BATCH_SIZE,
                              shuffle=True)
    val_loader = DataLoader(val_dset, batch_size=BATCH_SIZE,
                            shuffle=False)
    return train_loader, val_loader


def get_model():
    model = EPMLP(784, 10, HIDDEN_SIZES, device=torch.device(DEVICE))
    solver = FixedStepSolver(step_size=STEP_SIZE, max_steps=MAX_STEPS)
    return model, solver


def get_opt(model):
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    return opt


def get_avg_cost_and_corrects(free_states, labels, model):
    avg_costs = torch.mean(model.get_cost(free_states, labels))
    out = free_states[-1]
    preds = out.max(1)[1]
    trus = labels.max(1)[1]
    avg_corrects = (preds == trus).float().mean()
    return avg_costs.item(), avg_corrects.item()


def train(solver, model, opt, dataloader, global_step):
    acc_stats = RunningAvg()
    cost_stats = RunningAvg()
    device = model.device
    for imgs, labels in dataloader:
        imgs = imgs.to(device=device)
        labels = labels.to(device=device)
        start = time()
        free_states = model.free_phase(imgs, solver)
        clamp_states = model.clamp_phase(imgs, labels, solver, 1,
                                         out=free_states[-1],
                                         hidden_units=free_states[:-1])
        fp_time = time() - start

        opt.zero_grad()
        start = time()
        model.set_gradients(imgs, free_states, clamp_states)
        grad_time = time() - start
        opt.step()
        global_step += 1

        # Record stats and report
        with torch.no_grad():
            avg_cost, avg_corrects = get_avg_cost_and_corrects(free_states, labels, model)
        acc_stats.record(avg_corrects, imgs.size(0))
        cost_stats.record(avg_cost, imgs.size(0))
        if global_step % LOGGING_STEPS == 0:
            print('At step {}, cost: {:.4f}, acc: {:.2f}, '
                  'fp time: {:.3f}, grad time: {:.3f}'.format(global_step,
                                                              cost_stats.get_avg(),
                                                              acc_stats.get_avg() * 100, fp_time, grad_time))
            WRITER.add_scalar('train/cost', cost_stats.get_avg(), global_step=global_step)
            WRITER.add_scalar('train/acc', acc_stats.get_avg() * 100, global_step=global_step)
            acc_stats.reset()
            cost_stats.reset()
    return global_step


def validate(solver, model, dataloader, global_step):
    acc_stats = RunningAvg()
    cost_stats = RunningAvg()
    device = model.device
    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        free_states = model.free_phase(imgs, solver)

        # Record stats and report
        with torch.no_grad():
            avg_cost, avg_corrects = get_avg_cost_and_corrects(free_states, labels, model)
        acc_stats.record(avg_corrects, imgs.size(0))
        cost_stats.record(avg_cost, imgs.size(0))
    print('At step {}, '
          'validation cost： {:.4f}, '
          'validation acc: {:.2f}'.format(global_step,
                                          cost_stats.get_avg(),
                                          acc_stats.get_avg() * 100))
    WRITER.add_scalar('valid/cost', cost_stats.get_avg(), global_step=global_step)
    WRITER.add_scalar('valid/acc', acc_stats.get_avg() * 100, global_step=global_step)


def main():
    train_loader, val_loader = get_data_loaders()
    model, solver = get_model()
    opt = get_opt(model)
    print('Train on {}'.format(model.device))
    global_step = 0
    epoch = 0
    while epoch < EPOCHS:
        global_step = train(solver, model, opt, val_loader, global_step)
        validate(solver, model, val_loader, global_step)
        epoch += 1



if __name__ == '__main__':
    """ Main loop """
    main()
