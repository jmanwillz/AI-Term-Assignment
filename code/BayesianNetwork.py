import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm, trange

from BayesianLinear import BayesianLinear
from Helper import Helper


class BayesianNetwork(nn.Module):
    def __init__(self, pi, sigma_1, sigma_2):
        super().__init__()
        self.l1 = BayesianLinear(28 * 28, 400, pi=pi, sigma_1=sigma_1, sigma_2=sigma_2)
        self.l2 = BayesianLinear(400, 400, pi=pi, sigma_1=sigma_1, sigma_2=sigma_2)
        self.l3 = BayesianLinear(400, 10, pi=pi, sigma_1=sigma_1, sigma_2=sigma_2)

    def forward(self, x, sample=False):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = F.log_softmax(self.l3(x, sample), dim=1)
        return x

    def log_prior(self):
        return self.l1.log_prior + self.l2.log_prior + self.l3.log_prior

    def log_variational_posterior(self):
        return (
            self.l1.log_variational_posterior
            + self.l2.log_variational_posterior
            + self.l3.log_variational_posterior
        )

    def sample_elbo(self, input, target, samples):
        outputs = torch.zeros(samples, BATCH_SIZE, CLASSES).to(Helper.get_device())
        log_priors = torch.zeros(samples).to(Helper.get_device())
        log_variational_posteriors = torch.zeros(samples).to(Helper.get_device())
        for i in range(samples):
            outputs[i] = self(input, sample=True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = F.nll_loss(
            outputs.mean(0), target, size_average=False
        )
        loss = (
            log_variational_posterior - log_prior
        ) / NUM_BATCHES + negative_log_likelihood
        return loss, log_prior, log_variational_posterior, negative_log_likelihood


def write_weight_histograms(epoch):
    writer.add_histogram("histogram/w1_mu", net.l1.weight_mu, epoch)
    writer.add_histogram("histogram/w1_rho", net.l1.weight_rho, epoch)
    writer.add_histogram("histogram/w2_mu", net.l2.weight_mu, epoch)
    writer.add_histogram("histogram/w2_rho", net.l2.weight_rho, epoch)
    writer.add_histogram("histogram/w3_mu", net.l3.weight_mu, epoch)
    writer.add_histogram("histogram/w3_rho", net.l3.weight_rho, epoch)
    writer.add_histogram("histogram/b1_mu", net.l1.bias_mu, epoch)
    writer.add_histogram("histogram/b1_rho", net.l1.bias_rho, epoch)
    writer.add_histogram("histogram/b2_mu", net.l2.bias_mu, epoch)
    writer.add_histogram("histogram/b2_rho", net.l2.bias_rho, epoch)
    writer.add_histogram("histogram/b3_mu", net.l3.bias_mu, epoch)
    writer.add_histogram("histogram/b3_rho", net.l3.bias_rho, epoch)


def write_loss_scalars(
    epoch,
    batch_idx,
    loss,
    log_prior,
    log_variational_posterior,
    negative_log_likelihood,
):
    writer.add_scalar("logs/loss", loss, epoch * NUM_BATCHES + batch_idx)
    writer.add_scalar(
        "logs/complexity_cost",
        log_variational_posterior - log_prior,
        epoch * NUM_BATCHES + batch_idx,
    )
    writer.add_scalar("logs/log_prior", log_prior, epoch * NUM_BATCHES + batch_idx)
    writer.add_scalar(
        "logs/log_variational_posterior",
        log_variational_posterior,
        epoch * NUM_BATCHES + batch_idx,
    )
    writer.add_scalar(
        "logs/negative_log_likelihood",
        negative_log_likelihood,
        epoch * NUM_BATCHES + batch_idx,
    )


def train(net, optimizer, epoch):
    net.train()
    if epoch == 0:  # write initial distributions
        write_weight_histograms(epoch)
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(Helper.get_device()), target.to(Helper.get_device())
        net.zero_grad()
        loss, log_prior, log_variational_posterior, negative_log_likelihood = (
            net.sample_elbo(data, target)
        )
        loss.backward()
        optimizer.step()
        write_loss_scalars(
            epoch,
            batch_idx,
            loss,
            log_prior,
            log_variational_posterior,
            negative_log_likelihood,
        )
    write_weight_histograms(epoch + 1)


def test_ensemble():
    net.eval()
    correct = 0
    corrects = np.zeros(TEST_SAMPLES + 1, dtype=int)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(Helper.get_device()), target.to(Helper.get_device())
            outputs = torch.zeros(TEST_SAMPLES + 1, TEST_BATCH_SIZE, CLASSES).to(
                Helper.get_device()
            )
            for i in range(TEST_SAMPLES):
                outputs[i] = net(data, sample=True)
            outputs[TEST_SAMPLES] = net(data, sample=False)
            output = outputs.mean(0)
            preds = preds = outputs.max(2, keepdim=True)[1]
            pred = output.max(1, keepdim=True)[1]  # index of max log-probability
            corrects += (
                preds.eq(target.view_as(pred)).sum(dim=1).squeeze().cpu().numpy()
            )
            correct += pred.eq(target.view_as(pred)).sum().item()
    for index, num in enumerate(corrects):
        if index < TEST_SAMPLES:
            print("Component {} Accuracy: {}/{}".format(index, num, TEST_SIZE))
        else:
            print("Posterior Mean Accuracy: {}/{}".format(num, TEST_SIZE))
    print("Ensemble Accuracy: {}/{}".format(correct, TEST_SIZE))


if __name__ == "__main__":
    writer = SummaryWriter()
    LOADER_KWARGS = (
        {"num_workers": 1, "pin_memory": True} if torch.cuda.is_available() else {}
    )

    PI = 0.5
    SIGMA_1 = torch.FloatTensor([math.exp(-0)])
    SIGMA_2 = torch.FloatTensor([math.exp(-6)])

    BATCH_SIZE = 100
    TEST_BATCH_SIZE = 5

    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            "./fmnist", train=True, download=True, transform=transforms.ToTensor()
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        **LOADER_KWARGS
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            "./fmnist", train=False, download=True, transform=transforms.ToTensor()
        ),
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        **LOADER_KWARGS
    )

    TRAIN_SIZE = len(train_loader.dataset)
    TEST_SIZE = len(test_loader.dataset)
    NUM_BATCHES = len(train_loader)
    NUM_TEST_BATCHES = len(test_loader)

    CLASSES = 10
    TRAIN_EPOCHS = 20
    SAMPLES = 2
    TEST_SAMPLES = 10

    assert (TRAIN_SIZE % BATCH_SIZE) == 0
    assert (TEST_SIZE % TEST_BATCH_SIZE) == 0

    net = BayesianNetwork(pi=PI, sigma_1=SIGMA_1, sigma_2=SIGMA_2).to(Helper.get_device())
    optimizer = optim.Adam(net.parameters())
    for epoch in range(TRAIN_EPOCHS):
        train(net, optimizer, epoch)
    test_ensemble()
