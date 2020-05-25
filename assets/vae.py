# VAE implementation exercise for Depth First Learning Curriculum: Normalizing Flows for Variational Inference.
# This is a stripped-down version of the PyTorch VAE example available at https://github.com/pytorch/examples/blob/master/vae/main.py.

from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # fc1, fc21 and fc22 are used by the encoder.
        # fc1 takes a vectorized MNIST image as input
        # fc21 and fc22 are both attached to the activation output of fc1 (using ReLU).
        # fc21 outputs the means, and fc22 the log-variances of
        # each component of th 20-dimensional latent Gaussian.
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        # fc3 and fc4 are connected in series as the decoder.
        # fc3 takes a realization from the latent space as input
        # and the decoder generates a vectorized 28x28 image.
        # The output of fc3 passes through a ReLU,
        # while fc4 uses a sigmoid in order to output a probability for each pixel
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    # TODO: Implement the following four functions.  Note that they should be able to accept arguments containing stacked information for multiple observations
    # e.g. a minibatch rather than a single observation.  Your solution will need to handle this.  If you treat the arguments as
    # representing a single observation in your logic, in most cases broadcasting will do the rest of the job automatically for you.
    def encode(self, x):
        # This should return the outputs of fc21 and fc22 as a tuple
        pass

    def reparameterize(self, mu, logvar):
        # This should sample vectors from an isotropic Gaussian, and use these to generate
        # and return observations with a mean vectors from mu, and log-variances of log-var
        pass

    def decode(self, z):
        # Pass z through the decoder. For each 20-dimensional latent realization, there should be a 784-dimensional vector of
        #probabilities generated, one per pixel
        pass

    def forward(self, x):
        # For each observation in x:
        # 1. Pass it through the encoder to get predicted variational distribution parameters
        # 2. Reparameterize an isotropic Gaussian with these parameters to get sample latent variable realizations
        # 3. Pass the realization through the encoder to get predicted pixel probabilities
        # Return a tuple with 3 elements: (a) the predicted pixel probabilities, (b) the predicted variational means, and (c) the predicted variational log-variances
        x = x.view(-1,784) # Reshape x to provide suitable inputs to the encoder
        pass

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# TODO: Implement this loss function
def loss_function(recon_x, x, mu, logvar):
    # The loss should be (an estimate of) the negative ELBO - remember we wish to maximise the ELBO - but the ELBO can be written in a number of forms.
    # In this case, the prior for the latent variable and the variational posterior are both Gaussians, and we will exploit this.
    # Specifically, we can analytically calculate a part of the ELBO, and only use Monte Carlo estimation for the rest.
    # 1. We use the form of the ELBO which includes a KL divergence between the latent prior and the variational family
    # - see the form at the bottom of page 6 of Blei et al's "Variational Inference: A Review for Statisticians".
    # 2. In this case, the expression for the relevant KL divergence can be obtained from Exercise (e) in Week 1.
    #
    # The other term is the expected conditional log-likelihood, which is estimated using a single Monte-Carlo sample.
    # For the log-likelihood, one evaluates the probability of observing an input point given the "conditional distribution" for
    # observations output by the network - in this case, each pixel is independently Bernoulli with parameter equal to the output probability.
    # You may find torch.nn.functional's binary_cross_entropy function useful here.
    #
    # Additional: the extraction of the KL divergence as above reduces the variance.  Investigate the effect of directly estimating
    # the full ELBO term for each observation with a single Monte Carlo sample.
    #
    # You may find torch.nn.functional's binary_cross_entropy function useful.
    #
    # Return a single value accumulating the loss over the whole batch.
    #
    # Arguments:
    # x is the batch of observations
    # recon_x, mu, and logvar are the outputs of forward(x) (above) - see the usage below
    x = x.view(-1,784) # Reshape x to provide suitable inputs to the encoder
    pass

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')
