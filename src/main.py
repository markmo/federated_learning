from argparse import ArgumentParser
from model_setup import CNN
from model_util import load_hyperparams, merge_dict
from pathlib import Path
import syft as sy
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

ROOT_DIR = Path(__file__).parent


def train(constants, model, device, federated_train_loader, optimizer, epoch):
    batch_size = constants['batch_size']
    n = len(federated_train_loader)
    model.train()
    for batch_i, (data, target) in enumerate(federated_train_loader):
        model.send(data.location)  # send the model to the right location
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_i % constants['log_interval'] == 0:
            loss = loss.get()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_i * batch_size, n * batch_size, 100. * batch_i / n, loss.item()
            ))


def test(constants, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)
    ))


def run(constant_overwrites):
    config_path = ROOT_DIR / 'hyperparams.yml'
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)
    use_cuda = constants['cuda'] and torch.cuda.is_available()
    hook = sy.TorchHook(torch)

    # The organisations that will participate in training
    org1 = sy.VirtualWorker(hook, id="org1")
    org2 = sy.VirtualWorker(hook, id="org2")

    torch.manual_seed(constants['seed'])
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    batch_size = constants['batch_size']
    test_batch_size = constants['test_batch_size']
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])
    dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    federated_train_loader = sy.FederatedDataLoader(dataset.federate((org1, org2)),
                                                    batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=test_batch_size, shuffle=True, **kwargs)

    model = CNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=constants['learning_rate'])

    for epoch in range(1, constants['n_epochs'] + 1):
        train(constants, model, device, federated_train_loader, optimizer, epoch)
        test(constants, model, device, test_loader)

    if constants['save_model']:
        torch.save(model.state_dict(), 'mnist_cnn.pt')


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Federated Learning Example')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, help='learning_rate')
    parser.add_argument('--train', dest='train', help='training mode', action='store_true')
    parser.set_defaults(train=False)
    args = parser.parse_args()

    run(vars(args))
