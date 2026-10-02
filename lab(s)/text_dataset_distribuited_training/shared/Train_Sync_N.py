import os
import sys
import time
import torch
from torch import nn
import torch.distributed as dist  # Strumenti PyTorch per la comunicazione tra macchine
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler  # Partizionamento del dataset tra pc1 e pc2
from torchvision import datasets
from torchvision.transforms import v2


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '192.168.1.10'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    print(f"--> [PC{rank + 1}] Connesso. Sync ogni {SYNC_EVERY} batch")


def cleanup():
    dist.destroy_process_group()


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)


def sync_weights(model, world_size):
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)  # sommo i pesi di tutti i PC
        param.data /= world_size                           # divido per il numero di PC = media


def train(dataloader, model, loss_fn, optimizer, rank, world_size):
    size = len(dataloader.dataset)
    model.train()
    batch_count = 0
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        batch_count += 1
        if batch_count % SYNC_EVERY == 0:
            sync_weights(model, world_size)

        if batch % 20 == 0 and rank == 0:
            current = (batch + 1) * len(X)
            print(f"Loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0.0, 0.0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"\nTest Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")


def demo_basic(rank, world_size):
    setup(rank, world_size)

    start_time = time.time()
    with open('/sys/class/net/eth0/statistics/tx_bytes') as f:
        start_bytes = int(f.read())
    with open('/sys/class/net/eth0/statistics/tx_packets') as f:
        start_packets = int(f.read())

    training_data = datasets.FashionMNIST(
        root="/shared/data", train=True, download=False,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )
    test_data = datasets.FashionMNIST(
        root="/shared/data", train=False, download=False,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )

    batch_size = 256
    train_sampler = DistributedSampler(training_data, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, sampler=train_sampler)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    torch.manual_seed(42)
    model = NeuralNetwork()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    epochs = 2
    for t in range(epochs):
        train_sampler.set_epoch(t)
        if rank == 0:
            print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, rank, world_size)

        sync_weights(model, world_size)  # sync a fine epoca, sempre

    if rank == 0:
        test(test_dataloader, model, loss_fn)
        torch.save(model.state_dict(), "/shared/model.pth")
        print("Saved PyTorch Model State to /shared/model.pth")

        elapsed = time.time() - start_time
        with open('/sys/class/net/eth0/statistics/tx_bytes') as f:
            total_bytes = int(f.read()) - start_bytes
        with open('/sys/class/net/eth0/statistics/tx_packets') as f:
            total_packets = int(f.read()) - start_packets

        print(f"SYNC_EVERY: {SYNC_EVERY}")
        print(f"Tempo di esecuzione: {elapsed:.2f} s")
        print(f"Pacchetti inviati:   {total_packets}")
        print(f"Dimensioni totali:   {total_bytes / (1024 * 1024):.2f} MB ({total_bytes} bytes)")

    cleanup()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python3 /shared/Train_Sync_N.py <RANK> <SYNC_EVERY>")
        sys.exit(1)

    rank = int(sys.argv[1])
    SYNC_EVERY = int(sys.argv[2])
    world_size = 2
    demo_basic(rank, world_size)