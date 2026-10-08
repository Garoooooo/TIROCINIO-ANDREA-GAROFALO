import os
import sys
import time
import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from transformer.model import TransformerLM
from transformer.LMConfig import LMConfig
from transformer.dataset import PretrainDataset


# Inizializzazione processo distribuito
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '192.168.1.10'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    print(f"--> [PC{rank + 1}] Connesso. Sync ogni {SYNC_EVERY} batch")


# Chiusura gruppo distribuito
def cleanup():
    dist.destroy_process_group()


# Creazione del modello
def build_model():
    lm_config = LMConfig(dim=128, n_layers=4, max_seq_len=128)
    model = TransformerLM(lm_config)
    return model


# Sincronizzazione pesi via all-reduce
def sync_weights(model, world_size):
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= world_size


# Loop di training
def train(dataloader, model, loss_fn, optimizer, rank, world_size):
    model.train()
    batch_count = 0
    for batch, (X, Y, loss_mask) in enumerate(dataloader):
        res = model(X)
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        raw = loss_fct(res.logits.view(-1, res.logits.size(-1)), Y.view(-1)).view(Y.size())
        loss = (raw * loss_mask).sum() / loss_mask.sum()

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        batch_count += 1
        if batch_count % SYNC_EVERY == 0:
            sync_weights(model, world_size)

        if batch % 20 == 0 and rank == 0:
            print(f"Loss: {loss.item():>7f}  [{batch + 1}/{len(dataloader)}]")


# Routine principale
def demo_basic(rank, world_size):
    setup(rank, world_size)

    # Benchmark di rete iniziale
    start_time = time.time()
    with open('/sys/class/net/eth0/statistics/tx_bytes') as f:
        start_bytes = int(f.read())
    with open('/sys/class/net/eth0/statistics/tx_packets') as f:
        start_packets = int(f.read())

    # Caricamento dataset e tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/shared/train-tiny-llm/custom_tokenizer")
    training_data = PretrainDataset(
        "/shared/train-tiny-llm/pretrain_data.jsonl",
        tokenizer,
        max_length=128
    )

    # Configurazione dataloader distribuito
    batch_size = 8
    train_sampler = DistributedSampler(training_data, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, sampler=train_sampler, num_workers=0)

    # Inizializzazione pesi e ottimizzatore
    torch.manual_seed(42)
    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    # Ciclo epoche
    epochs = 1
    for t in range(epochs):
        train_sampler.set_epoch(t)
        if rank == 0:
            print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, None, optimizer, rank, world_size)

        sync_weights(model, world_size)

    # Salvataggio checkpoint e statistiche rete
    if rank == 0:
        torch.save(model.state_dict(), "/shared/train-tiny-llm/out/pretrain_distributed.pth")
        print("Saved PyTorch Model State to /shared/train-tiny-llm/out/pretrain_distributed.pth")

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


# avvio
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python3 /shared/train-tiny-llm/pretrain_distributed.py <RANK> <SYNC_EVERY>")
        sys.exit(1)

    rank = int(sys.argv[1])
    SYNC_EVERY = int(sys.argv[2])
    world_size = 2
    demo_basic(rank, world_size)
