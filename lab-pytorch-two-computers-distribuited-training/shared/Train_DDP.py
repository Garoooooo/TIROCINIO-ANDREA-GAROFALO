import os
import sys
import time
import torch
from torch import nn
import torch.distributed as dist  # Strumenti PyTorch per la comunicazione tra macchine
from torch.nn.parallel import DistributedDataParallel as DDP  # Wrapper per la sincronizzazione dei pesi via rete
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler  # Partizionamento del dataset tra pc1 e pc2
from torchvision import datasets
from torchvision.transforms import v2


# 1. Setup di rete: configurazione dei nodi per l'individuazione e la connessione reciproca
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '192.168.1.10'  # Indirizzo IP del pc master (pc1)
    os.environ['MASTER_PORT'] = '29500'  # Porta TCP di ascolto su pc1
    os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'  # Selezione dell'interfaccia di rete virtuale per Gloo (Libreria per comunicazioni collettive)

    # inizializzazione canale del gruppo TCP/IP
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    print(f"--> [PC{rank + 1}] Connesso al canale per apprendimento distribuito")


# Chiusura della connessione e rilascio delle porte di rete
def cleanup():
    dist.destroy_process_group()


# 2. Struttura della rete neurale
# Creazione del cervello del modello: prende l'immagine 28x28 e la stende in una striscia di 784 numeri.
# Passa i dati attraverso due livelli intermedi da 512 neuroni per imparare le forme dei vestiti.
# Genera 10 punteggi finali, uno per ciascuna categoria di vestito da riconoscere.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()  # Srotola l'immagine 28x28 in un vettore
        self.linear_relu_stack = nn.Sequential(     #
            nn.Linear(28 * 28, 512),                #
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )                                            # percorso che fa l'immagine dentro il cervello del modello per indovinare il vestito

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)           #
        return logits                                #


# 3. Funzione di allenamento e sincronizzazione
# Il PC fa le sue previsioni sul suo gruppo di immagini e misura quanto ha sbagliato.
# Con il backward pass scambia gli errori via rete con l'altro PC per calcolare una media comune.
# Aggiorna i pesi del modello con i dati mediati e resetta i calcoli per il gruppo successivo.
def train(dataloader, model, loss_fn, optimizer, rank):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        # Calcolo dell'errore
        # Scambio dei gradienti tra i due PC e media prima dell'aggiornamento dei parametri
        loss.backward()

        optimizer.step()  # Aggiornamento dei pesi (i due modelli restano identici)
        optimizer.zero_grad()  # Pulizia dei calcoli precedenti per il nuovo gruppo di immagini

        if batch % 20 == 0:
            loss_val = loss.item()
            current = (batch + 1) * len(X)
            print(f"[PC{rank + 1}] Loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")


# 4. Funzione di verifica finale
# Prova il modello su 10.000 immagini nuove di test senza toccare i pesi della rete.
# Guarda quale vestito e stato scelto dal modello e controlla se la risposta e corretta.
# Calcola la percentuale di risposte azzeccate e l'errore medio per vedere se ha imparato bene.
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0.0, 0.0

    with torch.no_grad():  # Disattiva i calcoli di aggiornamento perche stiamo solo testando
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # Prende la scelta con punteggio piu alto

    test_loss /= num_batches
    correct /= size
    print(f"\nTest Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")


# 5. Programma principale e gestione dei dati
# Prende le 60.000 immagini e le divide a meta: 30.000 a pc1 e 30.000 a pc2 senza doppioni.
# Avvia il modello distribuito con gli stessi identici pesi iniziali su entrambi i computer.
# Fa fare 5 giri di studio rimescolando i dati, poi fa fare verifica e salvataggio solo a pc1.
def demo_basic(rank, world_size):
    setup(rank, world_size)

    # parte aggiunta da me per il contatore di n pacchetti, dimensioni e tempi
    # -------------------------------------------------
    start_time = time.time()
    with open('/sys/class/net/eth0/statistics/tx_bytes') as f:
        start_bytes = int(f.read())
    with open('/sys/class/net/eth0/statistics/tx_packets') as f:
        start_packets = int(f.read())
    # -------------------------------------------------

    # Caricamento di FashionMNIST dalla cartella shared
    training_data = datasets.FashionMNIST(
        root="/shared/data",
        train=True,
        download=False,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )

    test_data = datasets.FashionMNIST(
        root="/shared/data",
        train=False,
        download=False,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )

    batch_size = 512
    train_sampler = DistributedSampler(training_data, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, sampler=train_sampler)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    torch.manual_seed(42)  # Fissa i valori casuali per far partire entrambi i PC dallo stesso punto
    model = NeuralNetwork()
    ddp_model = DDP(model)  # RIGA FONDAMENTALE: aggancia la rete alla comunicazione di gruppo

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=1e-2)

    epochs = 2
    for t in range(epochs):
        train_sampler.set_epoch(t)
        if rank == 0:
            print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, ddp_model, loss_fn, optimizer, rank)

    # Verifica finale e salvataggio eseguiti solo da pc1 per non sovrascrivere il file a vicenda
    if rank == 0:
        test(test_dataloader, ddp_model, loss_fn)
        torch.save(ddp_model.module.state_dict(), "/shared/model.pth")  # Salva la rete pulita senza il modulo distribuito
        print("Saved PyTorch Model State to /shared/model.pth")

        # parte aggiunta da me per il contatore di n pacchetti, dimensioni e tempi
        # ----------------------------------------------------------------------------
        elapsed = time.time() - start_time
        with open('/sys/class/net/eth0/statistics/tx_bytes') as f:
            total_bytes = int(f.read()) - start_bytes
        with open('/sys/class/net/eth0/statistics/tx_packets') as f:
            total_packets = int(f.read()) - start_packets

        print(f"Tempo di esecuzione: {elapsed:.2f} s")
        print(f"Pacchetti inviati:   {total_packets}")
        print(f"Dimensioni totali:   {total_bytes / (1024 * 1024):.2f} MB ({total_bytes} bytes)")
        # ----------------------------------------------------------------------------

    cleanup()


# 6. Avvio da terminale
# Legge il numero del PC passato da riga di comando (0 per pc1, 1 per pc2).
# Fissa la dimensione della squadra a 2 macchine in totale.
# Lancia il programma principale con i parametri impostati.
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 /shared/Train.py <RANK>")
        sys.exit(1)

    rank = int(sys.argv[1])  # 0 per pc1, 1 per pc2
    world_size = 2  # Totale macchine: pc1 e pc2
    demo_basic(rank, world_size)