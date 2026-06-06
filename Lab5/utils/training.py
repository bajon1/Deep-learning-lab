import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

def train_clf(model, X_tr, y_tr, X_v, y_v, writer, batch_size=1024, epochs=50, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    writer.add_graph(model, torch.tensor(X_tr[:1]).to(device))
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    val_accs = []
    checkpoint = None

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr).long()),
        batch_size=batch_size, shuffle=True,
        pin_memory=(device == "cuda")
    )

    X_val_t = torch.tensor(X_v).to(device)
    y_val_t = torch.tensor(y_v).long().to(device)

    for epoch in range(epochs):
        train_loss = 0
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(X_batch)
        train_loss /= len(X_tr)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            logits = model(X_val_t)
            val_loss = criterion(logits, y_val_t).item()
            val_losses.append(val_loss)
            val_acc = (logits.argmax(dim=1) == y_val_t).float().mean().item()
            val_accs.append(val_acc)
        writer.add_scalars('Losses', {'train_loss': train_loss, 'val_loss': val_loss}, epoch)
        writer.add_scalar('Val accuracy', val_acc, epoch)
        writer.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'best_val_acc': val_acc
            }

    writer.close()

    if device == "cuda":
        torch.cuda.empty_cache()

    return checkpoint