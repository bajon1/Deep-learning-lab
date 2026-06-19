import torch
import torch.nn as nn
import torch.optim as optim
import copy


def train_baseline(model, train_loader, val_loader, device, weight_decay, epochs=100, lr=1e-3):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(X_batch), y_batch).item() * len(X_batch)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

    model.load_state_dict(best_weights)

    return model



def train_mtl(model, train_loader, val_loader, device, weight_decay, alpha_recon, beta_reg, epochs=100, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    reconstruction_criterion = nn.MSELoss()
    regression_criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_weights  = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            X_hat, y_hat = model(X_batch)
            recon_loss = reconstruction_criterion(X_hat, X_batch)
            reg_loss = regression_criterion(y_hat, y_batch)
            total_loss = alpha_recon * recon_loss + beta_reg * reg_loss

            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item() * len(X_batch)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                X_hat, y_hat = model(X_batch)
                recon_loss = reconstruction_criterion(X_hat, X_batch)
                reg_loss = regression_criterion(y_hat, y_batch)
                val_loss += (alpha_recon * recon_loss + beta_reg * reg_loss).item() * len(X_batch)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

    model.load_state_dict(best_weights)

    return model