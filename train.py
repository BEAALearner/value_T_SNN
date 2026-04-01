import torch
from spikingjelly.activation_based import functional
from torch import nn
import time

def evaluate(model, loader, timesteps,  device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            data = data.permute(1, 0, 2, 3, 4)
            output_spikes = functional.multi_step_forward(data, model)
            output_mean = output_spikes.mean(dim=0)
            pred = output_mean.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            functional.reset_net(model)
    return 100. * correct / total

def train(model, train_loader, test_loader, optimizer, scheduler, criterion, logger, timesteps,  device, epochs=10):
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_correct = 0
        total_loss = 0
        total_samples = 0
        total_spike_rate = 0.0 

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.permute(1, 0, 2, 3, 4)
            optimizer.zero_grad()
            output_spikes = functional.multi_step_forward(data, model)
            output_mean = output_spikes.mean(dim=0)
            loss = criterion(output_mean, target)
            loss.backward()
            optimizer.step()
            functional.reset_net(model)

            spike_rate = output_spikes.sum().item() / output_spikes.numel()
            total_spike_rate += spike_rate

            pred = output_mean.argmax(dim=1)
            total_correct += (pred == target).sum().item()
            total_samples += target.size(0)
            total_loss += loss.item()

        scheduler.step()

        train_acc = 100. * total_correct / total_samples
        test_acc = evaluate(model, test_loader, timesteps,  device)
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        logger.update_train(epoch, avg_loss, train_acc, output_spikes, model)
        logger.update_test(test_acc)
        logger.update_epoch_time(epoch_time)

        print(f"Epoch {epoch}: Train Acc = {train_acc:.2f}%, Test Acc = {test_acc:.2f}%, Loss = {avg_loss:.4f}, Spike Rate = {spike_rate:.4f}, Time: {epoch_time:.2f} sec")
