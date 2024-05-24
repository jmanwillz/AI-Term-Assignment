import torch
import torch.nn as nn
import torch.optim as optim


class FFNN:
    def __init__(self):
        self.is_trained = False
        self.model = nn.Sequential(nn.Linear(16, 20), nn.ReLU(), nn.Linear(20, 2))
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def negative_log_likelihood(self, outputs, targets):
        mean = outputs[:, 0]
        log_variance = outputs[:, 1]
        variance = torch.exp(log_variance)
        nll = 0.5 * torch.mean(log_variance + ((targets - mean) ** 2) / variance)
        return nll

    def train(self, x, y, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.negative_log_likelihood(outputs, y)
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        self.is_trained = True

    def test(self, x_test):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x_test)
            mean = outputs[:, 0]
            log_variance = outputs[:, 1]
            variance = torch.exp(log_variance)
        return mean, variance
