import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn


class WUNN:
    def __init__(
        self, beta: float, gamma: float, mu_0: float, sigma_0: float, update_beta: bool
    ):
        self.is_trained = False
        self.model = nn.Sequential(
            bnn.BayesLinear(
                prior_mu=mu_0, prior_sigma=sigma_0, in_features=16, out_features=20
            ),
            nn.ReLU(),
            bnn.BayesLinear(
                prior_mu=mu_0, prior_sigma=sigma_0, in_features=20, out_features=1
            ),
        )

        self.mse_loss = nn.MSELoss()
        self.kl_loss = bnn.BKLLoss(reduction="mean", last_layer_only=False)
        self.beta = beta
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.update_beta: bool = update_beta
        self.gamma: float = gamma

    def train(self, x, y, num_epochs):
        if self.update_beta:
            self.beta = self.gamma * self.beta

        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            outputs = self.model(x).squeeze()

            mse = self.mse_loss(outputs, y)
            kl_loss = self.kl_loss(self.model)
            loss = mse + self.beta * kl_loss

            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        self.is_trained = True

    def test(self, x_test):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(x_test)
        return predictions
