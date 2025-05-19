import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PrototypeMapper(nn.Module):
    def __init__(self, num_prototypes, feature_dim, gamma=1000):
        """
        Initialize the PrototypeMapper module.

        Args:
            num_prototypes (int): Number of prototypes, typically equal to number of classes.
            feature_dim (int): Dimensionality of input features.
            gamma (float): Temperature parameter controlling similarity sharpness (default: 1000).
        """
        super(PrototypeMapper, self).__init__()
        self.num_prototypes = num_prototypes
        self.feature_dim = feature_dim
        self.gamma = gamma

        # Learnable prototype vectors, shape: (num_prototypes, feature_dim)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, feature_dim))

        # A simple linear mapper G_p: maps input features to prototype space
        self.mapper = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        """
        Forward pass: maps input features to prototype space.

        Args:
            x (Tensor): Input features of shape (batch_size, feature_dim)

        Returns:
            Tensor: Mapped features of shape (batch_size, feature_dim)
        """
        return self.mapper(x)

    def get_prototype_loss(self, x, y):
        """
        Compute prototype loss based on similarity between mapped features and prototypes.

        Args:
            x (Tensor): Input features, shape (batch_size, feature_dim)
            y (Tensor): Class labels, shape (batch_size, 1), each entry in [0, num_prototypes-1]

        Returns:
            Tensor: Scalar loss value
        """
        batch_size = x.size(0)
        mapped_x = self.mapper(x)  # (batch_size, feature_dim)

        # Compute squared Euclidean distance to each prototype
        distances = torch.norm(
            mapped_x.unsqueeze(1) - self.prototypes.unsqueeze(0),
            dim=2
        ) ** 2  # Shape: (batch_size, num_prototypes)

        # Compute similarity: o(x_i, p_k) = exp(-||G_p(x_i) - p_k||^2 / gamma)
        similarity = torch.exp(-distances / self.gamma)

        # One-hot encode labels: shape (batch_size, num_prototypes)
        T = torch.zeros(batch_size, self.num_prototypes, device=x.device)
        T.scatter_(1, y, 1)

        # Compute binary cross-entropy-like loss
        eps = 1e-8
        loss = -T * torch.log(similarity + eps) - (1 - T) * torch.log(1 - similarity + eps)

        return loss.sum()

    def print_gradient(self):
        """
        Print gradient statistics (min, max, mean, norm) for all trainable parameters.
        """
        print("Gradient Information:")
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad = param.grad
                print(f"  Parameter '{name}':")
                print(f"    Min:  {grad.min().item():.6f}")
                print(f"    Max:  {grad.max().item():.6f}")
                print(f"    Mean: {grad.mean().item():.6f}")
                print(f"    Norm: {grad.norm().item():.6f}")
            else:
                print(f"  Parameter '{name}' has no gradient.")


def test_gradient():
    """
    Test whether PrototypeMapper's loss function properly supports backpropagation.
    Also print gradient statistics to observe behavior.
    """
    torch.manual_seed(42)

    num_prototypes = 2
    feature_dim = 128
    gamma = 1000

    model = PrototypeMapper(num_prototypes, feature_dim, gamma)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    batch_size = 32
    x = torch.randn(batch_size, feature_dim, device=device)
    y = torch.randint(0, num_prototypes, (batch_size, 1), device=device)

    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        loss = model.get_prototype_loss(x, y)

        print(f"\nEpoch {epoch}: Prototype Loss = {loss.item():.6f}")
        loss.backward()

        print(f"Epoch {epoch}: Gradient Stats")
        model.print_gradient()

        optimizer.step()

        # Optionally monitor prototype updates
        with torch.no_grad():
            print(f"Epoch {epoch}: First dimension of prototypes = {model.prototypes[:, 0]}")


if __name__ == "__main__":
    print("\nStarting gradient test...")
    test_gradient()
