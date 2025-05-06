# This architecture is used for plotting the SE not for running accuracy test
# To save Tau and the visualization, ti cause some small subtle difference in the forward step

import argparse
import collections
import time
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter_add, scatter_mean, scatter_max


def compute_state_evolution(eta_fn, tau0, sigma2=0.0, delta=1.0, T=3, n_samples=20000):
    """
    Monte Carlo state-evolution: returns [tau2_0, tau2_1, ..., tau2_T]
    """
    dist = torch.distributions.Laplace(0.0, 1.0)
    X = dist.sample((n_samples,)).to(tau0.device)
    taus = [tau0]
    for t in range(T):
        Z = torch.randn_like(X)
        sigma = math.sqrt(taus[-1])
        Y = X + sigma * Z
        X_hat = eta_fn(Y)
        mse = ((X_hat - X) ** 2).mean().item()
        taus.append(sigma2 + mse / delta)
    return taus


class AMPConv(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        damping=0.5,
        denoiser_hidden=None,
        memory=0,
        use_skip=False,
        lamp=False,
        agg="add",
    ):
        super().__init__(aggr=agg if agg != "mix" else "add")
        self.lin = Linear(in_channels, out_channels)
        self.damping = damping
        self.memory = memory
        self.use_skip = use_skip
        self.lamp = lamp
        self.agg_type = agg
        self.history = collections.deque(maxlen=memory)
        if lamp:
            self.alpha = nn.Parameter(torch.tensor(1.0))
            self.theta = nn.Parameter(torch.tensor(1.0))
        if denoiser_hidden:
            self.denoiser = Sequential(
                Linear(out_channels, denoiser_hidden),
                ReLU(),
                Linear(denoiser_hidden, out_channels),
            )
        else:
            self.denoiser = ReLU()
        self.register_buffer("onsager", torch.tensor(0.0))

    def apply_onsager_and_denoise(self, r, x_lin):
        x0 = x_lin.clone()
        # Onsager correction
        if self.lamp:
            out = r - (self.alpha * self.onsager) * x0
        else:
            out = r - self.onsager * x0
        # Damping
        if not hasattr(self, "prev_out") or self.prev_out is None:
            out_d = out
        else:
            out_d = self.damping * out + (1 - self.damping) * self.prev_out
        self.prev_out = out_d.detach()
        # Memory AMP
        if self.memory > 0 and len(self.history) > 0:
            weights = torch.linspace(1.0, 0.5, len(self.history), device=out.device)
            mem = sum(w * h for w, h in zip(weights, self.history))
            out_d = out_d + mem
        self.history.append(out_d.detach())
        # Denoising
        if self.lamp:
            out_dn = torch.sign(out_d) * F.relu(out_d.abs() - self.theta)
        else:
            out_dn = self.denoiser(out_d)
        # Skip connection
        if self.use_skip:
            out_dn = out_dn + x0
        # Update Onsager via Hutchinson
        with torch.no_grad():
            eps = torch.randn_like(out_d)
            pert = (
                self.denoiser(out_d + eps)
                if not self.lamp
                else torch.sign(out_d + eps) * F.relu((out_d + eps).abs() - self.theta)
            )
            base = (
                self.denoiser(out_d)
                if not self.lamp
                else torch.sign(out_d) * F.relu(out_d.abs() - self.theta)
            )
            div = (eps * (pert - base)).sum() / (eps * eps).sum()
            self.onsager = torch.clamp(div, -1.0, 1.0)
        return out_dn

    def forward(self, x, edge_index):
        # Linear projection
        x_lin = self.lin(x)
        # Normalize & aggregate
        edge_index, _ = add_self_loops(edge_index, num_nodes=x_lin.size(0))
        row, col = edge_index
        deg = degree(row, x_lin.size(0), dtype=x_lin.dtype)
        norm = deg.pow(-0.5)[row] * deg.pow(-0.5)[col]
        if self.agg_type == "mix":
            vals = x_lin[col] * norm.view(-1, 1)
            m_sum = scatter_add(vals, row, dim=0, dim_size=x_lin.size(0))
            m_mean = scatter_mean(vals, row, dim=0, dim_size=x_lin.size(0))
            m_max = scatter_max(vals, row, dim=0, dim_size=x_lin.size(0))[0]
            r = (0.4 * m_sum + 0.3 * m_mean + 0.3 * m_max) / (0.4 + 0.3 + 0.3)
        else:
            r = self.propagate(edge_index, x=x_lin, norm=norm)
        # Apply Onsager, damping, memory, denoiser, skip
        return self.apply_onsager_and_denoise(r, x_lin)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class VAMPBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.linear = AMPConv(in_channels, out_channels, damping=1.0, lamp=False)
        self.denoiser = Sequential(
            Linear(out_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, out_channels),
        )

    def forward(self, x, edge_index):
        z = self.linear(x, edge_index)
        return self.denoiser(z)


class GCNNet(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1), []


class AMPNet(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, damping, use_skip, lamp, agg, memory):
        super().__init__()
        # First AMPConv: in_dim → hid_dim
        self.conv1 = AMPConv(
            in_dim,
            hid_dim,
            damping=damping,
            denoiser_hidden=hid_dim,
            memory=memory,
            use_skip=use_skip,
            lamp=lamp,
            agg=agg,
        )
        # Second AMPConv: hid_dim → out_dim (num classes)
        self.conv2 = AMPConv(
            hid_dim,
            out_dim,
            damping=damping,
            denoiser_hidden=None,
            memory=memory,
            use_skip=use_skip,
            lamp=lamp,
            agg=agg,
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        taus_emp = []

        # Layer 1
        x_lin1 = self.conv1.lin(x)
        ei1, _ = add_self_loops(edge_index, num_nodes=x_lin1.size(0))
        row1, col1 = ei1
        deg1 = degree(row1, x_lin1.size(0), dtype=x_lin1.dtype)
        norm1 = deg1.pow(-0.5)[row1] * deg1.pow(-0.5)[col1]
        r1 = self.conv1.propagate(ei1, x=x_lin1, norm=norm1)
        taus_emp.append((r1**2).mean())
        out1 = self.conv1.apply_onsager_and_denoise(r1, x_lin1)
        h1 = F.relu(out1)
        h1 = F.dropout(h1, p=0.5, training=self.training)

        # Layer 2
        x_lin2 = self.conv2.lin(h1)
        ei2, _ = add_self_loops(edge_index, num_nodes=x_lin2.size(0))
        row2, col2 = ei2
        deg2 = degree(row2, x_lin2.size(0), dtype=x_lin2.dtype)
        norm2 = deg2.pow(-0.5)[row2] * deg2.pow(-0.5)[col2]
        r2 = self.conv2.propagate(ei2, x=x_lin2, norm=norm2)
        taus_emp.append((r2**2).mean())
        out2 = self.conv2.apply_onsager_and_denoise(r2, x_lin2)

        # Use the AMPConv output as logits (out2 has shape [N, num_classes])
        logits = out2
        return F.log_softmax(logits, dim=-1), taus_emp


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out, _ = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, data):
    model.eval()
    out, _ = model(data)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        pred = out[mask].max(1)[1]
        accs.append(pred.eq(data.y[mask]).sum().item() / mask.sum().item())
    return accs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_skip", action="store_true")
    parser.add_argument("--lamp", action="store_true")
    parser.add_argument("--agg", choices=["add", "mean", "max", "mix"], default="add")
    parser.add_argument("--memory", type=int, default=1)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--model", choices=["gcn", "amp", "vamp"], default="gcn")
    parser.add_argument("--dataset", default="Cora")
    parser.add_argument("--hid", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = Planetoid(root="data", name=args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = dataset[0].to(device)

    if args.model == "gcn":
        model = GCNNet(dataset.num_features, args.hid, dataset.num_classes).to(device)
    elif args.model == "amp":
        model = AMPNet(
            dataset.num_features,
            args.hid,
            dataset.num_classes,
            args.damping,
            args.use_skip,
            args.lamp,
            args.agg,
            args.memory,
        ).to(device)
    else:
        # simple VAMP baseline
        class VAMPNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.b1 = VAMPBlock(dataset.num_features, args.hid, args.hid)
                self.b2 = VAMPBlock(args.hid, args.hid, dataset.num_classes)

            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                x = F.relu(self.b1(x, edge_index))
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.b2(x, edge_index)
                return F.log_softmax(x, dim=-1), []

        model = VAMPNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val = 0.0
    test_acc = 0.0
    all_taus = []

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        loss = train(model, data, optimizer)
        train_acc, val_acc, tmp_test_acc = test(model, data)
        if val_acc > best_val:
            best_val = val_acc
            test_acc = tmp_test_acc
        if epoch % 50 == 0:
            print(
                f"Epoch {epoch:03d} | Loss {loss:.4f} | "
                f"Train {train_acc:.4f} | Val {val_acc:.4f} | "
                f"Test {tmp_test_acc:.4f} | {(time.time()-start):.1f}s"
            )

    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print(f"Total time: {(time.time()-start):.1f}s")

    # Only for AMP model, collect residual variances and plot variance alignment
    if args.model == "amp":
        # --- collect empirical taus at final epoch ---
        model.eval()
        with torch.no_grad():
            _, taus_emp = model(data)
            print("Final empirical τ² per layer:", [t.item() for t in taus_emp])
            all_taus.append([t.item() for t in taus_emp])

        mean_taus_emp = torch.tensor(all_taus).mean(dim=0)

        # --- compute state-evolution curve ---
        def eta_fn(u):
            theta = (
                model.conv1.theta if args.lamp else torch.tensor(0.0, device=u.device)
            )
            theta = theta.to(u.device)
            return torch.sign(u) * F.relu(u.abs() - theta)

        taus_se = compute_state_evolution(
            eta_fn, mean_taus_emp[0], sigma2=0.0, delta=1.0, T=len(mean_taus_emp)
        )

        # --- plot alignment ---
        layers = list(range(1, len(mean_taus_emp) + 1))
        plt.plot(layers, mean_taus_emp.tolist(), "o-", label="Empirical τ²")
        plt.plot(layers, taus_se[1:], "x--", label="State-Evolution τ²")
        plt.xlabel("Layer")
        plt.ylabel(r"$\tau^2$")
        plt.title("Variance Alignment")
        plt.legend()
        plt.savefig("variance_alignment.png")


if __name__ == "__main__":
    main()
