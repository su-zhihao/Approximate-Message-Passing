# amp_gnn_experiment.py

import argparse
import collections
import time

import random, numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter_add, scatter_mean, scatter_max


class AMPConv(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        damping: float = 0.5,
        denoiser_hidden: int = None,
        memory: int = 0,
        use_skip: bool = False,
        lamp: bool = False,
        agg: str = "add",
    ):
        # For 'mix', we do manual scatter; otherwise rely on MessagePassing aggr
        super().__init__(aggr=agg if agg != "mix" else "add")
        self.lin = Linear(in_channels, out_channels)
        self.damping = damping
        self.memory = memory
        self.use_skip = use_skip
        self.lamp = lamp
        self.agg_type = agg

        # History buffer for MAMP
        self.history = collections.deque(maxlen=memory)

        # Optional learned AMP parameters
        if lamp:
            self.alpha = nn.Parameter(torch.tensor(1.0))
            self.theta = nn.Parameter(torch.tensor(1.0))

        # Denoiser: either MLP or ReLU
        if denoiser_hidden:
            self.denoiser = Sequential(
                Linear(out_channels, denoiser_hidden),
                ReLU(),
                Linear(denoiser_hidden, out_channels),
            )
        else:
            self.denoiser = ReLU()

        # Onsager term
        self.register_buffer("onsager", torch.tensor(0.0))

    def forward(self, x, edge_index):
        # Linear projection
        x_lin = self.lin(x)
        # Preserve for skip or Onsager
        x0 = x_lin.clone()

        # Add self-loops and compute symmetric normalization
        edge_index, _ = add_self_loops(edge_index, num_nodes=x_lin.size(0))
        row, col = edge_index
        deg = degree(row, x_lin.size(0), dtype=x_lin.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Aggregation
        if self.agg_type == "mix":
            # Compute sum / mean / max manually
            vals = x_lin[col] * norm.view(-1, 1)
            m_sum = scatter_add(vals, row, dim=0, dim_size=x_lin.size(0))
            m_mean = scatter_mean(vals, row, dim=0, dim_size=x_lin.size(0))
            m_max = scatter_max(vals, row, dim=0, dim_size=x_lin.size(0))[0]
            # Fixed mixture weights; could be learned
            m = 0.4 * m_sum + 0.3 * m_mean + 0.3 * m_max
        else:
            # Use PyG propagate
            m = self.propagate(edge_index, x=x_lin, norm=norm)

        # Onsager correction (with optional learned step-size)
        if self.lamp:
            out = m - (self.alpha * self.onsager) * x0
        else:
            out = m - self.onsager * x0

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

        # Denoising step
        if self.lamp:
            # Soft-threshold denoiser (LAMP)
            out_dn = torch.sign(out_d) * F.relu(out_d.abs() - self.theta)
        else:
            out_dn = self.denoiser(out_d)

        # Optional skip connection
        if self.use_skip:
            out_dn = out_dn + x0

        # Update Onsager via Hutchinson estimator
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
            # Clamp the Onsager term to keep it from exploding:
            self.onsager = torch.clamp(div, -1.0, 1.0)

        return out_dn

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class VAMPBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # Linear AMP step (no damping, lamp off)
        self.linear = AMPConv(in_channels, out_channels, damping=1.0, lamp=False)
        # Nonlinear denoiser
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
        return F.log_softmax(x, dim=-1)


class AMPNet(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, damping, use_skip, lamp, agg, memory):
        super().__init__()
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
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(data)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        pred = out[mask].max(1)[1]
        accs.append(pred.eq(data.y[mask]).sum().item() / mask.sum().item())
    return accs


def main():
    parser = argparse.ArgumentParser()
    # New flexible flags
    parser.add_argument(
        "--use_skip", action="store_true", help="add residual skip connection"
    )
    parser.add_argument(
        "--lamp", action="store_true", help="use learned AMP (LAMP) soft-threshold"
    )
    parser.add_argument(
        "--agg",
        choices=["add", "mean", "max", "mix"],
        default="add",
        help="aggregation type for AMPConv",
    )
    parser.add_argument(
        "--memory", type=int, default=1, help="history length for MAMPConv"
    )
    parser.add_argument("--damping", type=float, default=0.5, help="damping factor λ")
    # Core parameters
    parser.add_argument("--model", choices=["gcn", "amp", "vamp"], default="gcn")
    parser.add_argument("--dataset", default="Cora")
    parser.add_argument("--hid", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", "--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load dataset
    dataset = Planetoid(root="data", name=args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = dataset[0].to(device)

    # Select model
    if args.model == "gcn":
        model = GCNNet(dataset.num_features, args.hid, dataset.num_classes).to(device)

    elif args.model == "amp":
        model = AMPNet(
            in_dim=dataset.num_features,
            hid_dim=args.hid,
            out_dim=dataset.num_classes,
            damping=args.damping,
            use_skip=args.use_skip,
            lamp=args.lamp,
            agg=args.agg,
            memory=args.memory,
        ).to(device)

    else:  # vamp

        class VAMPNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.block1 = VAMPBlock(dataset.num_features, args.hid, args.hid)
                self.block2 = VAMPBlock(args.hid, args.hid, dataset.num_classes)

            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                x = F.relu(self.block1(x, edge_index))
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.block2(x, edge_index)
                return F.log_softmax(x, dim=-1)

        model = VAMPNet().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
    )

    best_val = 0.0
    test_acc = 0.0
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        loss = train(model, data, optimizer)
        train_acc, val_acc, tmp_test_acc = test(model, data)
        if val_acc > best_val:
            best_val = val_acc
            test_acc = tmp_test_acc
        if epoch % 50 == 0:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:03d} | Loss {loss:.4f} | "
                f"Train {train_acc:.4f} | Val {val_acc:.4f} | "
                f"Test {tmp_test_acc:.4f} | {elapsed:.1f}s elapsed"
            )

    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print(f"Total time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
