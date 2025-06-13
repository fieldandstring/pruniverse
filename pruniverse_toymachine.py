#!/usr/bin/env python3
import time, numpy as np, argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import csv
import sys

# ----- FAST BLOCK & FIELD IMPLEMENTATION -----
class Block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.active = True

    def forward(self, x):
        if not self.active:
            return torch.zeros(x.shape[0], self.linear.out_features, device=x.device)
        return self.linear(x)

    def freeze(self):
        self.active = False
        for param in self.linear.parameters():
            param.requires_grad = False

    def revive(self):
        self.active = True
        for param in self.linear.parameters():
            param.requires_grad = True

class FieldLayer(nn.Module):
    def __init__(self, num_blocks, in_dim, out_dim):
        super().__init__()
        self.blocks = nn.ModuleList([Block(in_dim, out_dim) for _ in range(num_blocks)])

    def forward(self, x):
        outs = [block(x) for block in self.blocks]
        return torch.cat(outs, dim=1)

class FieldNet(nn.Module):
    def __init__(self, in_dim, field_dim, out_dim, num_blocks):
        super().__init__()
        per_block = field_dim // num_blocks
        self.field = FieldLayer(num_blocks, in_dim, per_block)
        self.out = nn.Linear(field_dim, out_dim)

    def forward(self, x):
        x = self.field(x)
        return self.out(x)

def estimate_flops(net, batch_size):
    per_block = net.field.blocks[0].linear.in_features * net.field.blocks[0].linear.out_features
    block_flops = len(net.field.blocks) * per_block * 2
    out_flops = net.out.in_features * net.out.out_features * 2
    return (block_flops + out_flops) * batch_size

# --- ARGPARSE FOR FULL DIAL-ABILITY ---
def parse_args():
    P = argparse.ArgumentParser(description="Pruniverse Toymachine kernel demo")
    P.add_argument("--epochs", type=int, default=10)
    P.add_argument("--batch", type=int, default=64)
    P.add_argument("--field_dim", type=int, default=128)
    P.add_argument("--num_blocks", type=int, default=16)
    P.add_argument("--prune_every", type=int, default=50)
    P.add_argument("--freeze_threshold", type=float, default=30)
    P.add_argument("--revive_margin", type=float, default=2.0)
    P.add_argument("--logfile", type=str, default="pruniverse_log.csv")
    P.add_argument("--adam", action='store_true', help="Use Adam (no freezing)")
    P.add_argument("--freeze_n", type=int, default=1, help="Freeze N lowest-norm blocks each cycle")
    return P.parse_args()

def main():
    args = parse_args()
    in_dim = 28 * 28
    out_dim = 10
    field_dim = args.field_dim
    num_blocks = args.num_blocks
    prune_every = args.prune_every
    freeze_threshold = args.freeze_threshold
    revive_margin = args.revive_margin
    logfile = args.logfile
    batch_size = args.batch

    # --- DATA ---
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('.', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)

    # --- MODEL ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = FieldNet(in_dim, field_dim, out_dim, num_blocks).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    logger = open(logfile, "w", newline="")
    writer = csv.writer(logger)
    writer.writerow(["step", "loss", "frozen", "active", "effort", "entropy", "flops", "step_time"])
    step = 0
    start_time = last_time = time.time()
    prev_loss = None
    min_loss = 1e9
    last_revival = -999
    revival_cooldown = 200
    block_states = []
    global_start = time.time()

    for epoch in range(args.epochs):
        net.train()
        for data, target in train_loader:
            step += 1
            data = data.view(data.shape[0], -1).to(device)
            target = target.to(device)
            opt.zero_grad()
            output = net(data)
            loss = loss_fn(output, target)
            loss.backward()
            opt.step()
            # --- ADAM MODE ---
            if args.adam:
                frozen = 0
                active = len(net.field.blocks)
            else:
                # --- Prune: every N steps, freeze N lowest-norm blocks ---
                if step % prune_every == 0:
                    norms = [block(data).norm().item() if block.active else float('inf') for block in net.field.blocks]
                    to_freeze = np.argsort(norms)[:args.freeze_n]
                    for idx in to_freeze:
                        if net.field.blocks[idx].active and norms[idx] < freeze_threshold:
                            net.field.blocks[idx].freeze()
                            print(f"[{step}] Block {idx} frozen (norm={norms[idx]:.4f})")
                # --- Revival: sticky, rare, only after cooldown ---
                if prev_loss is not None:
                    min_loss = min(min_loss, loss.item())
                    if loss.item() > revive_margin * min_loss and (step - last_revival > revival_cooldown):
                        for block in net.field.blocks:
                            if not block.active:
                                block.revive()
                        last_revival = step
                        print(f"[{step}] Revival: Loss {loss.item():.4f} > {revive_margin}× min_loss {min_loss:.4f}")
                frozen = sum(not b.active for b in net.field.blocks)
                active = sum(b.active for b in net.field.blocks)
            if args.adam:
                block_states.append([1] * num_blocks)
            else:
                block_states.append([int(block.active) for block in net.field.blocks])
            prev_loss = loss.item()
            # --- Entropy, effort, flops ---
            probs = nn.functional.softmax(output, dim=1)
            entropy = -probs.mean().log().item()
            effort = sum(block(data).norm().item() for block in net.field.blocks)
            flops = estimate_flops(net, data.shape[0])
            now = time.time()
            writer.writerow([step, loss.item(), frozen, active, effort, entropy, flops, now - last_time])
            last_time = now
            if step % 200 == 0:
                print(f"[{step}] loss={loss.item():.4f} frozen={frozen} effort={effort:.2f} entropy={entropy:.3f} flops={flops}")
            if step > 1000:
                break
        else:
            continue
        break
    logger.close()
    print("Done. Log written to", logfile)
    
    wall_time_total = time.time() - global_start
    # --- FINAL SUMMARY ---

    # --- PLOT ---
    with open(logfile) as f:
        reader = csv.DictReader(f)
        steps, loss, frozen, active, effort, entropy, flops, stime = [], [], [], [], [], [], [], []
        for row in reader:
            steps.append(int(row["step"]))
            loss.append(float(row["loss"]))
            frozen.append(int(row["frozen"]))
            active.append(int(row["active"]))
            effort.append(float(row["effort"]))
            entropy.append(float(row["entropy"]))
            flops.append(float(row["flops"]))
            stime.append(float(row["step_time"]))
            
    print("\n--- Pruniverse Toymachine: Final Stats ---")
    print(f"Steps:       {steps[-1]}")
    print(f"Final Loss:  {loss[-1]:.4f}")
    print(f"Final Entropy: {entropy[-1]:.4f}")
    print(f"Frozen Blocks: {frozen[-1]} / {num_blocks}")
    print(f"Effort:       {effort[-1]:.2f}")
    print(f"FLOPS (final): {flops[-1]:.0f}")
    print(f"Total Wall Time: {wall_time_total:.2f} seconds")
    if any(f > 0 for f in frozen):
        print(f"Max Frozen Blocks: {max(frozen)}")
    else:
        print("No blocks were ever frozen (tune freeze_threshold or pruning logic!)")
    print("Log file:", logfile)
    
    import matplotlib as mpl
    mpl.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18})
    
    # Main plot (as before)
    fig, ax1 = plt.subplots(figsize=(14,8))
    ax1.plot(steps, loss, label="Loss", color="#1f77b4", linewidth=2)
    ax1.plot(steps, entropy, label="Entropy", color="#ff7f0e", linewidth=2, linestyle='--')
    ax1.set_ylabel("Loss / Entropy")
    ax1b = ax1.twinx()
    ax1b.plot(steps, effort, label="Effort", color="#2ca02c", linestyle=':', linewidth=2)
    ax1b.plot(steps, flops, label="FLOPS", color="#d62728", linestyle='-.', linewidth=2)
    ax1b.set_ylabel("Effort / FLOPS")
    ax1.grid(True, linestyle=':', linewidth=0.7)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1b.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=12)
    
    plt.title("Pruniverse Toymachine: Fieldblock Demo, Loss/Entropy/FLOPS", pad=18)
    plt.tight_layout(rect=[0,0.15,1,1])
    
    # Per-block frozen state heatmap
    block_states_arr = np.array(block_states).T  # shape: [num_blocks, steps]
    fig2, ax = plt.subplots(figsize=(14,3))
    ax.imshow(1-block_states_arr, aspect="auto", cmap="Purples", interpolation="nearest")
    ax.set_yticks(np.arange(0, num_blocks, 2))
    ax.set_yticklabels([f"Block {i}" for i in range(0, num_blocks, 2)], fontsize=8)
    ax.set_xlabel("Step")
    ax.set_title("Frozen Blocks Per Step (dark = frozen, light = active)")
    plt.tight_layout()
    
    plt.show()

if __name__ == "__main__":
    main()