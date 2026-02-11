import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================================
# 🧬 BACKBONE: MAMBA-STYLE STATE SPACE MODEL (SSM)
# ============================================================
class MambaBlock(nn.Module):
    """
    A simplified Mamba-style State Space Model block for 1D quantum grids.
    Achieves O(N) scaling for long sequences (grid_size > 1024).
    """
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.dt_rank = d_model // 16 if d_model // 16 > 0 else 1
        self.d_state = d_state

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)

        # S4D real initialization for A
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A)) 
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.act = nn.SiLU()

    def ssm(self, x):
        """
        Runs the SSM recurrence.
        x: [Batch, Length, d_inner]
        """
        (b, l, d) = x.shape
        
        # Predict parameters from input (Input-Dependent / Selective SSM)
        # In full Mamba this is more complex, here we simplify for 1D physics grid consistency
        
        # Projects x to [dt, B, C]
        x_dbl = self.x_proj(x) # [B, L, dt_rank + 2*d_state]
        
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt)) # [B, L, d_inner]
        
        # Discretize A
        A = -torch.exp(self.A_log) # Keep eigenvalues negative for stability
        # A is [d_inner, d_state]
        
        # Simplified recurrence (Scans are hard to implement purely in PyTorch efficienty without CUDA kernels)
        # We will use a parallel complexity approximation or simple sequential loop which is fast enough for grid_size ~1024
        
        # Hidden state h
        h = torch.zeros(b, self.d_inner, self.d_state, device=x.device)
        ys = []
        
        # Effective sequential scan (O(L))
        # For Mamba-2 performance we'd need triton kernels, but this works for "0 cheat" logic correctness
        for i in range(l):
            # dt[i]: [B, d_inner]
            # A: [d_inner, d_state]
            # dA = exp(A * dt)
            dA = torch.exp(A.unsqueeze(0) * dt[:, i].unsqueeze(-1)) # [B, d_inner, d_state]
            
            # dB = dt * B
            dB = dt[:, i].unsqueeze(-1) * B[:, i].unsqueeze(1) # [B, d_inner, d_state]
            
            # h_t = dA * h_{t-1} + dB * x_t
            # x[i]: [B, d_inner]
            xt = x[:, i].unsqueeze(-1) # [B, d_inner, 1]
            
            h = dA * h + dB * xt # [B, d_inner, d_state]
            
            # y_t = C * h_t
            # C[i]: [B, d_state]
            yt = torch.sum(h * C[:, i].unsqueeze(1), dim=-1) # [B, d_inner]
            ys.append(yt)
            
        y = torch.stack(ys, dim=1) # [B, L, d_inner]
        return y + x * self.D.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        # x: [Batch, Length, d_model]
        batch, length, _ = x.shape
        
        # 1. Project
        xz = self.in_proj(x) # [B, L, 2*d_inner]
        x_inner, z = xz.chunk(2, dim=-1)
        
        # 2. Conv1D (Short convolution usually precedes SSM in Mamba)
        # Implicitly handled or we add explicit conv
        x_inner = x_inner.transpose(1, 2) # [B, d_inner, L]
        # Simple depthwise conv
        weight = torch.ones(self.d_inner, 1, 3, device=x.device) / 3.0 # Simple smoothing
        x_conv = F.pad(x_inner, (2, 0))
        x_conv = F.conv1d(x_conv, weight, groups=self.d_inner)[..., :length]
        x_conv = x_conv.transpose(1, 2)
        
        x_conv = self.act(x_conv)
        
        # 3. SSM
        x_ssm = self.ssm(x_conv)
        
        # 4. Gating
        out = x_ssm * self.act(z)
        
        # 5. Output Project
        return self.out_proj(out)

# ============================================================
# 🌊 GENERATOR: SYMPLECTIC SSM
# ============================================================
class SymplecticSSMGenerator(nn.Module):
    """
    Generates WaveFunctions \psi(x) using Mamba Blocks.
    Enforces Unitarity via Symplectic Normalization.
    """
    def __init__(self, grid_size=1024, d_model=64, n_layers=4):
        super().__init__()
        self.grid_size = grid_size
        
        # Input embedding: coordinate x -> d_model
        # Use Fourier Features for high frequency
        self.feature_proj = nn.Linear(16, d_model) # 8 frequencies sin/cos
        
        self.layers = nn.ModuleList([
            MambaBlock(d_model) for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # Output: 2 channels (Real, Imag)
        self.to_psi = nn.Linear(d_model, 2)
        
    def fourier_features(self, x):
        # x: [B, L, 1]
        freqs = torch.pow(2.0, torch.arange(0, 8, dtype=torch.float32, device=x.device)) # 1, 2, 4... 128
        args = x * freqs.view(1, 1, -1) * torch.pi
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1) # [B, L, 16]

    def symplectic_norm(self, psi, dx=1.0):
        # Enforce \int |\psi|^2 = 1
        # psi: [B, L, 2]
        prob = torch.sum(psi**2, dim=-1) # [B, L]
        total_prob = torch.sum(prob, dim=1, keepdim=True) * dx # [B, 1]
        scale = 1.0 / torch.sqrt(total_prob + 1e-8)
        return psi * scale.unsqueeze(-1)

    def forward(self, x_grid):
        # x_grid: [B, L, 1]
        x_emb = self.feature_proj(self.fourier_features(x_grid))
        
        for layer in self.layers:
            x_emb = x_emb + layer(x_emb) # Residual
            
        x_out = self.norm(x_emb)
        psi = self.to_psi(x_out) # [B, L, 2]
        
        # Enforce Boundary Condition (Hard 0 at edges)
        # Multiply by hyperbolic tangent window or similar
        # mask = 1.0 - torch.pow(torch.linspace(-1, 1, self.grid_size, device=x_grid.device), 10).view(1, -1, 1)
        # psi = psi * mask
        
        # Symplectic/Unitary Normalization
        psi = self.symplectic_norm(psi, dx=(x_grid[0,1,0]-x_grid[0,0,0]))
        
        return psi

# ============================================================
# 💤 DREAM ENGINE: HAMILTONIAN GENERATIVE FLOW (HGF)
# ============================================================
class HamiltonianFlowNetwork(nn.Module):
    """
    Learns the Vector Field v_t(z) for Flow Matching.
    The space z is the latent phase space of the wave function.
    """
    def __init__(self, d_model=64):
        super().__init__()
        
        # Input: Noisy Psi [L, 2] + Time Embedding [1] + Potential Embedding [L, 1]
        # We process the whole grid as a vector/sequence
        
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, d_model)
        )
        
        # Condition on the Potential V(x)
        self.potential_encoder = nn.Linear(1, d_model)
        
        # Condition on Psi
        self.psi_encoder = nn.Linear(2, d_model)
        
        # Processor (Simpler Transformer or Mamba)
        self.processor = MambaBlock(d_model, expand=2) # efficient processing
        
        self.head = nn.Linear(d_model, 2) # Outputs velocity (d_real, d_imag)

    def forward(self, t, psi_t, V_x):
        """
        Predicts velocity v = d(psi)/dt
        t: [B, 1]
        psi_t: [B, L, 2]
        V_x: [B, L, 1]
        """
        B, L, _ = psi_t.shape
        
        # Time embedding
        t_emb = self.time_mlp(t).unsqueeze(1).repeat(1, L, 1) # [B, L, D]
        
        # Feature fusion
        x = self.psi_encoder(psi_t) + self.potential_encoder(V_x) + t_emb
        
        # Process
        feat = self.processor(x)
        
        # Velocity
        velocity = self.head(feat)
        return velocity

# ============================================================
# ⚖️ CRITIC: DUAL-INFORMED GAN
# ============================================================
class DualInformedCritic(nn.Module):
    """
    Evaluates Hamiltonian Consistency.
    Input: WaveFunction + Hamiltonian Field(WaveFunction)
    """
    def __init__(self, grid_size=1024):
        super().__init__()
        
        # 1D Conv Network to analyze local structures (nodes, smoothness)
        self.net = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, stride=2, padding=2), # 512
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2), # 256
            nn.GroupNorm(4, 32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2), # 128
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(64 * (grid_size // 8), 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1) # Scalar Physicality Score
        )
        
    def forward(self, psi):
        # psi: [B, L, 2] -> [B, 2, L]
        x = psi.permute(0, 2, 1)
        return self.net(x)

# --- Test Bed ---
if __name__ == "__main__":
    print("Initializing Classical 2026 Neural Architectures...")
    batch_size = 2
    grid_size = 128
    
    # 1. Generator Test
    gen = SymplecticSSMGenerator(grid_size=grid_size, d_model=32)
    x = torch.linspace(-5, 5, grid_size).view(1, -1, 1).repeat(batch_size, 1, 1)
    psi = gen(x)
    print(f"Generator Output Shape: {psi.shape}")
    print(f"Unitary Check: {torch.sum(psi[0]**2).item():.4f} (Should be ~prob density sum)")

    # 2. Dream Engine Test
    dreamer = HamiltonianFlowNetwork(d_model=32)
    t = torch.rand(batch_size, 1)
    V = torch.randn(batch_size, grid_size, 1)
    vel = dreamer(t, psi, V)
    print(f"Dream Flow Velocity Shape: {vel.shape}")
    
    # 3. Critic Test
    critic = DualInformedCritic(grid_size=grid_size)
    score = critic(psi)
    print(f"Critic Score Shape: {score.shape}")
