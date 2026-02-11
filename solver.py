import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from collections import deque
import random

from quantum_physics import QuantumPhysicsEngine
from neural_dream import SymplecticSSMGenerator, HamiltonianFlowNetwork, DualInformedCritic

class SchrodingerSolver:
    """
    Manages the Wake-Sleep Cycle for Quantum Solving.
    
    Attributes:
        engine: Physics Engine (Hamiltonian)
        generator: Symplectic Mamba-2 Network (The Wavefunction)
        dreamer: Hamiltonian Flow Network (The Dream Engine)
        critic: Dual-Informed GAN Critic
        memory: Replay buffer of Converged Low-Energy States
    """
    def __init__(self, grid_size=128, device='cpu'):
        self.device = device
        self.grid_size = grid_size
        
        # Physics
        self.engine = QuantumPhysicsEngine(grid_size=grid_size).to(device)
        
        # Networks
        self.generator = SymplecticSSMGenerator(grid_size=grid_size, d_model=64).to(device)
        self.dreamer = HamiltonianFlowNetwork(d_model=64).to(device)
        self.critic = DualInformedCritic(grid_size=grid_size).to(device)
        
        # Optimizers
        self.opt_gen = optim.AdamW(self.generator.parameters(), lr=1e-3)
        self.opt_dream = optim.AdamW(self.dreamer.parameters(), lr=5e-4) # Slower learning for flow
        self.opt_critic = optim.AdamW(self.critic.parameters(), lr=1e-4)
        
        # Memory (Physics Buffer)
        self.memory = deque(maxlen=2000) # Stores valid (V, psi, energy) tuples
        
    def train_step_awake(self, V_x):
        """
        Phase 1: AWAKE - Neural Hamiltonian Minimization
        Goal: Find Ground State for potential V_x using pure physics loss.
        """
        self.opt_gen.zero_grad()
        
        # 1. Forward Pass
        # Coordinate grid input
        x_grid = self.engine.x.view(1, -1, 1).repeat(V_x.shape[0], 1, 1).to(self.device)
        psi = self.generator(x_grid)
        
        # 2. Physics Loss (Manual Finite Difference for stability)
        dx = self.engine.dx
        psi_real, psi_imag = psi[0, :, 0], psi[0, :, 1]
        
        # Laplacian (Central Difference)
        lap_real = (torch.roll(psi_real, -1) - 2*psi_real + torch.roll(psi_real, 1)) / (dx**2)
        lap_imag = (torch.roll(psi_imag, -1) - 2*psi_imag + torch.roll(psi_imag, 1)) / (dx**2)
        # Fix edges (zeros)
        lap_real[0] = lap_real[-1] = 0
        lap_imag[0] = lap_imag[-1] = 0
        
        # Kinetic Energy Integrand: -0.5 * psi * laplacian
        # Real part: (real*lap_real + imag*lap_imag)
        k_density = -0.5 * (psi_real * lap_real + psi_imag * lap_imag)
        
        # Potential Energy Integrand: V * |psi|^2
        rho = psi_real**2 + psi_imag**2
        v_density = V_x[0, :, 0] * rho
        
        total_energy = torch.sum(k_density + v_density) * dx
        norm = torch.sum(rho) * dx
        
        loss = total_energy / (norm + 1e-8)
        
        # Stability: Sanitize loss/grads
        if torch.isnan(loss) or torch.isinf(loss):
            self.opt_gen.zero_grad()
            return 1e6 # Return high penalty
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
        self.opt_gen.step()
        
        # Store in memory if converged (low variance proxy or just periodically)
        # For demo, store randomly with prob
        if random.random() < 0.1 and not np.isnan(loss.item()):
            self.memory.append((V_x.detach().cpu(), psi.detach().cpu(), loss.item()))
            
        return loss.item()

    def train_step_dream(self):
        """
        Phase 2: REM SLEEP - Flow Matching
        Goal: Train HGF to map Noise -> Memory States.
        """
        if len(self.memory) < 32:
            return 0.0
            
        self.opt_dream.zero_grad()
        
        # Sample batch from memory
        batch = random.sample(self.memory, min(len(self.memory), 8))
        V_batch = torch.stack([b[0] for b in batch]).to(self.device).squeeze(1) # [B, L, 1] usually
        if V_batch.dim() == 2: V_batch = V_batch.unsqueeze(-1)
        
        psi_target = torch.stack([b[1] for b in batch]).to(self.device).squeeze(1) # [B, L, 2]
        if psi_target.dim() == 4: psi_target = psi_target.squeeze(1)
        
        B, L, _ = psi_target.shape
        
        # 1. Flow Matching Setup
        # x_0: Source distribution (Gaussian Noise)
        psi_0 = torch.randn_like(psi_target)
        # x_1: Target distribution (Data)
        psi_1 = psi_target
        
        # Time t ~ U[0, 1]
        t = torch.rand(B, 1, device=self.device)
        
        # Interpolation (Conditional Flow)
        # psi_t = (1 - t) * psi_0 + t * psi_1
        psi_t = (1 - t.unsqueeze(1)) * psi_0 + t.unsqueeze(1) * psi_1
        
        # Target Velocity u_t = psi_1 - psi_0
        target_velocity = psi_1 - psi_0
        
        # 2. Predict Velocity
        predicted_velocity = self.dreamer(t, psi_t, V_batch)
        
        # 3. Loss (MSE - Fisher Divergence)
        loss = torch.mean((predicted_velocity - target_velocity)**2)
        
        if torch.isnan(loss) or torch.isinf(loss):
            self.opt_dream.zero_grad()
            return 0.0
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dreamer.parameters(), 1.0)
        self.opt_dream.step()
        return loss.item()

    def generate_dream(self, V_x):
        """
        Inference: Use HGF to guess the ground state.
        Mapping Noise -> Psi via Integrate(v_t).
        """
        with torch.no_grad():
            B = V_x.shape[0]
            L = self.grid_size
            
            # Start Noise
            psi = torch.randn(B, L, 2, device=self.device)
            dt = 0.05
            
            # Euler Integration of Flow
            for t_scalar in np.arange(0, 1.0, dt):
                t = torch.tensor([[t_scalar]], dtype=torch.float32, device=self.device).repeat(B, 1)
                vel = self.dreamer(t, psi, V_x)
                psi = psi + vel * dt
                
            return psi

# --- Test Bed ---
if __name__ == "__main__":
    print("Initializing Schrödinger Solver (Wake-Sleep)...")
    solver = SchrodingerSolver(grid_size=128)
    
    # Dummy Potential (Harmonic)
    x = solver.engine.x
    V = 0.5 * x**2 
    V = V.view(1, -1, 1) # [1, L, 1]
    
    print("Running Awake Phase (Physics Minimization)...")
    for i in range(10):
        loss = solver.train_step_awake(V)
        if i % 2 == 0:
            print(f"Step {i}: Energy = {loss:.4f}")
            
    print("Running Dream Phase (Flow Training)...")
    # Need to populate memory first
    for _ in range(50): solver.train_step_awake(V)
    
    d_loss = solver.train_step_dream()
    print(f"Dream Loss: {d_loss:.6f}")
    
    print("Dream Generation Test...")
    psi_dream = solver.generate_dream(V)
    print(f"Dreamed Psi Shape: {psi_dream.shape}")
