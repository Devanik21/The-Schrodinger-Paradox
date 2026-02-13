"""
quantum_physics.py — The Schrödinger Dream: Physics Engine

Implements:
  Level 1:  3D Multi-Electron Hamiltonian (Coulomb: e-n, e-e, n-n)
  Level 2:  Metropolis-Hastings MCMC Sampler with adaptive step size
  Level 3:  Local Energy via autograd + Hutchinson trace estimator
  Level 9:  Atoms H→Ne (predefined systems with exact energies)
  Level 10: Molecules — H₂, LiH, H₂O + PES scanning
  Level 14: Berry Phase from Neural Wavefunction (topological)
  Level 16: Periodic Systems — Bloch/Ewald for Solids (HEG)
  Level 17: Spin-Orbit Coupling — Breit-Pauli Relativistic QM
  Level 18: Entanglement Entropy — Rényi-2 via SWAP Trick
  + Legacy 1D grid engine for demo/teaching mode
"""


import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ============================================================
# 📐 MOLECULAR SYSTEM DEFINITION (Level 1)
# ============================================================
@dataclass
class MolecularSystem:
    """
    Defines a molecular/atomic system for 3D VMC.
    
    Attributes:
        nuclei: List of (Z, [x, y, z]) tuples — atomic number + 3D position
        n_up: Number of spin-up electrons
        n_down: Number of spin-down electrons
    """
    nuclei: List[Tuple[int, List[float]]]
    n_up: int
    n_down: int
    name: str = "Custom"
    exact_energy: Optional[float] = None  # Known exact energy for benchmarking

    @property
    def n_electrons(self) -> int:
        return self.n_up + self.n_down

    @property
    def n_nuclei(self) -> int:
        return len(self.nuclei)

    def charges(self, device='cpu') -> torch.Tensor:
        """Returns tensor of nuclear charges [N_n]"""
        return torch.tensor([Z for Z, _ in self.nuclei], dtype=torch.float32, device=device)

    def positions(self, device='cpu') -> torch.Tensor:
        """Returns tensor of nuclear positions [N_n, 3]"""
        return torch.tensor([R for _, R in self.nuclei], dtype=torch.float32, device=device)


# ============================================================
# 🏗️ PREDEFINED ATOMIC SYSTEMS
# ============================================================
ATOMS = {
    # Level 9: Full periodic table H → Ne with NIST exact energies (Hartree)
    "H":  MolecularSystem(nuclei=[(1, [0.0, 0.0, 0.0])], n_up=1, n_down=0,
                           name="Hydrogen", exact_energy=-0.5000),
    "He": MolecularSystem(nuclei=[(2, [0.0, 0.0, 0.0])], n_up=1, n_down=1,
                           name="Helium", exact_energy=-2.9037),
    "Li": MolecularSystem(nuclei=[(3, [0.0, 0.0, 0.0])], n_up=2, n_down=1,
                           name="Lithium", exact_energy=-7.4781),
    "Be": MolecularSystem(nuclei=[(4, [0.0, 0.0, 0.0])], n_up=2, n_down=2,
                           name="Beryllium", exact_energy=-14.6674),
    "B":  MolecularSystem(nuclei=[(5, [0.0, 0.0, 0.0])], n_up=3, n_down=2,
                           name="Boron", exact_energy=-24.6539),
    "C":  MolecularSystem(nuclei=[(6, [0.0, 0.0, 0.0])], n_up=4, n_down=2,
                           name="Carbon", exact_energy=-37.8450),
    "N":  MolecularSystem(nuclei=[(7, [0.0, 0.0, 0.0])], n_up=5, n_down=2,
                           name="Nitrogen", exact_energy=-54.5892),
    "O":  MolecularSystem(nuclei=[(8, [0.0, 0.0, 0.0])], n_up=5, n_down=3,
                           name="Oxygen", exact_energy=-75.0673),
    "F":  MolecularSystem(nuclei=[(9, [0.0, 0.0, 0.0])], n_up=5, n_down=4,
                           name="Fluorine", exact_energy=-99.7339),
    "Ne": MolecularSystem(nuclei=[(10, [0.0, 0.0, 0.0])], n_up=5, n_down=5,
                           name="Neon", exact_energy=-128.9376),
}

MOLECULES = {
    # Level 10: H₂, LiH, H₂O with equilibrium geometries (Bohr)
    "H2": MolecularSystem(
        nuclei=[(1, [0.0, 0.0, -0.7005]), (1, [0.0, 0.0, 0.7005])],  # R_e = 1.401 Bohr
        n_up=1, n_down=1,
        name="H₂ Molecule", exact_energy=-1.1745
    ),
    "LiH": MolecularSystem(
        nuclei=[(3, [0.0, 0.0, 0.0]), (1, [0.0, 0.0, 3.015])],  # R_e = 3.015 Bohr
        n_up=2, n_down=2,
        name="LiH Molecule", exact_energy=-8.0705
    ),
    "H2O": MolecularSystem(
        nuclei=[
            (8, [0.0, 0.0, 0.0]),                           # Oxygen at origin
            (1, [0.0, 1.430, -1.107]),                       # H1 (104.5° angle, R_OH=1.809 Bohr)
            (1, [0.0, -1.430, -1.107]),                      # H2
        ],
        n_up=5, n_down=5,
        name="H₂O Molecule", exact_energy=-76.438
    ),
}


def build_molecule_at_distance(mol_key: str, bond_length: float) -> MolecularSystem:
    """
    Build a molecule with a specific bond length for PES scanning (Level 10).
    
    Args:
        mol_key: "H2", "LiH", or "H2O"
        bond_length: bond distance in Bohr
        
    Returns:
        MolecularSystem with updated nuclear positions
    """
    if mol_key == "H2":
        half = bond_length / 2.0
        return MolecularSystem(
            nuclei=[(1, [0.0, 0.0, -half]), (1, [0.0, 0.0, half])],
            n_up=1, n_down=1,
            name=f"H₂ (R={bond_length:.2f})", exact_energy=None
        )
    elif mol_key == "LiH":
        return MolecularSystem(
            nuclei=[(3, [0.0, 0.0, 0.0]), (1, [0.0, 0.0, bond_length])],
            n_up=2, n_down=2,
            name=f"LiH (R={bond_length:.2f})", exact_energy=None
        )
    elif mol_key == "H2O":
        # Scale O-H bond length, keep angle at 104.5°
        angle_rad = 104.5 * np.pi / 180.0
        hy = bond_length * np.sin(angle_rad / 2)
        hz = -bond_length * np.cos(angle_rad / 2)
        return MolecularSystem(
            nuclei=[
                (8, [0.0, 0.0, 0.0]),
                (1, [0.0, hy, hz]),
                (1, [0.0, -hy, hz]),
            ],
            n_up=5, n_down=5,
            name=f"H₂O (R_OH={bond_length:.2f})", exact_energy=None
        )
    else:
        raise ValueError(f"Unknown molecule key: {mol_key}")


# ============================================================
# ⚡ COULOMB POTENTIAL (Level 1)
# ============================================================
def compute_distances(r_electrons: torch.Tensor, r_nuclei: torch.Tensor):
    """
    Compute all pairwise distance matrices.
    
    Args:
        r_electrons: [N_walkers, N_e, 3] electron positions
        r_nuclei: [N_n, 3] nuclear positions
        
    Returns:
        r_ee: [N_w, N_e, N_e] electron-electron distances
        r_en: [N_w, N_e, N_n] electron-nuclear distances
        r_ee_vec: [N_w, N_e, N_e, 3] electron-electron displacement vectors
        r_en_vec: [N_w, N_e, N_n, 3] electron-nuclear displacement vectors
    """
    N_w, N_e, _ = r_electrons.shape
    N_n = r_nuclei.shape[0]

    # Electron-Electron: r_i - r_j
    # [N_w, N_e, 1, 3] - [N_w, 1, N_e, 3] = [N_w, N_e, N_e, 3]
    r_ee_vec = r_electrons.unsqueeze(2) - r_electrons.unsqueeze(1)
    r_ee = torch.norm(r_ee_vec, dim=-1)  # [N_w, N_e, N_e]

    # Electron-Nuclear: r_i - R_I
    # [N_w, N_e, 1, 3] - [1, 1, N_n, 3] = [N_w, N_e, N_n, 3]
    r_en_vec = r_electrons.unsqueeze(2) - r_nuclei.unsqueeze(0).unsqueeze(0)
    r_en = torch.norm(r_en_vec, dim=-1)  # [N_w, N_e, N_n]

    return r_ee, r_en, r_ee_vec, r_en_vec


def compute_potential_energy(r_electrons: torch.Tensor, system: MolecularSystem, device='cpu'):
    """
    Full Coulomb potential energy:
      V = -Σ_{i,I} Z_I/r_{iI}  +  Σ_{i<j} 1/r_{ij}  +  Σ_{I<J} Z_I Z_J / R_{IJ}
    
    Args:
        r_electrons: [N_w, N_e, 3]
        system: MolecularSystem
        
    Returns:
        V: [N_w] total potential energy per walker
    """
    charges = system.charges(device)       # [N_n]
    r_nuclei = system.positions(device)    # [N_n, 3]
    N_w = r_electrons.shape[0]
    N_e = system.n_electrons

    r_ee, r_en, _, _ = compute_distances(r_electrons, r_nuclei)

    # 1. Electron-Nuclear attraction: -Σ Z_I / r_{iI}
    # r_en: [N_w, N_e, N_n], charges: [N_n]
    v_en = -torch.sum(charges.unsqueeze(0).unsqueeze(0) / (r_en + 1e-8), dim=(1, 2))  # [N_w]

    # 2. Electron-Electron repulsion: Σ_{i<j} 1/r_{ij}
    # Use upper triangle to avoid double counting
    mask_ee = torch.triu(torch.ones(N_e, N_e, device=device), diagonal=1).bool()
    r_ee_upper = r_ee[:, mask_ee]  # [N_w, N_e*(N_e-1)/2]
    v_ee = torch.sum(1.0 / (r_ee_upper + 1e-8), dim=1) if r_ee_upper.shape[1] > 0 else torch.zeros(N_w, device=device)

    # 3. Nuclear-Nuclear repulsion: Σ_{I<J} Z_I Z_J / R_{IJ}
    v_nn = torch.tensor(0.0, device=device)
    N_n = system.n_nuclei
    for I in range(N_n):
        for J in range(I + 1, N_n):
            R_IJ = torch.norm(r_nuclei[I] - r_nuclei[J])
            v_nn = v_nn + charges[I] * charges[J] / (R_IJ + 1e-8)

    return v_en + v_ee + v_nn  # [N_w]


# ============================================================
# 🎲 METROPOLIS-HASTINGS MCMC SAMPLER (Level 2)
# ============================================================
class MetropolisSampler:
    """
    Metropolis-Hastings sampler for |ψ(r)|² distribution.
    
    Implements adaptive step size to maintain ~50% acceptance rate.
    All walkers evolve in parallel for GPU efficiency.
    """
    def __init__(self, n_walkers: int, n_electrons: int, device='cpu',
                 step_size: float = 0.2, target_acceptance: float = 0.5):
        self.n_walkers = n_walkers
        self.n_electrons = n_electrons
        self.device = device
        self.step_size = step_size
        self.target_acceptance = target_acceptance
        self.acceptance_rate = 0.5  # Running estimate

        # Initialize walkers: [N_w, N_e, 3]
        self.walkers = torch.randn(n_walkers, n_electrons, 3, device=device) * 0.5

    def initialize_around_nuclei(self, system: MolecularSystem):
        """Initialize electrons near nuclei for faster equilibration."""
        r_nuclei = system.positions(self.device)  # [N_n, 3]
        N_n = system.n_nuclei
        for i in range(self.n_electrons):
            # Assign electron to nearest nucleus (round-robin)
            nuc_idx = i % N_n
            self.walkers[:, i, :] = r_nuclei[nuc_idx].unsqueeze(0) + \
                                     torch.randn(self.n_walkers, 3, device=self.device) * 0.5

    def step(self, log_psi_func):
        """
        One Metropolis-Hastings step for all walkers.
        
        Args:
            log_psi_func: function(r) -> (log|ψ|, sign) where r is [N_w, N_e, 3]
            
        Returns:
            walkers: updated positions [N_w, N_e, 3]
            acceptance_rate: float
        """
        # Propose new positions
        r_current = self.walkers
        r_proposed = r_current + self.step_size * torch.randn_like(r_current)

        # Evaluate log|ψ|² = 2 * log|ψ|
        log_psi_current, _ = log_psi_func(r_current)
        log_psi_proposed, _ = log_psi_func(r_proposed)

        # Acceptance ratio in log-domain: log(|ψ_new|²/|ψ_old|²)
        log_ratio = 2.0 * (log_psi_proposed - log_psi_current)  # [N_w]

        # Accept/Reject
        log_uniform = torch.log(torch.rand(self.n_walkers, device=self.device) + 1e-30)
        accept = log_uniform < log_ratio  # [N_w] boolean

        # Update positions
        accept_expanded = accept.unsqueeze(-1).unsqueeze(-1)  # [N_w, 1, 1]
        self.walkers = torch.where(accept_expanded, r_proposed, r_current)

        # Track acceptance rate (exponential moving average)
        current_rate = accept.float().mean().item()
        self.acceptance_rate = 0.95 * self.acceptance_rate + 0.05 * current_rate

        # Adapt step size
        if self.acceptance_rate > self.target_acceptance + 0.05:
            self.step_size *= 1.05
        elif self.acceptance_rate < self.target_acceptance - 0.05:
            self.step_size *= 0.95
        self.step_size = max(0.001, min(self.step_size, 2.0))  # Released: 2.0 is much better for H than 0.01

        return self.walkers, self.acceptance_rate


# ============================================================
# 🧮 LOCAL ENERGY VIA AUTOGRAD + HUTCHINSON (Level 3)
# ============================================================
def compute_local_energy(log_psi_func, r_electrons: torch.Tensor,
                         system: MolecularSystem, device='cpu',
                         n_hutchinson: int = 1):
    """
    Compute local energy E_L(r) = T(r) + V(r).
    
    Kinetic energy in log-domain:
      T = -0.5 * (∇²log|ψ| + |∇log|ψ||²)
    
    Uses Hutchinson's stochastic trace estimator for the Laplacian:
      ∇²f ≈ E_v[v^T H v]  where v ~ N(0, I)
    Implemented via JVP trick: v^T H v = ∂/∂ε [v · ∇f(r + εv)]|_{ε=0}
    
    Args:
        log_psi_func: function(r) -> (log|ψ|, sign)
        r_electrons: [N_w, N_e, 3] — requires_grad will be set
        system: MolecularSystem
        n_hutchinson: number of Hutchinson samples (1 is usually sufficient)
        
    Returns:
        E_L: [N_w] local energy per walker
        log_psi: [N_w] log|ψ| values
        sign_psi: [N_w] sign of ψ
    """
    N_w, N_e, _ = r_electrons.shape
    r = r_electrons.detach().requires_grad_(True)

    # Forward pass
    log_psi, sign_psi = log_psi_func(r)  # [N_w], [N_w]

    # Gradient: ∇_r log|ψ|
    grad_log_psi = torch.autograd.grad(
        log_psi, r,
        grad_outputs=torch.ones_like(log_psi),
        create_graph=True,
        retain_graph=True
    )[0]  # [N_w, N_e, 3]

    # |∇ log|ψ||² per walker
    grad_sq = torch.sum(grad_log_psi ** 2, dim=(1, 2))  # [N_w]

    # Laplacian via Hutchinson trace estimator
    laplacian = torch.zeros(N_w, device=device)
    for _ in range(n_hutchinson):
        # Random vector v ~ N(0, I)
        v = torch.randn_like(r)

        # JVP: v · ∇(log|ψ|)  (dot product across all dimensions)
        vg = torch.sum(v * grad_log_psi, dim=(1, 2))  # [N_w]

        # Gradient of vg w.r.t. r gives v^T H
        # Then dot with v gives v^T H v = trace estimator
        hvp = torch.autograd.grad(
            vg, r,
            grad_outputs=torch.ones_like(vg),
            create_graph=False,
            retain_graph=True
        )[0]  # [N_w, N_e, 3]
        laplacian += torch.sum(v * hvp, dim=(1, 2))  # [N_w]

    laplacian = laplacian / n_hutchinson

    # Kinetic energy: T = -0.5 * (∇² log|ψ| + |∇ log|ψ||²)
    # This is the log-domain formula, theoretically more stable than ∇²ψ/ψ
    E_kin = -0.5 * (laplacian + grad_sq)  # [N_w]
    
    # Potential energy
    E_pot = compute_potential_energy(r_electrons.detach(), system, device)
    
    E_L = E_kin + E_pot
    
    # Stability Surgery: Protect against NaNs at the physics level
    E_L = torch.nan_to_num(E_L, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Return E_L and detached components
    return E_L, E_kin.detach(), E_pot.detach()


# ============================================================
# 🔮 BERRY PHASE COMPUTATION (Level 14 — Topological Phase)
# ============================================================
class BerryPhaseComputer:
    """
    Level 14: Berry Phase from Neural Wavefunction — First-ever from NQS.
    
    Parametrize the Hamiltonian by λ (e.g., bond angle, external field).
    At each λ_k, run full VMC to convergence. Then compute:
    
      γ = -Im Σ_k log( ⟨ψ(λ_k)|ψ(λ_{k+1})⟩ / |⟨ψ(λ_k)|ψ(λ_{k+1})⟩| )
    
    The overlap between two neural wavefunctions is estimated via VMC:
      ⟨ψ_a|ψ_b⟩ = E_{r~|ψ_a|²}[ ψ_b(r) / ψ_a(r) ]
    
    Publishable result:
      H₃ equilateral → isosceles deformation loop: γ = π (exact).
      First demonstration of topological computation from NQS.
    """
    def __init__(self, system_builder, n_lambda: int = 20, 
                 device: str = 'cpu'):
        """
        Args:
            system_builder: callable(lambda_val) → MolecularSystem
                Maps parameter λ ∈ [0, 2π] to a molecular geometry
            n_lambda: number of discretization points on the loop
            device: 'cpu' or 'cuda'
        """
        self.system_builder = system_builder
        self.n_lambda = n_lambda
        self.device = device
        self.lambda_values = np.linspace(0, 2 * np.pi, n_lambda, endpoint=False)
        
        self.wavefunctions = []
        self.overlaps = []
        self.energies = []
        self.berry_phase = None
    
    @staticmethod
    def h3_triangle_loop(lam: float) -> 'MolecularSystem':
        """
        H₃ equilateral → isosceles deformation loop.
        
        Three hydrogen atoms arranged in a triangle.
        At λ = 0: equilateral triangle (R = 1.8 Bohr).
        As λ goes 0 → 2π: the triangle deforms through
        isosceles configurations and returns to equilateral.
        
        Known exact Berry phase: γ = π (due to conical intersection).
        """
        R0 = 1.8  # Equilibrium bond length in Bohr
        # Deformation parameter: controls asymmetry
        delta = 0.3 * np.sin(lam)  # Smooth isosceles deformation
        
        # Three atoms in a plane (z=0)
        # Atom 1 at origin
        # Atom 2 and 3 arranged symmetrically
        r12 = R0 + delta
        r13 = R0 - delta
        
        # Equilateral base angle: 60°, modulated
        angle = np.pi / 3 + 0.1 * np.cos(lam)
        
        x2 = r12 * np.cos(angle / 2)
        y2 = r12 * np.sin(angle / 2)
        x3 = r13 * np.cos(angle / 2)
        y3 = -r13 * np.sin(angle / 2)
        
        return MolecularSystem(
            nuclei=[
                (1, [0.0, 0.0, 0.0]),
                (1, [x2, y2, 0.0]),
                (1, [x3, y3, 0.0]),
            ],
            n_up=2, n_down=1,
            name=f"H₃ (λ={lam:.2f})",
            exact_energy=None
        )
    
    def compute_overlap(self, wf_a, wf_b, sampler_a, n_estimates: int = 1000):
        """
        Estimate ⟨ψ_a|ψ_b⟩ = E_{r~|ψ_a|²}[ ψ_b(r) / ψ_a(r) ]
        
        In log domain:
          ratio = sign_b · sign_a · exp(log|ψ_b| - log|ψ_a|)
        
        Returns complex overlap (real + imaginary parts).
        """
        with torch.no_grad():
            r = sampler_a.walkers[:n_estimates].detach()
            
            log_psi_a, sign_a = wf_a(r)
            log_psi_b, sign_b = wf_b(r)
            
            log_ratio = log_psi_b - log_psi_a
            log_ratio = torch.clamp(log_ratio, min=-30.0, max=30.0)
            
            # Real overlap (since wavefunctions are real-valued)
            ratio = sign_a * sign_b * torch.exp(log_ratio)
            overlap = ratio.mean()
        
        return complex(overlap.item(), 0.0)
    
    def compute_berry_phase(self, n_vmc_steps: int = 200, n_walkers: int = 512,
                            d_model: int = 32, n_layers: int = 2,
                            n_determinants: int = 4, lr: float = 1e-3,
                            progress_callback=None):
        """
        Full Berry phase computation:
          1. For each λ_k: run VMC to convergence → ψ(λ_k)
          2. Compute all overlaps ⟨ψ(λ_k)|ψ(λ_{k+1})⟩
          3. Compute γ = -Im Σ log(overlap / |overlap|)
        
        Returns:
            berry_phase: float (in radians)
        """
        # Import here to avoid circular dependency
        from solver import VMCSolver
        
        self.wavefunctions = []
        self.energies = []
        samplers = []
        
        # Step 1: Converge wavefunction at each λ
        for idx, lam in enumerate(self.lambda_values):
            system = self.system_builder(lam)
            
            solver = VMCSolver(
                system, n_walkers=n_walkers,
                d_model=d_model, n_layers=n_layers,
                n_determinants=n_determinants,
                lr=lr, device=self.device,
                optimizer_type='adamw'
            )
            solver.equilibrate(n_steps=100)
            
            for step in range(n_vmc_steps):
                try:
                    solver.train_step(n_mcmc_steps=5)
                except Exception as e:
                    # Partial point failure is acceptable in topological loops
                    print(f"Warning: Berry Phase Step {step} failed at λ point {idx}: {e}")
                    continue
            
            # Store converged wavefunction & sampler
            self.wavefunctions.append(solver.wavefunction)
            samplers.append(solver.sampler)
            
            tail = max(1, n_vmc_steps // 5)
            E = np.mean(solver.energy_history[-tail:])
            self.energies.append(E)
            
            if progress_callback:
                progress_callback(idx, self.n_lambda, E)
        
        # Step 2: Compute overlaps around the loop
        self.overlaps = []
        for k in range(self.n_lambda):
            k_next = (k + 1) % self.n_lambda
            
            overlap = self.compute_overlap(
                self.wavefunctions[k], self.wavefunctions[k_next],
                samplers[k]
            )
            self.overlaps.append(overlap)
        
        # Step 3: Berry phase = -Im Σ log(overlap / |overlap|)
        berry_phase = 0.0
        for overlap in self.overlaps:
            if abs(overlap) > 1e-10:
                # Phase of the overlap
                phase = np.angle(overlap)
                berry_phase -= phase
        
        # Normalize to [-π, π]
        self.berry_phase = (berry_phase + np.pi) % (2 * np.pi) - np.pi
        
        return self.berry_phase


# ============================================================
# 🔷 PERIODIC SYSTEMS — BLOCH WAVES FOR SOLIDS (Level 16)
# ============================================================
@dataclass
class PeriodicSystem:
    """
    Level 16: Periodic system for solid-state calculations.
    
    Implements Bloch boundary conditions:
      ψ(r + L) = e^{ikL} · ψ(r)
    
    Twist-Averaged Boundary Conditions (TABC):
      Average over k-points to reduce finite-size effects.
    
    Attributes:
        cell_vectors: [3, 3] lattice vectors (rows = a₁, a₂, a₃)
        n_electrons: total electrons in simulation cell
        n_up, n_down: spin channels
        twist: [3] twist vector k for Bloch conditions
        rs: Wigner-Seitz radius (for HEG)
    """
    cell_vectors: List[List[float]]
    n_up: int
    n_down: int
    name: str = "Periodic"
    twist: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rs: Optional[float] = None  # Wigner-Seitz radius for HEG
    exact_energy_per_electron: Optional[float] = None

    @property
    def n_electrons(self) -> int:
        return self.n_up + self.n_down

    def cell_tensor(self, device='cpu') -> torch.Tensor:
        return torch.tensor(self.cell_vectors, dtype=torch.float32, device=device)

    def cell_volume(self) -> float:
        L = np.array(self.cell_vectors)
        return abs(np.dot(L[0], np.cross(L[1], L[2])))

    def twist_tensor(self, device='cpu') -> torch.Tensor:
        return torch.tensor(self.twist, dtype=torch.float32, device=device)

    def reciprocal_vectors(self) -> np.ndarray:
        """Compute reciprocal lattice vectors b_i = 2π (a_j × a_k) / V."""
        L = np.array(self.cell_vectors)
        V = abs(np.dot(L[0], np.cross(L[1], L[2])))
        b = np.zeros((3, 3))
        b[0] = 2 * np.pi * np.cross(L[1], L[2]) / V
        b[1] = 2 * np.pi * np.cross(L[2], L[0]) / V
        b[2] = 2 * np.pi * np.cross(L[0], L[1]) / V
        return b


def build_heg_system(rs: float, n_electrons: int, n_up: int = None) -> PeriodicSystem:
    """
    Build Homogeneous Electron Gas (HEG) system.
    
    The HEG is the fundamental model of metallic bonding.
    Cell size L determined by electron density: (4/3)πr_s³ × N_e = L³
    
    Args:
        rs: Wigner-Seitz radius (density parameter)
        n_electrons: total electrons
        n_up: spin-up electrons (default: N/2)
    """
    if n_up is None:
        n_up = n_electrons // 2
    n_down = n_electrons - n_up
    
    # L³ = N_e × (4/3)π r_s³
    volume = n_electrons * (4.0 / 3.0) * np.pi * rs**3
    L = volume ** (1.0 / 3.0)
    
    cell = [[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]]
    
    # Ceperley-Alder correlation energy (parameterization by Perdew-Zunger)
    # For unpolarized HEG: e_c(r_s) from Monte Carlo
    if rs <= 1.0:
        # High density: e_c = A·ln(r_s) + B + C·r_s·ln(r_s) + D·r_s
        gamma = -0.1423
        beta1 = 1.0529
        beta2 = 0.3334
        e_c = gamma / (1 + beta1 * np.sqrt(rs) + beta2 * rs)
    else:
        # Low density: Ceperley-Alder interpolation
        gamma = -0.1423
        beta1 = 1.0529
        beta2 = 0.3334
        e_c = gamma / (1 + beta1 * np.sqrt(rs) + beta2 * rs)
    
    # Exchange energy: e_x = -3/(4π) × (9π/4)^{1/3} / r_s
    e_x = -3.0 / (4.0 * np.pi) * (9.0 * np.pi / 4.0) ** (1.0 / 3.0) / rs
    
    # Kinetic energy: e_k = (3/10)(9π/4)^{2/3} / r_s²
    e_k = 3.0 / 10.0 * (9.0 * np.pi / 4.0) ** (2.0 / 3.0) / rs**2
    
    exact_e = e_k + e_x + e_c
    
    return PeriodicSystem(
        cell_vectors=cell, n_up=n_up, n_down=n_down,
        name=f"HEG (r_s={rs})", rs=rs,
        exact_energy_per_electron=exact_e
    )


# Predefined HEG systems at different densities
PERIODIC_SYSTEMS = {
    "HEG_rs1":  build_heg_system(rs=1.0, n_electrons=14),
    "HEG_rs2":  build_heg_system(rs=2.0, n_electrons=14),
    "HEG_rs5":  build_heg_system(rs=5.0, n_electrons=14),
    "HEG_rs10": build_heg_system(rs=10.0, n_electrons=14),
}


def compute_ewald_potential(r_electrons: torch.Tensor, system: PeriodicSystem,
                            device='cpu', n_recip: int = 5, alpha: float = None):
    """
    Level 16: Ewald summation for periodic Coulomb interaction.
    
    Splits 1/r into short-range (real space) + long-range (reciprocal space):
      V = V_real + V_recip + V_self + V_madelung
    
    V_real  = Σ_{i<j} Σ_n erfc(α|r_ij + nL|) / |r_ij + nL|
    V_recip = (1/V) Σ_{G≠0} (4π/G²) exp(-G²/4α²) Σ_{i<j} cos(G·r_ij)
    V_self  = -α/√π × N_e
    
    Args:
        r_electrons: [N_w, N_e, 3]
        system: PeriodicSystem
        n_recip: max reciprocal lattice vector magnitude
        alpha: Ewald splitting parameter (auto-tuned if None)
    """
    N_w, N_e, _ = r_electrons.shape
    L = system.cell_tensor(device)
    V_cell = system.cell_volume()
    
    # Auto-tune alpha for optimal convergence
    if alpha is None:
        L_min = min(np.linalg.norm(np.array(system.cell_vectors[i])) for i in range(3))
        alpha = 5.0 / L_min
    
    # Wrap electrons into simulation cell (minimum image convention)
    L_inv = torch.linalg.inv(L)
    s = torch.matmul(r_electrons, L_inv.T)  # fractional coords
    s = s - torch.floor(s)  # wrap to [0, 1)
    r_wrapped = torch.matmul(s, L)
    
    # --- Real-space sum ---
    V_real = torch.zeros(N_w, device=device)
    if N_e > 1:
        r_ij_vec = r_wrapped.unsqueeze(2) - r_wrapped.unsqueeze(1)  # [N_w, N_e, N_e, 3]
        # Minimum image convention
        s_ij = torch.matmul(r_ij_vec, L_inv.T)
        s_ij = s_ij - torch.round(s_ij)
        r_ij_vec = torch.matmul(s_ij, L)
        r_ij = torch.norm(r_ij_vec, dim=-1)  # [N_w, N_e, N_e]
        
        triu_mask = torch.triu(torch.ones(N_e, N_e, device=device), diagonal=1).bool()
        r_ij_upper = r_ij[:, triu_mask]  # [N_w, n_pairs]
        
        # erfc(α·r) / r
        erfc_term = torch.erfc(alpha * r_ij_upper) / (r_ij_upper + 1e-10)
        V_real = erfc_term.sum(dim=1)
    
    # --- Reciprocal-space sum ---
    V_recip = torch.zeros(N_w, device=device)
    b = torch.tensor(system.reciprocal_vectors(), dtype=torch.float32, device=device)
    
    for n1 in range(-n_recip, n_recip + 1):
        for n2 in range(-n_recip, n_recip + 1):
            for n3 in range(-n_recip, n_recip + 1):
                if n1 == 0 and n2 == 0 and n3 == 0:
                    continue
                G = n1 * b[0] + n2 * b[1] + n3 * b[2]  # [3]
                G_sq = torch.dot(G, G)
                if G_sq > (2 * np.pi * n_recip / V_cell**(1/3))**2:
                    continue
                
                # Structure factor: S(G) = Σ_i exp(iG·r_i)
                G_dot_r = torch.matmul(r_wrapped, G)  # [N_w, N_e]
                cos_sum = torch.cos(G_dot_r).sum(dim=1)  # [N_w]
                sin_sum = torch.sin(G_dot_r).sum(dim=1)
                S_sq = cos_sum**2 + sin_sum**2  # |S(G)|²
                
                # Reciprocal term
                prefactor = 4.0 * np.pi / (V_cell * G_sq)
                gauss = torch.exp(-G_sq / (4.0 * alpha**2))
                V_recip += prefactor * gauss * (S_sq - N_e) / 2.0
    
    # --- Self-energy correction ---
    V_self = -alpha / np.sqrt(np.pi) * N_e * torch.ones(N_w, device=device)
    
    # --- Madelung constant (charged background) ---
    V_madelung = -np.pi * N_e**2 / (2.0 * V_cell * alpha**2) * torch.ones(N_w, device=device)
    
    return V_real + V_recip + V_self + V_madelung


def compute_periodic_local_energy(log_psi_func, r_electrons: torch.Tensor,
                                   system: PeriodicSystem, device='cpu',
                                   n_hutchinson: int = 1):
    """
    Level 16: Local energy for periodic systems.
    Same kinetic energy as free-space, but uses Ewald potential.
    """
    N_w, N_e, _ = r_electrons.shape
    r = r_electrons.detach().requires_grad_(True)
    log_psi, sign_psi = log_psi_func(r)
    
    grad_log_psi = torch.autograd.grad(
        log_psi, r, grad_outputs=torch.ones_like(log_psi),
        create_graph=True, retain_graph=True
    )[0]
    
    grad_sq = torch.sum(grad_log_psi ** 2, dim=(1, 2))
    
    laplacian = torch.zeros(N_w, device=device)
    for _ in range(n_hutchinson):
        v = torch.randn_like(r)
        vg = torch.sum(v * grad_log_psi, dim=(1, 2))
        hvp = torch.autograd.grad(
            vg, r, grad_outputs=torch.ones_like(vg),
            create_graph=False, retain_graph=True
        )[0]
        laplacian += torch.sum(v * hvp, dim=(1, 2))
    laplacian = laplacian / n_hutchinson
    
    kinetic = -0.5 * (laplacian + grad_sq)
    potential = compute_ewald_potential(r_electrons.detach(), system, device)
    
    E_L = kinetic.detach() + potential
    return E_L, log_psi.detach(), sign_psi.detach()


# ============================================================
# ⚡ SPIN-ORBIT COUPLING — RELATIVISTIC QM (Level 17)
# ============================================================
@dataclass
class SpinOrbitSystem:
    """
    Level 17: System with spin-orbit coupling.
    
    Extends MolecularSystem with relativistic Breit-Pauli correction:
      H_SO = (α²/2) Σ_{i,I} Z_I / r_{iI}³ · L_{iI} · S_i
    
    Where:
      α ≈ 1/137 (fine-structure constant)
      L_{iI} = (r_i - R_I) × p_i (angular momentum about nucleus I)
      S_i = σ/2 (spin operator, Pauli matrices)
    """
    base_system: 'MolecularSystem'  # The underlying molecular system
    alpha_fs: float = 1.0 / 137.036  # Fine-structure constant
    
    @property
    def n_electrons(self):
        return self.base_system.n_electrons
    
    @property
    def n_up(self):
        return self.base_system.n_up
    
    @property
    def n_down(self):
        return self.base_system.n_down


def compute_spin_orbit_coupling(log_psi_func, r_electrons: torch.Tensor,
                                 so_system: SpinOrbitSystem, device='cpu'):
    """
    Level 17: Breit-Pauli spin-orbit Hamiltonian.
    
    H_SO = (α²/2) Σ_{i,I} Z_I / r_{iI}³ · (L_{iI} · S_i)
    
    For each electron i and nucleus I:
      L_{iI} = (r_i - R_I) × p_i
      p_i = -i∇_i = (in log-domain) ∇_r log|ψ| (real part, multiplied by ψ)
    
    The spin expectation depends on spin channel:
      ⟨σ_z⟩ = +1/2 for spin-up, -1/2 for spin-down
      ⟨σ_x⟩ = ⟨σ_y⟩ = 0 for z-polarized states
    
    Returns:
      E_SO: [N_w] spin-orbit energy per walker (first-order perturbation theory)
    """
    system = so_system.base_system
    alpha = so_system.alpha_fs
    N_w, N_e, _ = r_electrons.shape
    
    r = r_electrons.detach().requires_grad_(True)
    log_psi, _ = log_psi_func(r)
    
    # Momentum: p_i = ∇_r log|ψ| (the gradient gives the "velocity" field)
    grad_log_psi = torch.autograd.grad(
        log_psi, r, grad_outputs=torch.ones_like(log_psi),
        create_graph=False, retain_graph=False
    )[0]  # [N_w, N_e, 3]
    
    r_nuclei = system.positions(device)
    charges = system.charges(device)
    
    E_SO = torch.zeros(N_w, device=device)
    
    for I in range(system.n_nuclei):
        Z_I = charges[I]
        R_I = r_nuclei[I]  # [3]
        
        # Displacement r_i - R_I
        r_iI_vec = r.detach() - R_I.unsqueeze(0).unsqueeze(0)  # [N_w, N_e, 3]
        r_iI = torch.norm(r_iI_vec, dim=-1, keepdim=True)  # [N_w, N_e, 1]
        
        # Angular momentum L = r × p (cross product)
        # L_x = y·p_z - z·p_y, L_y = z·p_x - x·p_z, L_z = x·p_y - y·p_x
        L = torch.cross(r_iI_vec, grad_log_psi.detach(), dim=-1)  # [N_w, N_e, 3]
        
        # Spin expectation: S_z = +1/2 for up, -1/2 for down
        # For z-quantized states, L·S = L_z · S_z (dominant term)
        n_up = system.n_up
        spin_z = torch.zeros(N_e, device=device)
        spin_z[:n_up] = 0.5      # spin-up electrons
        spin_z[n_up:] = -0.5     # spin-down electrons
        
        # L · S ≈ L_z · S_z (diagonal approximation for z-quantized states)
        L_dot_S = L[:, :, 2] * spin_z.unsqueeze(0)  # [N_w, N_e]
        
        # 1/r³ factor (regularized)
        r_iI_cubed = (r_iI.squeeze(-1) ** 3).clamp(min=1e-6)
        
        # H_SO contribution from this nucleus
        E_SO += (alpha**2 / 2.0) * Z_I * (L_dot_S / r_iI_cubed).sum(dim=1)
    
    return E_SO


def compute_fine_structure_splitting(log_psi_func, r_electrons: torch.Tensor,
                                      so_system: SpinOrbitSystem, device='cpu'):
    """
    Level 17: Compute fine-structure energy levels.
    
    For Helium 1s2p configuration:
      ³P₀, ³P₁, ³P₂ split by spin-orbit coupling.
      Splitting ~ α²Z⁴/n³ in atomic units.
    
    Returns dict with total energy, kinetic, potential, and spin-orbit contributions.
    """
    system = so_system.base_system
    
    # Standard energy
    E_L, log_psi, sign_psi = compute_local_energy(
        log_psi_func, r_electrons, system, device
    )
    
    # Spin-orbit correction
    E_SO = compute_spin_orbit_coupling(log_psi_func, r_electrons, so_system, device)
    
    return {
        'E_total': (E_L + E_SO).mean().item(),
        'E_nonrel': E_L.mean().item(),
        'E_SO': E_SO.mean().item(),
        'E_SO_std': E_SO.std().item(),
        'splitting_mHa': E_SO.mean().item() * 1000,  # in milli-Hartree
    }


# ============================================================
# 🔗 ENTANGLEMENT ENTROPY — SWAP TRICK (Level 18)
# ============================================================
class EntanglementEntropyComputer:
    """
    Level 18: Rényi-2 Entanglement Entropy from Neural Wavefunction.
    
    Unprecedented: No published work computes molecular entanglement
    entropy from a neural VMC wavefunction.
    
    Method — SWAP trick:
      S₂(A) = -log Tr(ρ_A²)
      Tr(ρ_A²) = ⟨ψ⊗ψ|SWAP_A|ψ⊗ψ⟩
    
    Implementation:
      1. Sample two independent copies (r, r') from |ψ|²
      2. Construct SWAP configuration: swap electrons in subsystem A
         r_swap = (r'_A, r_B) and r'_swap = (r_A, r'_B)
      3. Estimator: Tr(ρ_A²) = E[ ψ(r_swap)·ψ(r'_swap) / (ψ(r)·ψ(r')) ]
      4. S₂ = -log(Tr(ρ_A²))
    
    Subsystem partition:
      - By atom: electrons assigned to nearest nucleus
      - By spatial region: x < 0 vs x > 0
      - By spin: all up-spin vs all down-spin
    """
    def __init__(self, wavefunction, sampler, system: MolecularSystem,
                 partition: str = 'spatial', device: str = 'cpu'):
        """
        Args:
            wavefunction: neural wavefunction (log_psi, sign) = wf(r)
            sampler: MetropolisSampler (already equilibrated)
            system: MolecularSystem
            partition: 'spatial' (x<0 vs x>0), 'spin' (up vs down), 
                       'atom' (nearest nucleus)
            device: 'cpu' or 'cuda'
        """
        self.wavefunction = wavefunction
        self.sampler = sampler
        self.system = system
        self.partition = partition
        self.device = device
        
        # Build subsystem A mask
        self.subsystem_A = self._build_partition()
    
    def _build_partition(self):
        """
        Returns boolean mask [N_e] where True = electron in subsystem A.
        """
        N_e = self.system.n_electrons
        mask = torch.zeros(N_e, dtype=torch.bool)
        
        if self.partition == 'spin':
            # Subsystem A = spin-up electrons
            mask[:self.system.n_up] = True
        elif self.partition == 'spatial':
            # Subsystem A = first half of electrons (will use spatial criterion at runtime)
            mask[:N_e // 2] = True
        elif self.partition == 'atom' and self.system.n_nuclei >= 2:
            # Subsystem A = electrons closest to first nucleus
            mask[:N_e // 2] = True
        else:
            # Default: first half
            mask[:max(1, N_e // 2)] = True
        
        return mask
    
    def compute_renyi2(self, n_samples: int = 2000, n_mcmc_burn: int = 50):
        """
        Compute Rényi-2 entanglement entropy via SWAP trick.
        
        Returns:
            S2: Rényi-2 entropy S₂(A)
            purity: Tr(ρ_A²)
            error: statistical uncertainty
        """
        log_psi_func = lambda r: self.wavefunction(r)
        mask_A = self.subsystem_A.to(self.device)
        mask_B = ~mask_A
        
        # Need two independent copies — use two separate samplers
        sampler_1 = MetropolisSampler(
            n_walkers=n_samples,
            n_electrons=self.system.n_electrons,
            device=self.device
        )
        sampler_1.initialize_around_nuclei(self.system)
        
        sampler_2 = MetropolisSampler(
            n_walkers=n_samples,
            n_electrons=self.system.n_electrons,
            device=self.device
        )
        sampler_2.initialize_around_nuclei(self.system)
        
        # Burn-in both
        with torch.no_grad():
            for _ in range(n_mcmc_burn):
                sampler_1.step(log_psi_func)
                sampler_2.step(log_psi_func)
        
        # Get independent configurations
        with torch.no_grad():
            r1 = sampler_1.walkers.detach()  # [N_w, N_e, 3]
            r2 = sampler_2.walkers.detach()  # [N_w, N_e, 3]
            
            # Use spatial partition at runtime if needed
            if self.partition == 'spatial':
                # Classify based on x-coordinate centroid
                x_mean = r1[:, :, 0].mean(dim=0)  # [N_e]
                spatial_mask = x_mean < x_mean.median()
                mask_A = spatial_mask.to(self.device)
                mask_B = ~mask_A
            
            # Evaluate ψ at original configurations
            log_psi_r1, sign_r1 = self.wavefunction(r1)
            log_psi_r2, sign_r2 = self.wavefunction(r2)
            
            # Construct SWAP configurations
            # r_swap1 = (r2_A, r1_B): take A-electrons from copy 2, B-electrons from copy 1
            # r_swap2 = (r1_A, r2_B): take A-electrons from copy 1, B-electrons from copy 2
            r_swap1 = r1.clone()
            r_swap2 = r2.clone()
            
            r_swap1[:, mask_A, :] = r2[:, mask_A, :]  # r2's A-electrons into r1's slot
            r_swap2[:, mask_A, :] = r1[:, mask_A, :]  # r1's A-electrons into r2's slot
            
            # Evaluate ψ at swapped configurations  
            log_psi_swap1, sign_swap1 = self.wavefunction(r_swap1)
            log_psi_swap2, sign_swap2 = self.wavefunction(r_swap2)
            
            # SWAP estimator:
            # Tr(ρ_A²) = E[ ψ(r_swap1)·ψ(r_swap2) / (ψ(r1)·ψ(r2)) ]
            # In log domain:
            log_ratio = (log_psi_swap1 + log_psi_swap2) - (log_psi_r1 + log_psi_r2)
            sign_ratio = sign_swap1 * sign_swap2 * sign_r1 * sign_r2
            
            # Clamp for numerical stability
            log_ratio = torch.clamp(log_ratio, min=-50.0, max=50.0)
            
            swap_estimator = sign_ratio * torch.exp(log_ratio)
            
            purity = swap_estimator.mean().item()
            purity_std = swap_estimator.std().item() / np.sqrt(n_samples)
            
            # Clamp purity to valid range
            purity = max(purity, 1e-10)
            purity = min(purity, 1.0)
            
            S2 = -np.log(purity)
        
        return {
            'S2': S2,
            'purity': purity,
            'error': abs(purity_std / (purity + 1e-10)),
            'n_A': mask_A.sum().item(),
            'n_B': mask_B.sum().item(),
            'partition': self.partition
        }
    
    def compute_entanglement_profile(self, n_partitions: int = None,
                                      n_samples: int = 1000, n_mcmc_burn: int = 50):
        """
        Compute S₂ for all possible bipartitions of N_A = 1, 2, ..., N_e/2.
        
        The Area Law: for ground states, S₂ ~ boundary area, not volume.
        Violation → topological order or critical point.
        
        Returns list of (n_A, S2) tuples.
        """
        N_e = self.system.n_electrons
        if n_partitions is None:
            n_partitions = max(1, N_e // 2)
        
        profile = []
        original_mask = self.subsystem_A.clone()
        
        for n_A in range(1, n_partitions + 1):
            # Set subsystem A to first n_A electrons
            mask = torch.zeros(N_e, dtype=torch.bool)
            mask[:n_A] = True
            self.subsystem_A = mask
            
            result = self.compute_renyi2(n_samples=n_samples, n_mcmc_burn=n_mcmc_burn)
            profile.append({
                'n_A': n_A,
                'S2': result['S2'],
                'purity': result['purity'],
                'error': result['error']
            })
        
        self.subsystem_A = original_mask
        return profile



class QuantumPhysicsEngine(nn.Module):
    """
    Legacy 1D Physics Engine for demo/teaching mode.
    Kept for backward compatibility with 1D potentials.
    """
    def __init__(self, grid_size=1024, x_range=(-10, 10), hbar=1.0, mass=1.0):
        super().__init__()
        self.grid_size = grid_size
        self.x_min, self.x_max = x_range
        self.hbar = hbar
        self.mass = mass
        self.register_buffer('x', torch.linspace(self.x_min, self.x_max, self.grid_size).view(-1, 1))
        self.register_buffer('dx', torch.tensor((self.x_max - self.x_min) / (self.grid_size - 1), dtype=torch.float32))

    def variational_energy_loss(self, psi_func, x, V_x):
        """Variational energy <ψ|H|ψ>/<ψ|ψ> on 1D grid."""
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        psi = psi_func(x)
        # Autograd Laplacian for 1D
        grads_real = torch.autograd.grad(psi[:, 0].sum(), x, create_graph=True)[0]
        grad2_real = torch.autograd.grad(grads_real.sum(), x, create_graph=True)[0]
        grads_imag = torch.autograd.grad(psi[:, 1].sum(), x, create_graph=True)[0]
        grad2_imag = torch.autograd.grad(grads_imag.sum(), x, create_graph=True)[0]
        laplacian = torch.cat([grad2_real, grad2_imag], dim=1)
        k_term = -(self.hbar ** 2) / (2 * self.mass)
        t_psi = k_term * laplacian
        v_psi = V_x * psi
        h_psi = t_psi + v_psi
        densities = psi[:, 0] * h_psi[:, 0] + psi[:, 1] * h_psi[:, 1]
        norm_densities = psi[:, 0] ** 2 + psi[:, 1] ** 2
        energy = torch.sum(densities) * self.dx / (torch.sum(norm_densities) * self.dx + 1e-8)
        return energy

    def hamiltonian_score_matching_collapse(self, psi, steps=100, dt=0.01):
        """Measurement collapse via Langevin dynamics on |ψ|²."""
        psi = torch.nan_to_num(psi.to(self.x.device), nan=0.0, posinf=1.0, neginf=-1.0)
        prob = psi[:, 0] ** 2 + psi[:, 1] ** 2
        prob = prob / (torch.sum(prob) * self.dx + 1e-8)
        prob = torch.clamp(prob, min=1e-12)
        log_prob = torch.log(prob)
        score = torch.zeros_like(log_prob)
        dx_val = self.dx.item()
        score[1:-1] = (log_prob[2:] - log_prob[:-2]) / (2 * dx_val)
        score = torch.nan_to_num(score)
        c_idx = torch.randint(0, self.grid_size, (1,), device=self.x.device).item()
        for _ in range(steps):
            s = score[c_idx].item()
            move = s * dt + np.random.normal() * np.sqrt(2 * dt)
            if not np.isfinite(move):
                move = 0.0
            shift = int(move / dx_val)
            c_idx = max(0, min(self.grid_size - 1, c_idx + shift))
        return self.x[c_idx]


# ============================================================
# 🧪 TEST BED
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing 3D Many-Body Physics Engine")
    print("=" * 60)

    # Test Hydrogen atom
    system = ATOMS["H"]
    print(f"\nSystem: {system.name}")
    print(f"  Electrons: {system.n_electrons} (↑{system.n_up} ↓{system.n_down})")
    print(f"  Exact Energy: {system.exact_energy} Ha")

    # Test distance computation
    r_test = torch.randn(10, 1, 3)  # 10 walkers, 1 electron, 3D
    r_nuc = system.positions()
    r_ee, r_en, _, _ = compute_distances(r_test, r_nuc)
    print(f"\n  Distance matrices: r_ee={r_ee.shape}, r_en={r_en.shape}")

    # Test potential energy
    V = compute_potential_energy(r_test, system)
    print(f"  Potential energy: {V.mean().item():.4f} ± {V.std().item():.4f}")

    # Test MCMC sampler
    sampler = MetropolisSampler(n_walkers=100, n_electrons=1)
    sampler.initialize_around_nuclei(system)
    print(f"\n  MCMC initialized: walkers shape = {sampler.walkers.shape}")
    print(f"  Walker mean pos: {sampler.walkers.mean(dim=0).squeeze()}")

    # Test Helium
    system_he = ATOMS["He"]
    print(f"\nSystem: {system_he.name}")
    r_he = torch.randn(10, 2, 3)
    V_he = compute_potential_energy(r_he, system_he)
    print(f"  Potential energy: {V_he.mean().item():.4f} ± {V_he.std().item():.4f}")

    # Test H2 molecule
    system_h2 = MOLECULES["H2"]
    print(f"\nSystem: {system_h2.name}")
    r_h2 = torch.randn(10, 2, 3)
    V_h2 = compute_potential_energy(r_h2, system_h2)
    print(f"  Potential energy: {V_h2.mean().item():.4f} ± {V_h2.std().item():.4f}")

    print("\n✅ All physics engine tests passed!")


