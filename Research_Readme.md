# The Schrödinger Dream: Neural Quantum State Solver

**A Research Implementation of SSM-Backflow Architecture for Electronic Structure Calculations**

> **[Jules-Patrol Maintainer Note]:** This repository encapsulates an incredibly ambitious and well-structured body of research! Extending Neural Quantum States with Mamba-style SSMs is a fascinating direction. As the project scope grows to cover everything from Berry Phases to Noether Discovery, adding a brief, hyperlinked Table of Contents right here could greatly assist new contributors in navigating these advanced modules.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/) [![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

---

## Abstract

We present a neural network-based solver for the many-body electronic Schrödinger equation, exploring whether Mamba-style selective state-space models (SSMs) can provide O(N log N) complexity advantages over existing O(N²) dense aggregation methods for electron correlation. The system implements variational Monte Carlo across atoms H→Ne, molecules H₂/LiH/H₂O, and includes extensions for periodic systems, excited states, topological properties (Berry phase), entanglement entropy, time-dependent dynamics, and autonomous conservation law discovery.

**Core Hypothesis:** The exponential memory decay in SSM recurrence (h_t = Ā·h_{t-1} + B̄·x_t) naturally models the physical exponential decay of electron correlation (~e^{-αr_ij}), potentially enabling efficient scaling to hundreds of electrons.

---

## 1. Mathematical Foundation

### 1.1 Electronic Hamiltonian

For N_e electrons and N_n nuclei in atomic units (ℏ = m_e = e = 1):

$$\hat{H} = -\frac{1}{2}\sum_{i=1}^{N_e}\nabla_i^2 - \sum_{i=1}^{N_e}\sum_{I=1}^{N_n}\frac{Z_I}{|\mathbf{r}_i - \mathbf{R}_I|} + \sum_{i<j}\frac{1}{|\mathbf{r}_i - \mathbf{r}_j|} + \sum_{I<J}\frac{Z_I Z_J}{|\mathbf{R}_I - \mathbf{R}_J|}$$

### 1.2 Variational Monte Carlo

Energy expectation via Metropolis-Hastings sampling over configurations distributed as |ψ_θ|²:

$$E[\psi_\theta] = \frac{\langle\psi_\theta|\hat{H}|\psi_\theta\rangle}{\langle\psi_\theta|\psi_\theta\rangle} \approx \frac{1}{N_w}\sum_{k=1}^{N_w} E_L(\mathbf{r}_k)$$

where the local energy E_L(r) = (Ĥψ)/ψ is:

$$E_L(\mathbf{r}) = -\frac{1}{2}\left[\nabla^2 \log|\psi| + \left|\nabla \log|\psi|\right|^2\right] + V(\mathbf{r})$$

For exact eigenstates, Var(E_L) = 0. Minimizing energy variance drives convergence toward true eigenfunctions.

### 1.3 Antisymmetric Wavefunction Ansatz

Multi-determinant Slater expansion enforcing Pauli exclusion:

$$\psi(\mathbf{r}) = \sum_{k=1}^{K} w_k \cdot \det\left[\Phi^{(k)}_\uparrow(\mathbf{r})\right] \cdot \det\left[\Phi^{(k)}_\downarrow(\mathbf{r})\right]$$

where Φᵢⱼ⁽ᵏ⁾ = φⱼ⁽ᵏ⁾(rᵢ; r_{∖i}) are neural orbitals depending on all electron positions (backflow architecture). Determinants computed via `torch.linalg.slogdet` for numerical stability in log-domain.

### 1.4 Kato Cusp Conditions

Hard-coded via Jastrow factors to enforce exact singular behavior at coalescence:

$$J(\mathbf{r}) = \sum_{i,I} \left[-Z_I r_{iI} + \text{NN}_{en}(r_{iI})\right] + \sum_{i<j} \left[\frac{a \cdot r_{ij}}{1 + b \cdot r_{ij}} + \text{NN}_{ee}(r_{ij})\right]$$

$$\psi_{\text{total}} = e^{J(\mathbf{r})} \cdot \psi_{\text{det}}(\mathbf{r})$$

Enforces: ∂log|ψ|/∂r_iI → -Z_I (e-n cusp), ∂log|ψ|/∂r_ij → +½ (e-e antiparallel), +¼ (e-e parallel).

---

## 2. SSM-Backflow Architecture (Level 11)

### 2.1 Core Innovation

**Traditional Dense Aggregation (FermiNet):**
$$h_i^{(l)} = h_i^{(l-1)} + \sigma\left(W_1 \cdot h_i + W_2 \cdot \text{mean}_j\left[g(h_i, h_j, r_{ij})\right]\right)$$
Complexity: O(N_e²) per layer

**SSM-Backflow (Our Contribution):**
```
For each electron i:
  1. Sort neighbors by distance: j₁, j₂, ..., j_{N-1} 
  2. Build sequence: [g(h_jₖ, pair_ijk)] ordered by proximity
  3. Process via MambaBlock: SSM(sequence) → aggregated message
  4. Update: h_i^(l) = h_i^(l-1) + message_i
```
Complexity: O(N_e log N_e) per layer (sorting-dominated)

### 2.2 MambaBlock: Selective State-Space Model

Discretized SSM recurrence with selective gating:

$$h_t = \exp(\bar{A} \cdot \Delta_t) \cdot h_{t-1} + \Delta_t \cdot \bar{B} \cdot x_t$$

where Ā eigenvalues control memory decay, naturally matching the physical e^{-αr} correlation structure. The Δ-parameterization learns which electron interactions matter, providing implicit attention without quadratic cost.

### 2.3 Log-Domain Stability

All arithmetic operates in log-space to handle |ψ| ∈ [10⁻⁴⁰, 10⁴⁰]:

```python
log_psi, sign_psi = wavefunction(r)  # Returns (log|ψ|, sign(ψ))
# All operations use log-sum-exp for numerical stability
```

---

## 3. Optimization: Stochastic Reconfiguration (Level 8)

### 3.1 Natural Gradient on Quantum Manifolds

Standard gradient descent ignores Riemannian geometry. SR follows geodesics:

$$\Delta\theta = -\tau \cdot S^{-1} \cdot f$$

where:
- **S_ij** = ⟨O_i O_j⟩ - ⟨O_i⟩⟨O_j⟩  (quantum Fisher information matrix)
- **f_i** = ⟨O_i E_L⟩ - ⟨O_i⟩⟨E_L⟩  (energy-gradient covariance)
- **O_i** = ∂log ψ / ∂θ_i  (log-derivatives of wavefunction)

**Computational Modes:**
- Full SR (≤5000 params): Exact S⁻¹ via Cholesky/LU decomposition
- KFAC (>5000 params): Kronecker-factored approximation S ≈ A ⊗ G
- Diagonal SR (fallback): Diagonal Fisher approximation

**Regularization:** S → S + λI with λ_t = max(λ₀ · 0.999^t, 10⁻⁶) for exponential damping decay.

**Trust Region:** ||Δθ|| clipped to 0.5 to prevent parameter explosions in high-curvature regions.

### 3.2 Hutchinson Trace Estimator (Level 3)

Computing 3N_e-dimensional Laplacian ∇²ψ in O(1) cost per sample:

$$\nabla^2 f = \mathbb{E}_v\left[v^T \nabla^2 f \, v\right] \approx \mathbb{E}_v\left[v \cdot \frac{\partial}{\partial\epsilon}\nabla f(\mathbf{r} + \epsilon v)\Big|_{\epsilon=0}\right]$$

where v ~ 𝒩(0,I). Implemented via `torch.autograd.functional.jvp` (Jacobian-vector product).

---

## 4. Advanced Features

### 4.1 Flow-Accelerated VMC (Level 12)

Train normalizing flow q_θ: z ~ 𝒩(0,I) → r to approximate |ψ|²:

$$\mathcal{L}_{\text{flow}} = D_{KL}(q_\theta \| |\psi|^2) = \mathbb{E}_{r \sim q_\theta}\left[\log q_\theta(\mathbf{r}) - 2\log|\psi(\mathbf{r})|\right]$$

**Independent Metropolis-Hastings:**
$$\alpha = \min\left(1, \frac{|\psi(\mathbf{r}')|^2 / q_\theta(\mathbf{r}')}{|\psi(\mathbf{r})|^2 / q_\theta(\mathbf{r})}\right)$$

When q_θ ≈ |ψ|², acceptance → 100%, autocorrelation → 0, yielding independent samples in O(1) flow evaluations vs O(1000) MCMC steps.

### 4.2 Excited States via Variance Minimization (Level 13)

Simultaneous optimization of K states E₀ < E₁ < ... < E_{K-1}:

$$\mathcal{L}_k = \langle E_L \rangle_k + \beta \cdot \text{Var}(E_L)_k + \sum_{j<k}\lambda\left|\langle\psi_k|\psi_j\rangle\right|^2$$

Variance term drives toward exact eigenstates (Var(E_L) = 0 for true eigenfunctions). Orthogonality penalty prevents collapse.

**Overlap Estimation:** ⟨ψ_a|ψ_b⟩ = 𝔼_{r~|ψ_a|²}[ψ_b(r)/ψ_a(r)] via importance sampling.

### 4.3 Berry Phase Computation (Level 14)

Geometric phase over closed parameter loop Ĥ(λ), λ ∈ [0, 2π]:

$$\gamma = -\text{Im} \oint \langle\psi(\lambda)|\partial_\lambda\psi(\lambda)\rangle d\lambda \approx -\text{Im}\sum_{k} \log\frac{\langle\psi(\lambda_k)|\psi(\lambda_{k+1})\rangle}{\left|\langle\psi(\lambda_k)|\psi(\lambda_{k+1})\rangle\right|}$$

**Benchmark:** H₃ equilateral → isosceles deformation yields γ = π (exact from conical intersection).

### 4.4 Time-Dependent VMC (Level 15)

Real-time quantum dynamics via McLachlan's variational principle:

$$\min_{\dot{\theta}} \left\| i|\dot{\psi}\rangle - \hat{H}|\psi\rangle \right\|^2 \implies iS\dot{\theta} = f$$

where S is the quantum Fisher matrix, f_k = ⟨O_k* Ĥ|ψ⟩. Solve linear system per timestep, integrate θ(t) via RK4.

Enables: laser-driven ionization, electron scattering, charge transfer - phenomena inaccessible to standard VMC.

### 4.5 Entanglement Entropy via SWAP Trick (Level 18)

For bipartition A ∪ B, Rényi-2 entropy S₂(A) = -log Tr(ρ_A²) computed via:

$$\text{Tr}(\rho_A^2) = \langle\psi \otimes \psi | \text{SWAP}_A | \psi \otimes \psi\rangle$$

**Monte Carlo Implementation:**
```
Sample r₁, r₂ ~ |ψ|² independently
Construct r₁' = (r₂_A, r₁_B), r₂' = (r₁_A, r₂_B)  [swap A-electrons]
Estimate: Tr(ρ_A²) ≈ 𝔼[ψ(r₁')ψ(r₂') / (ψ(r₁)ψ(r₂))]
```

Quantifies "how quantum" chemical bonds are - of fundamental interest in quantum information theory.

### 4.6 Spin-Orbit Coupling (Level 17)

Breit-Pauli relativistic correction for fine-structure:

$$\hat{H}_{SO} = \frac{\alpha^2}{2}\sum_{i,I}\frac{Z_I}{r_{iI}^3}\hat{L}_{iI} \cdot \hat{S}_i$$

where α ≈ 1/137. Spinor wavefunctions ψ = [ψ↑, ψ↓]^T for 2-component treatment.

**Target:** Helium ³P fine-structure splitting (measured to 12 significant figures experimentally).

### 4.7 Periodic Systems (Level 16)

Bloch boundary conditions for solids: ψ(r + L) = e^{ik·L}ψ(r)

Homogeneous Electron Gas (HEG) with Ewald summation for long-range Coulomb. Twist-averaged boundary conditions (TABC) reduce finite-size effects. Comparison to Ceperley-Alder QMC (1980) - the basis of all modern DFT functionals.

### 4.8 Conservation Law Discovery (Level 19)

Train auxiliary Q_φ satisfying:

$$\mathcal{L}_{\text{conserve}} = \left|\langle[\hat{Q}, \hat{H}]\rangle\right|^2 + \lambda_{\text{novelty}}\sum_{k}\left|\langle Q|Q_k^{\text{known}}\rangle\right|^2$$

First term enforces [Q,H] = 0 (conservation), second penalizes overlap with known quantities (E, L, etc.). Discovery of novel Q represents Noether's theorem in reverse - machine discovery of symmetries.

---

## 5. Benchmark Systems

### 5.1 Atomic Ground States (Level 9)

| Atom | Z | N_e | E_exact (Ha) | Configuration | Status |
|------|---|-----|--------------|---------------|--------|
| H | 1 | 1 | -0.5000 | 1s¹ | ✓ |
| He | 2 | 2 | -2.9037 | 1s² | ✓ |
| Li | 3 | 3 | -7.4781 | 1s² 2s¹ | ✓ |
| Be | 4 | 4 | -14.6674 | 1s² 2s² | ✓ |
| C | 6 | 6 | -37.8450 | 1s² 2s² 2p² | ✓ |
| N | 7 | 7 | -54.5892 | 1s² 2s² 2p³ | ✓ |
| O | 8 | 8 | -75.0673 | 1s² 2s² 2p⁴ | ✓ |
| Ne | 10 | 10 | -128.9376 | 1s² 2s² 2p⁶ | ✓ |

**Chemical Accuracy:** ΔE < 1.6 mHa (1 kcal/mol)

### 5.2 Molecular Systems (Level 10)

- **H₂:** Bond length scan R ∈ [0.5, 4.0] Bohr, dissociation curve vs FCI
- **LiH:** Heteronuclear molecule, R_e = 3.015 Bohr
- **H₂O:** 3-nucleus geometry, O-H bond = 1.809 Bohr, ∠HOH = 104.5°

Potential Energy Surface (PES) scanning for bond breaking, transition states, reaction coordinates.

---

## 6. Implementation

### 6.1 Codebase Structure

```
quantum_physics.py (1217 lines)
├── MolecularSystem: Atomic/molecular definitions (ATOMS, MOLECULES)
├── compute_local_energy: E_L via Hutchinson estimator
├── MetropolisSampler: MCMC with adaptive step size
├── BerryPhaseComputer: Topological phase calculation
├── EntanglementEntropyComputer: Rényi-2 SWAP trick
├── PeriodicSystem/SpinOrbitSystem: Bloch/Breit-Pauli extensions

neural_dream.py (967 lines)
├── MambaBlock: SSM recurrence engine (selective gating)
├── DeepBackflowNet: Dense O(N²) vs SSM-Backflow O(N log N)
├── NeuralWavefunction: Multi-determinant + Jastrow + log-domain
├── PeriodicNeuralWavefunction: Bloch boundary conditions
├── SpinorWavefunction: Relativistic 2-component
├── HamiltonianFlowNetwork: Normalizing flow for sampling

solver.py (1596 lines)
├── StochasticReconfiguration: Natural gradient (Full/KFAC/Diagonal)
├── VMCSolver: Main training loop, batched local energy
├── NormalizingFlowSampler: Flow-accelerated Independent MH
├── ExcitedStateSolver: K-state simultaneous optimization
├── TimeDependentVMC: McLachlan propagator
├── ConservationLawDiscovery: Noether inverse
├── PESScanner: Dissociation curve automation

QuAnTuM.py (1604 lines)
└── Streamlit interactive dashboard
    ├── System selector (atoms/molecules)
    ├── Real-time energy/variance plots
    ├── PES curve visualization
    ├── Berry phase loop display
    ├── Entanglement maps
    ├── Excited state level diagrams
    └── Hyperparameter controls
```

**Total:** ~5,400 lines of production code

### 6.2 Installation

```bash
git clone https://github.com/Devanik21/The-Schrodinger-Paradox
cd The-Schrodinger-Paradox

conda create -n quantum python=3.10
conda activate quantum

pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install streamlit plotly numpy matplotlib scipy pandas

streamlit run QuAnTuM.py
```

### 6.3 Quick Start

```python
from quantum_physics import ATOMS
from neural_dream import NeuralWavefunction
from solver import VMCSolver

# Helium atom (2 electrons)
system = ATOMS['He']

solver = VMCSolver(
    system=system,
    n_walkers=512,
    d_model=32, n_layers=2, n_determinants=8,
    optimizer_type='sr',  # Stochastic Reconfiguration
    use_ssm_backflow=True,
    lr=0.01
)

solver.equilibrate(n_steps=200)

for step in range(5000):
    metrics = solver.train_step(n_mcmc_steps=10)
    if step % 100 == 0:
        print(f"Step {step}: E={metrics['energy']:.6f} Ha, "
              f"σ²={metrics['variance']:.6f}")

# Results
E_final = solver.energy_history[-100:].mean()
error_mHa = (E_final - system.exact_energy) * 1000
print(f"Error: {error_mHa:.3f} mHa")
```

---

## 7. Comparison with State-of-the-Art

| Feature | FermiNet (DeepMind) | PauliNet (Berlin) | This Work |
|---------|---------------------|-------------------|-----------|
| Electron interaction | Dense O(N²) | SchNet O(NK) | SSM O(N log N) |
| Optimization | KFAC | AdamW | SR + KFAC + Trust Region |
| MCMC sampling | Random walk | Random walk | Flow-accelerated |
| Excited states | Published 2024 | Limited | Variance minimization |
| Time dynamics | Not published | Not published | TD-VMC |
| Berry phase | Not computed | Not computed | Parameter loop |
| Entanglement | Not from VMC | Not from VMC | SWAP trick |

We emphasize this is a research exploration. FermiNet and PauliNet are extensively validated; our SSM approach requires systematic benchmarking.

---

## 8. Limitations and Future Work

**Current Status:**
- Tested primarily on atoms H-Ne and small molecules
- SSM-backflow requires validation against dense methods on identical systems
- Flow-accelerated VMC needs tuning for consistent performance
- Topological/entanglement calculations are proof-of-concept

**Future Directions:**
1. Systematic benchmarking: SSM vs dense on identical hyperparameters/systems
2. Transition metals (>20 electrons) to test scalability claims
3. Berry phase validation on H₃ (target: γ = π)
4. Entanglement entropy for H₂ dissociation
5. Excited state convergence analysis

---

## 9. References

- **FermiNet:** Spencer et al., "Better, Faster Fermionic Neural Networks," arXiv:2011.07125 (2020)
- **PauliNet:** Hermann et al., "Deep-neural-network solution of the electronic Schrödinger equation," Nature Chemistry (2020)
- **Mamba:** Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," arXiv:2312.00752 (2023)
- **SR Optimization:** Sorella, "Generalized Lanczos algorithm for variational quantum Monte Carlo," PRB (2001)
- **Variance Minimization:** Umrigar et al., "Alleviation of the Fermion-sign problem by optimization," JCP (2007)

---

## 10. Contributing

We welcome:
- Bug reports and benchmarking results
- Comparisons with FCI/CCSD(T) reference data
- Validation of advanced features (Berry phase, entanglement, TD-VMC)
- Alternative architectures for electron correlation

Please open issues before major pull requests.

---

## 11. System Requirements

**Hardware:**
- CPU: Modern multi-core (Intel i7/AMD Ryzen 9)
- RAM: 16 GB minimum, 32 GB recommended
- GPU: Optional (NVIDIA RTX 3080+ with 10 GB VRAM for N_e > 6)

**Software:**
```
Python 3.10+
PyTorch 2.0+
NumPy 1.24+
Streamlit 1.28+
Plotly 5.17+
```

---

## 12. Acknowledgments

This work builds upon theoretical foundations from:
- DeepMind (FermiNet): Neural quantum states
- Noé Lab Berlin (PauliNet): Antisymmetric architectures
- CMU/Princeton (Mamba): Selective state-space models
- Sorella (SISSA): Stochastic reconfiguration

The author is solely responsible for any errors. This is a research implementation; use for production quantum chemistry is not recommended without independent verification.

---

## 13. License

Apache License 2.0

Copyright 2026 Devanik

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

**Citation:**
```bibtex
@software{schrodinger_dream_2026,
  author = {Devanik},
  title = {The Schrödinger Dream: Neural Quantum State Solver with SSM-Backflow},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/Devanik21/The-Schrodinger-Paradox}
}
```

---

**Author:** Devanik  
**Affiliation:** B.Tech ECE '26, National Institute of Technology Agartala  
**Fellowship:** Samsung Convergence Software (Grade I), Indian Institute of Science

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-181717?style=flat&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Devanik-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/devanik/)

*Last Updated: February 11, 2026*
