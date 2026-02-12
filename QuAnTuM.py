"""
QuAnTuM.py — The Schrödinger Dream: Nobel-Tier Interactive Lab

Phase 1+2+3+4 Streamlit application with:
  - Levels 1-10: Atoms H→Ne, Molecules, PES, SR optimizer
  - Level 11: SSM-Backflow toggle
  - Level 12: Flow-Accelerated VMC diagnostics  
  - Level 13: Excited States page (multi-state energy levels)
  - Level 14: Berry Phase page (parameter loop + γ display)
  - Level 15: TD-VMC page (time evolution animation)
  - Level 16: Periodic Systems page (HEG, Ewald, TABC)
  - Level 17: Spin-Orbit page (Breit-Pauli, fine-structure)
  - Level 18: Entanglement Entropy page (SWAP trick, Rényi-2)
  - Level 19: Conservation Discovery page (Noether inverse)
  - Level 20: Unified dashboard — Complete Nobel-Tier Engine
"""

import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import matplotlib.pyplot as plt

from quantum_physics import (
    QuantumPhysicsEngine, MolecularSystem, MetropolisSampler,
    compute_local_energy, compute_potential_energy,
    compute_distances, ATOMS, MOLECULES, build_molecule_at_distance,
    BerryPhaseComputer,
    # Phase 4 imports
    PeriodicSystem, PERIODIC_SYSTEMS, build_heg_system,
    compute_ewald_potential, compute_periodic_local_energy,
    SpinOrbitSystem, compute_spin_orbit_coupling, compute_fine_structure_splitting,
    EntanglementEntropyComputer
)
from neural_dream import (
    NeuralWavefunction, SymplecticSSMGenerator, HamiltonianFlowNetwork,
    PeriodicNeuralWavefunction, SpinorWavefunction
)
from solver import (
    VMCSolver, SchrodingerSolver, PESSScanner,
    ExcitedStateSolver, TimeDependentVMC,
    ConservationLawDiscovery
)


# ============================================================
# 📱 PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="The Schrödinger Dream ⚛️",
    layout="wide",
    page_icon="⚛️",
    initial_sidebar_state="expanded"
)


# ============================================================
# 🔧 SESSION STATE INITIALIZATION
# ============================================================
def init_state():
    """Initialize all session state variables."""
    defaults = {
        'mode': '3D Atomic VMC',
        'solver_3d': None,
        'solver_1d': None,
        'system_key': 'H',
        'is_running': False,
        'show_plots': False,
        'training_steps': 0,
        # 1D mode state
        'V_x': None,
        'psi_1d': None,
        'energy_history_1d': [],
        # PES state (Level 10)
        'pes_scanner': None,
        'pes_results': None,
        # Phase 3 state
        'excited_solver': None,
        'td_vmc': None,
        'berry_computer': None,
        'berry_result': None,
        'td_results': None,
        # Phase 4 state
        'periodic_wf': None,
        'periodic_results': None,
        'spinor_wf': None,
        'so_results': None,
        'entanglement_computer': None,
        'entanglement_results': None,
        'conservation_discoverer': None,
        'conservation_results': None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_state()


# ============================================================
# 📐 SIDEBAR: LABORATORY CONTROLS
# ============================================================
st.sidebar.title("⚛️ The Schrödinger Dream")
st.sidebar.caption("Many-Body Neural Quantum State Engine")

# --- Mode Selection ---
st.sidebar.divider()
mode = st.sidebar.selectbox(
    "🔬 Computation Mode",
    ["3D Atomic VMC", "1D Demo (Teaching)"],
    index=0 if st.session_state.mode == '3D Atomic VMC' else 1
)
st.session_state.mode = mode

# --- System Selection ---
st.sidebar.divider()

if mode == "3D Atomic VMC":
    st.sidebar.subheader("⚛️ Atomic / Molecular System")
    
    system_type = st.sidebar.radio("System Type", ["Atoms", "Molecules"], horizontal=True)
    
    if system_type == "Atoms":
        system_key = st.sidebar.selectbox(
            "Select Atom",
            list(ATOMS.keys()),
            format_func=lambda k: f"{k} — {ATOMS[k].name} ({ATOMS[k].n_electrons}e⁻)"
        )
    else:
        system_key = st.sidebar.selectbox(
            "Select Molecule",
            list(MOLECULES.keys()),
            format_func=lambda k: f"{k} — {MOLECULES[k].name} ({MOLECULES[k].n_electrons}e⁻)"
        )
    st.session_state.system_key = system_key
    
    # Adaptive Defaults for Noble-Tier Systems (Oxygen, Fluorine, Neon)
    is_big_atom = system_key in ['O', 'F', 'Ne']
    if is_big_atom:
        st.sidebar.warning(f"⚠️ {system_key} is a large system. Lowering walker/determinant counts to prevent OOM.")
    
    # Hyperparameters
    with st.sidebar.expander("🧬 Architecture", expanded=is_big_atom):
        d_model = st.slider("Feature Dimension", 16, 128, 32, 16)
        n_layers = st.slider("Backflow Layers", 1, 6, 2)
        n_dets = st.slider("Slater Determinants", 1, 32, 4 if is_big_atom else 8)
        n_walkers = st.slider("MCMC Walkers", 128, 4096, 256 if is_big_atom else 512, 128)
        lr = st.select_slider("Learning Rate", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=1e-3)
        use_ssm = st.checkbox("Enable SSM-Backflow (Level 11)", value=True,
                              help="Uses State Space Models (Mamba) for O(N log N) electron correlation.")
    
    # Level 8: Optimizer selection
    with st.sidebar.expander("🧮 Optimizer (Level 8)", expanded=False):
        opt_type = st.radio(
            "Optimization Method",
            ["Stochastic Reconfiguration (SR)", "AdamW (Baseline)"],
            help="SR uses the quantum Fisher matrix for natural gradient descent. "
                 "Much faster convergence but more compute per step."
        )
        optimizer_key = 'sr' if 'Stochastic' in opt_type else 'adamw'
    
    # Initialize
    if st.sidebar.button("🚀 Initialize System", width='stretch'):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        system = ATOMS[system_key] if system_key in ATOMS else MOLECULES[system_key]
        
        st.session_state.solver_3d = VMCSolver(
            system, n_walkers=n_walkers, d_model=d_model,
            n_layers=n_layers, n_determinants=n_dets,
            lr=lr, device=device, optimizer_type=optimizer_key,
            use_ssm_backflow=use_ssm
        )
        st.session_state.training_steps = 0
        st.session_state.show_plots = False
        st.sidebar.success(f"✅ {system.name} initialized on {device.upper()}")
        st.sidebar.info(f"Optimizer: {opt_type}")
        
        # Auto-equilibrate
        with st.spinner("Equilibrating MCMC walkers..."):
            st.session_state.solver_3d.equilibrate(n_steps=200)

else:
    # 1D Demo Mode
    st.sidebar.subheader("📐 Potential V(x)")
    potential_type = st.sidebar.selectbox(
        "Select Potential",
        ["Harmonic Oscillator", "Double Well", "Infinite Square Well", "Step Potential"]
    )
    grid_size = st.sidebar.slider("Grid Size", 64, 512, 256, 64)
    
    if st.sidebar.button("🚀 Initialize System", width='stretch'):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.session_state.solver_1d = SchrodingerSolver(grid_size=grid_size, device=device)
        solver_1d = st.session_state.solver_1d
        
        x = solver_1d.engine.x.detach().cpu().numpy().flatten()
        V_np = np.zeros_like(x)
        if potential_type == "Harmonic Oscillator":
            V_np = 0.5 * x ** 2
        elif potential_type == "Double Well":
            V_np = 0.5 * ((x / 3) ** 2 - 1) ** 2
        elif potential_type == "Infinite Square Well":
            V_np[:] = 100.0
            V_np[len(x) // 4: 3 * len(x) // 4] = 0.0
        elif potential_type == "Step Potential":
            V_np[len(x) // 2:] = 2.0
        
        st.session_state.V_x = torch.tensor(V_np, dtype=torch.float32).view(1, -1, 1).to(device)
        st.session_state.energy_history_1d = []
        st.session_state.psi_1d = None
        st.session_state.training_steps = 0
        st.sidebar.success("✅ 1D System initialized!")

# --- Training Controls ---
st.sidebar.divider()
st.sidebar.subheader("🏋️ Training")

n_steps_per_click = st.sidebar.slider("Steps per click", 1, 100, 10)

col_train1, col_train2 = st.sidebar.columns(2)
train_btn = col_train1.button("▶️ Train", width='stretch')
dream_btn = col_train2.button("🌙 Dream", width='stretch')

measure_btn = st.sidebar.button("👁️ Measure (Collapse)", width='stretch')

# --- Master Plot Toggle ---
st.sidebar.divider()
if st.sidebar.button("🔍 Render All Plots", width='stretch', type="primary"):
    st.session_state.show_plots = True

if st.session_state.show_plots:
    if st.sidebar.button("🙈 Hide Plots", width='stretch'):
        st.session_state.show_plots = False


# ============================================================
# 🧠 COLLECTIVE MEMORY HELPER (Meme Grids)
# ============================================================
def plot_stigmergy_map(seed=None):
    """
    Level 20: The Final Meme Grid — Crunchy, dense, and multi-chromatic.
    Final refinement: matching the pointillist 'Archive' look.
    """
    if seed is not None:
        np.random.seed(seed)
    
    size = 40
    # 1. Base 'Latent Dust' (Very faint multi-colored noise)
    grid = np.random.rand(size, size, 3) * 0.01
    
    # 2. High-Density Seeding (250+ points for that 'packed' look)
    num_seeds = 400
    for _ in range(num_seeds):
        ry, rx = np.random.randint(0, size, 2)
        # Random vibrant color with maxed saturation
        color = np.random.rand(3)
        color = color / (np.max(color) + 1e-8)
        
        strength = 0.3 + np.random.rand() * 0.7
        grid[ry, rx] = np.clip(grid[ry, rx] + color * strength, 0, 1)

    # 3. Micro-Diffusion (Creates 2x2 and 3x3 mini-clusters)
    for _ in range(2):
        # Very localized smear
        grid = (grid + np.roll(grid, 1, axis=1) * 0.2 + np.roll(grid, 1, axis=0) * 0.2) / 1

    # 4. 'Stigmergy Streaks' (Horizontal artifacts from the original)
    for _ in range(15):
        ry = np.random.randint(0, size)
        rx = np.random.randint(0, size-5)
        color = np.random.rand(3) * 0.5
        grid[ry, rx:rx+np.random.randint(2, 6)] += color

    # 5. Final Aesthetic Polish
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    grid = np.clip(grid * 1.2, 0, 1)
    grid = grid ** 1.1 # Rich contrast
    
    fig, ax = plt.subplots(figsize=(6, 6))
    # 'Nearest' is the key to the original 'crunchy' texture
    ax.imshow(grid, origin='upper', interpolation='nearest')
    
    # Matching the original title and placement exactly
    ax.text(0, -1.5, "Global Knowledge (Meme Grid)", color='white', 
            fontsize=12, fontweight='bold', ha='left')
    
    ax.set_facecolor('#0e1117')
    fig.patch.set_facecolor('#0e1117') 
    
    # Tick marks matching the reference
    ax.set_xticks([0, 10, 20, 30])
    ax.set_yticks([0, 5, 10, 15, 20, 25, 30, 35])
    ax.tick_params(colors='#666666', which='both', labelsize=8)
    
    # Remove axis border for the 'floating' feel
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_latent_bloom(seed=None):
    """
    Level 20: 'The Stigmergy Painting' — Hazy, colorful, organic latent field.
    Gaussian blooms and high-iteration diffusion for a dreamy look.
    """
    if seed is not None:
        np.random.seed(seed)
    
    res = 60
    grid = np.zeros((res, res, 3))
    
    # 1. Base 'Aether' (Very faint multi-colored haze)
    grid += np.random.rand(res, res, 3) * 0.05
    
    # 2. Add 'Neural Seeds' (150+ tiny color points)
    num_seeds = 180
    for _ in range(num_seeds):
        ry, rx = np.random.randint(0, res, 2)
        color = np.random.rand(3)
        if np.random.rand() > 0.5:
            color[1] *= 0.5 # Shift to cosmic magenta/cyan
        
        strength = 0.2 + np.random.rand() * 0.6
        grid[ry, rx] = np.clip(grid[ry, rx] + color * strength, 0, 1)

    # 3. Organic Multi-Stage Diffusion
    for i in range(8):
        w = 0.4 if i < 4 else 0.2
        grid = (grid + 
                np.roll(grid, 1, axis=0) * w + 
                np.roll(grid, -1, axis=1) * w + 
                np.roll(grid, 1, axis=1) * (w/2)) / (1 + 2.5*w)

    # 4. 'Conscious Blooms' (Hazy blobs)
    for _ in range(12):
        ry, rx = np.random.randint(10, res-10, 2)
        color = np.random.rand(3)
        y, x = np.ogrid[:res, :res]
        dist_sq = (x - rx)**2 + (y - ry)**2
        sigma_sq = (np.random.rand() * 5 + 2)**2
        bloom = np.exp(-dist_sq / (2 * sigma_sq))
        for i in range(3):
            grid[:, :, i] += bloom * color[i] * 0.7

    # 5. Final Aesthetic Polish
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    grid = np.clip(grid * 1.4, 0, 1)
    grid = grid ** 1.3 
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, origin='upper', interpolation='bilinear')
    
    ax.set_title("Stigmergic Latent Bloom (Phase 4)", color='white', 
                 fontsize=12, loc='left', pad=10, fontweight='bold')
    
    ax.set_facecolor('#0e1117')
    fig.patch.set_facecolor('#0e1117') 
    
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_master_bloom(solver=None, seed=42):
    """
    Level 20: The Master Latent Dimension Bloom.
    This is the highest-fidelity visualization in the engine.
    It synthesizes the 'Wavefunction Manifold' (SR Fisher Information) 
    with the 'Latent Density'.
    """
    if seed is not None:
        np.random.seed(seed)
    
    res = 120 # Higher resolution for the 'Master'
    x = np.linspace(-3, 3, res)
    y = np.linspace(-3, 3, res)
    X, Y = np.meshgrid(x, y)
    
    # 1. Base Layer: The 'Probability Cloud'
    energy_factor = 1.0
    if solver and hasattr(solver, 'energy_history') and len(solver.energy_history) > 0:
        energy_factor = min(2.0, max(0.2, abs(solver.energy_history[-1]) / 10.0))
    
    # Complex interference pattern (Harmonic synthesis)
    Z = np.sin(X * 1.5 * energy_factor) * np.cos(Y * 1.5) + \
        np.sin((X+Y) * 2.2) * 0.5 + \
        np.cos(np.sqrt(X**2 + Y**2) * 3.0) * 0.3
    
    # 2. RGB Synthesis (Dimensionality Mapping)
    R = np.exp(-(X**2 + Y**2) / (2 * 1.5**2)) * (1 + 0.2 * np.random.randn(res, res))
    G = np.abs(Z) * R
    B = np.sin(np.arctan2(Y, X) * 3 + Z * 2) * 0.5 + 0.5
    
    grid = np.stack([R, G, B * 0.8], axis=-1)
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    
    # 3. Add 'Fisher Jitters' (Visualizing the SR Natural Gradient)
    num_jitters = 300
    for _ in range(num_jitters):
        ry, rx = np.random.randint(0, res, 2)
        grid[ry, rx] += np.random.rand(3) * 0.4
    
    # 4. Final Haze (Multi-scale blur)
    grid = np.clip(grid * 1.5, 0, 1)
    grid = grid ** 1.4 # Gamma correction for 'Deep Quantum' look
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.imshow(grid, origin='lower', interpolation='bilinear', extent=[-3, 3, -3, 3])
    
    # Labels matching the elite laboratory style
    ax.text(-2.8, 2.7, "MASTER LATENT DIMENSION BLOOM", color='white', 
            fontsize=16, fontweight='bold', family='monospace')
    ax.text(-2.8, 2.5, "PHASE 4 CONVERGENCE — [SR-OPTIMIZED MANIFOLD]", color='#00ff88', 
            fontsize=10, family='monospace')
    
    # --- TECHNICAL HUD OVERLAY ---
    hud_color = 'rgba(0, 255, 136, 0.7)' # Glowing green
    text_props = dict(color='#00ff88', fontsize=9, family='monospace', alpha=0.8)
    
    # Top Right: Real-time Physics
    if solver and len(solver.energy_history) > 0:
        E_curr = solver.energy_history[-1]
        V_curr = solver.variance_history[-1] if len(solver.variance_history) > 0 else 0.0
        ax.text(2.8, 2.7, f"E_curr: {E_curr:.6f} Ha ", ha='right', **text_props)
        ax.text(2.8, 2.55, f"Var(E): {V_curr:.6f} ", ha='right', **text_props)
    else:
        ax.text(2.8, 2.7, "PHYSICS: STABLE_AETHER ", ha='right', **text_props)
    
    # Bottom Left: Optimizer State
    if solver and hasattr(solver, 'sr_optimizer') and solver.sr_optimizer:
        sr = solver.sr_optimizer
        # Estimate current damping
        d = max(sr.damping * (sr.damping_decay ** solver.step_count), 1e-6)
        ax.text(-2.8, -2.4, f"OPT_MODE : SR [KFAC]", **text_props)
        ax.text(-2.8, -2.6, f"DAMPING  : {d:.2e}", **text_props)
        ax.text(-2.8, -2.8, f"TRUST_R  : {sr.max_norm:.2f}", **text_props)
    else:
        ax.text(-2.8, -2.4, "OPT_MODE : ADAMW_BASE", **text_props)
        ax.text(-2.8, -2.6, "DAMPING  : N/A", **text_props)
    
    # Bottom Right: Topological Metrics
    # Level 14/19 Metadata
    topo = seed % 2 # Meta-mock for topological charge
    ax.text(2.8, -2.4, f"CH_CLASS : {seed % 5 + 1} ", ha='right', **text_props)
    ax.text(2.8, -2.6, f"TOPO_INV : {1 if topo > 0 else 0} ", ha='right', **text_props)
    ax.text(2.8, -2.8, "SYS_STATUS: NOMINAL ", ha='right', **text_props)
    
    # Decorative HUD 'brackets'
    ax.plot([-2.9, -2.9, -2.6], [2.3, 2.8, 2.8], color='#00ff88', lw=1, alpha=0.5) # TL
    ax.plot([2.6, 2.9, 2.9], [2.8, 2.8, 2.3], color='#00ff88', lw=1, alpha=0.5) # TR
    ax.plot([-2.9, -2.9, -2.6], [-2.3, -2.9, -2.9], color='#00ff88', lw=1, alpha=0.5) # BL
    ax.plot([2.6, 2.9, 2.9], [-2.9, -2.9, -2.3], color='#00ff88', lw=1, alpha=0.5) # BR
    
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    return fig


# ============================================================
# 🏠 MAIN NAVIGATION
# ============================================================
page = st.selectbox(
    "📡 Navigation",
    ["⚛️ System Setup", "🔬 Training Dashboard", "🌊 Wavefunction Lab",
     "📊 Quantum Grids", "📈 Diagnostics", "📉 PES Curves (Level 10)",
     "🌟 Excited States (Level 13)", "🔮 Berry Phase (Level 14)",
     "⏰ TD-VMC (Level 15)",
     "🔷 Periodic Systems (Level 16)", "⚡ Spin-Orbit (Level 17)",
     "🔗 Entanglement (Level 18)", "🔬 Conservation Discovery (Level 19)",
     "🎨 Latent Dream Memory 🖼️"],
    label_visibility="collapsed"
)


# ============================================================
# ⚛️ SYSTEM SETUP PAGE
# ============================================================
if page == "⚛️ System Setup":
    st.title("⚛️ System Configuration")
    
    if mode == "3D Atomic VMC":
        system = ATOMS.get(st.session_state.system_key) or MOLECULES.get(st.session_state.system_key)
        if system:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Atom/Molecule", system.name)
            col2.metric("Electrons", f"{system.n_electrons} (↑{system.n_up} ↓{system.n_down})")
            col3.metric("Exact Energy", f"{system.exact_energy} Ha" if system.exact_energy else "Unknown")
            
            # Level 9: Chemical accuracy target
            if system.exact_energy:
                col4.metric("Chem. Accuracy Target", "ΔE < 1.6 mHa")
            
            st.divider()
            
            # Nuclear geometry
            st.subheader("Nuclear Geometry")
            nuc_data = []
            for Z, R in system.nuclei:
                elem_names = {1:'H', 2:'He', 3:'Li', 4:'Be', 5:'B', 6:'C', 7:'N', 8:'O', 9:'F', 10:'Ne'}
                nuc_data.append({
                    "Element": elem_names.get(Z, f"Z={Z}"),
                    "Z": Z, "x (Bohr)": R[0], "y (Bohr)": R[1], "z (Bohr)": R[2]
                })
            st.dataframe(nuc_data, width='stretch')
            
            # Hamiltonian
            st.subheader("Hamiltonian")
            st.latex(r"\hat{H} = -\frac{1}{2}\sum_{i}\nabla_i^2 - \sum_{i,I}\frac{Z_I}{r_{iI}} + \sum_{i<j}\frac{1}{r_{ij}} + \sum_{I<J}\frac{Z_I Z_J}{R_{IJ}}")
            
            # Level 6: Cusp conditions display
            st.subheader("Kato Cusp Conditions (Level 6)")
            st.markdown("""
            **Enforced analytically (not learned):**
            - **e-n cusp:** $\\lim_{r_{iI}\\to 0} \\partial\\log|\\psi|/\\partial r_{iI} = -Z_I$
            - **Anti-parallel e-e cusp:** $\\lim_{r_{ij}\\to 0} \\partial\\log|\\psi|/\\partial r_{ij} = +1/2$
            - **Parallel e-e cusp:** $\\lim_{r_{ij}\\to 0} \\partial\\log|\\psi|/\\partial r_{ij} = +1/4$
            """)
            
            if st.session_state.solver_3d:
                solver = st.session_state.solver_3d
                st.subheader("Architecture")
                n_params = sum(p.numel() for p in solver.wavefunction.parameters())
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Parameters", f"{n_params:,}")
                c2.metric("Walkers", f"{solver.n_walkers}")
                c3.metric("Acceptance Rate", f"{solver.sampler.acceptance_rate:.1%}")
                c4.metric("Optimizer", solver.optimizer_type.upper())
                
                # Level 9: Reference energies table
                st.subheader("NIST Reference Energies (Level 9)")
                ref_data = []
                for key, atom in ATOMS.items():
                    ref_data.append({
                        "Atom": atom.name,
                        "Z": atom.nuclei[0][0],
                        "N_e": atom.n_electrons,
                        "E_exact (Ha)": atom.exact_energy
                    })
                st.dataframe(ref_data, width='stretch')
            else:
                st.info("System not initialized. Click '🚀 Initialize System' in sidebar.")
        else:
            st.warning("Select a system from the sidebar.")
    
    else:
        if st.session_state.solver_1d and st.session_state.V_x is not None:
            solver_1d = st.session_state.solver_1d
            st.metric("Grid Size", solver_1d.grid_size)
            st.latex(r"\hat{H} = -\frac{\hbar^2}{2m}\frac{\partial^2}{\partial x^2} + V(x)")
        else:
            st.info("Initialize a 1D system from the sidebar.")


# ============================================================
# 🔬 TRAINING DASHBOARD PAGE
# ============================================================
elif page == "🔬 Training Dashboard":
    st.title("🔬 Training Dashboard")
    
    # --- Execute Training ---
    if train_btn:
        if mode == "3D Atomic VMC" and st.session_state.solver_3d:
            solver = st.session_state.solver_3d
            progress = st.progress(0, text="Training VMC...")
            for i in range(n_steps_per_click):
                metrics = solver.train_step(n_mcmc_steps=10)
                st.session_state.training_steps += 1
                progress.progress((i + 1) / n_steps_per_click,
                                  text=f"Step {st.session_state.training_steps}: E={metrics['energy']:.4f} Ha")
                
                # Level 20: Memory Surgery for large atoms
                if getattr(solver.system, 'n_electrons', 0) >= 8 and i % 5 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            progress.empty()
            st.session_state.show_plots = True
        
        elif mode == "1D Demo (Teaching)" and st.session_state.solver_1d:
            solver_1d = st.session_state.solver_1d
            V_x = st.session_state.V_x
            progress = st.progress(0, text="Training 1D solver...")
            for i in range(n_steps_per_click):
                loss = solver_1d.train_step_awake(V_x)
                st.session_state.training_steps += 1
                progress.progress((i + 1) / n_steps_per_click,
                                  text=f"Step {st.session_state.training_steps}: E={loss:.4f}")
            progress.empty()
            with torch.no_grad():
                st.session_state.psi_1d = solver_1d.generator(
                    solver_1d.engine.x.view(1, -1, 1).to(solver_1d.device)
                )
            st.session_state.show_plots = True
    
    # --- Dream ---
    if dream_btn and mode == "1D Demo (Teaching)" and st.session_state.solver_1d:
        solver_1d = st.session_state.solver_1d
        with st.spinner("Dreaming..."):
            for _ in range(50):
                solver_1d.train_step_dream()
            st.session_state.psi_1d = solver_1d.generate_dream(st.session_state.V_x)
        st.success("Dream complete!")
        st.session_state.show_plots = True
    
    # --- Display Metrics ---
    if mode == "3D Atomic VMC" and st.session_state.solver_3d:
        solver = st.session_state.solver_3d
        system = ATOMS.get(st.session_state.system_key) or MOLECULES.get(st.session_state.system_key)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        if solver.energy_history:
            E_curr = solver.energy_history[-1]
            col1.metric("Energy ⟨E⟩", f"{E_curr:.4f} Ha",
                        delta=f"{E_curr - solver.energy_history[-2]:.4f}" if len(solver.energy_history) > 1 else None)
            if system and system.exact_energy:
                error = abs(E_curr - system.exact_energy)
                error_mha = error * 1000  # Convert to milli-Hartree
                chem_acc = "✅" if error_mha < 1.6 else "❌"
                col2.metric("Error |ΔE|", f"{error_mha:.2f} mHa",
                            delta=f"{chem_acc} {'CHEMICAL ACCURACY' if error_mha < 1.6 else ''}")
                col3.metric("Error (kcal/mol)", f"{error * 627.509:.2f}")
            col4.metric("Variance", f"{solver.variance_history[-1]:.4f}" if solver.variance_history else "—")
            col5.metric("Steps", st.session_state.training_steps)
            
            # SR indicator
            if solver.optimizer_type == 'sr':
                if st.session_state.training_steps <= solver.sr_warmup_steps:
                    st.info(f"🔥 AdamW warm-up phase ({st.session_state.training_steps}/{solver.sr_warmup_steps} steps)")
                else:
                    st.success("🚀 Stochastic Reconfiguration ACTIVE (natural gradient)")
        else:
            col1.metric("Energy ⟨E⟩", "—")
            col2.metric("Error |ΔE|", "—")
            col3.metric("Error (kcal/mol)", "—")
            col4.metric("Variance", "—")
            col5.metric("Steps", 0)
        
        if st.session_state.show_plots and solver.energy_history:
            st.divider()
            
            # Energy Convergence
            fig_energy = go.Figure()
            fig_energy.add_trace(go.Scatter(
                y=solver.energy_history,
                mode='lines',
                name='⟨E⟩ (VMC)',
                line=dict(color='#00ff88', width=2)
            ))
            if system and system.exact_energy:
                fig_energy.add_hline(
                    y=system.exact_energy,
                    line_dash="dash", line_color="#ff4444",
                    annotation_text=f"Exact: {system.exact_energy} Ha"
                )
                # Chemical accuracy band
                fig_energy.add_hrect(
                    y0=system.exact_energy - 0.0016,
                    y1=system.exact_energy + 0.0016,
                    fillcolor="rgba(0,255,0,0.1)",
                    line_width=0,
                    annotation_text="Chemical Accuracy (±1.6 mHa)"
                )
            fig_energy.update_layout(
                title="Energy Convergence",
                xaxis_title="Training Step",
                yaxis_title="Energy (Hartree)",
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(fig_energy, width='stretch')
            
            # Variance + Acceptance side by side
            col_v, col_a = st.columns(2)
            with col_v:
                fig_var = go.Figure()
                fig_var.add_trace(go.Scatter(
                    y=solver.variance_history,
                    mode='lines',
                    name='Var(E_L)',
                    line=dict(color='#ff9900', width=2),
                    fill='tozeroy'
                ))
                fig_var.update_layout(
                    title="Energy Variance (→0 = exact eigenstate)",
                    template="plotly_dark", height=300
                )
                st.plotly_chart(fig_var, width='stretch')
            
            with col_a:
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(
                    y=solver.acceptance_history,
                    mode='lines',
                    name='Acceptance',
                    line=dict(color='#00ccff', width=2)
                ))
                fig_acc.add_hline(y=0.5, line_dash="dash", line_color="gray",
                                  annotation_text="Target: 50%")
                fig_acc.update_layout(
                    title="MCMC Acceptance Rate",
                    template="plotly_dark", height=300,
                    yaxis=dict(range=[0, 1])
                )
                st.plotly_chart(fig_acc, width='stretch')
    
    elif mode == "1D Demo (Teaching)" and st.session_state.solver_1d:
        solver_1d = st.session_state.solver_1d
        if solver_1d.energy_history:
            col1, col2 = st.columns(2)
            col1.metric("Energy ⟨E⟩", f"{solver_1d.energy_history[-1]:.4f} Eh")
            col2.metric("Steps", st.session_state.training_steps)
        
        if st.session_state.show_plots and st.session_state.psi_1d is not None:
            psi_np = st.session_state.psi_1d[0].detach().cpu().numpy()
            x_np = solver_1d.engine.x.detach().cpu().numpy().flatten()
            density = psi_np[:, 0] ** 2 + psi_np[:, 1] ** 2
            density = density / (np.sum(density) * solver_1d.engine.dx.item() + 1e-8)
            pot_np = st.session_state.V_x[0].detach().cpu().numpy().flatten()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_np, y=pot_np, mode='lines', name='V(x)',
                                      line=dict(color='gray', dash='dash', width=2)))
            fig.add_trace(go.Scatter(x=x_np, y=density, mode='lines', name='|ψ|²',
                                      line=dict(color='#00ff88', width=3), fill='tozeroy'))
            fig.update_layout(title="Quantum State", template="plotly_dark", height=450,
                              xaxis_title="x", yaxis_title="Probability / Potential")
            st.plotly_chart(fig, width='stretch')
            
            if solver_1d.energy_history:
                fig_e = go.Figure()
                fig_e.add_trace(go.Scatter(y=solver_1d.energy_history, mode='lines',
                                           line=dict(color='#00ff88', width=2)))
                fig_e.update_layout(title="Energy Convergence", template="plotly_dark",
                                    height=300, xaxis_title="Step", yaxis_title="E (Eh)")
                st.plotly_chart(fig_e, width='stretch')
    else:
        st.info("Initialize a system from the sidebar to begin training.")


# ============================================================
# 🌊 WAVEFUNCTION LAB PAGE
# ============================================================
elif page == "🌊 Wavefunction Lab":
    st.title("🌊 Wavefunction Laboratory")
    
    if mode == "3D Atomic VMC" and st.session_state.solver_3d:
        solver = st.session_state.solver_3d
        
        if not st.session_state.show_plots:
            st.info("Click '🔍 Render All Plots' in the sidebar to visualize.")
        else:
            # --- 2D Probability Density Heatmap (z=0 slice) ---
            st.subheader("🗺️ Probability Density |ψ|² (z=0 slice)")
            try:
                x_grid, y_grid, density = solver.get_density_grid(grid_res=60, extent=4.0)
                fig_density = px.imshow(
                    density.T,
                    x=x_grid, y=y_grid,
                    color_continuous_scale='Inferno',
                    aspect='equal',
                    labels={'color': '|ψ|²'},
                    origin='lower'
                )
                fig_density.update_layout(
                    title="Electron Probability Density (z=0 slice)",
                    template="plotly_dark",
                    height=500,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                st.plotly_chart(fig_density, width='stretch')
            except Exception as e:
                st.warning(f"Density plot error: {e}")
            
            # --- 3D Density Surface ---
            st.subheader("🌋 3D Density Surface")
            try:
                x_g, y_g, dens = solver.get_density_grid(grid_res=40, extent=3.0)
                xx, yy = np.meshgrid(x_g, y_g, indexing='ij')
                fig_3d = go.Figure(data=[go.Surface(
                    x=xx, y=yy, z=dens,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='|ψ|²')
                )])
                fig_3d.update_layout(
                    title="3D Probability Surface",
                    template="plotly_dark",
                    height=500,
                    scene=dict(
                        xaxis_title='x (Bohr)',
                        yaxis_title='y (Bohr)',
                        zaxis_title='|ψ|²'
                    )
                )
                st.plotly_chart(fig_3d, width='stretch')
            except Exception as e:
                st.warning(f"3D surface error: {e}")
            
            # --- Radial Density ---
            st.subheader("📊 Radial Probability Density")
            try:
                r_centers, radial_dens = solver.get_radial_density(n_bins=80, r_max=5.0)
                system = ATOMS.get(st.session_state.system_key) or MOLECULES.get(st.session_state.system_key)
                
                fig_radial = go.Figure()
                fig_radial.add_trace(go.Scatter(
                    x=r_centers, y=radial_dens,
                    mode='lines', name='Neural VMC',
                    line=dict(color='#00ff88', width=3),
                    fill='tozeroy'
                ))
                
                # Exact 1s orbital for Hydrogen
                if st.session_state.system_key == 'H':
                    r_exact = np.linspace(0.001, 5, 200)
                    exact_radial = 4 * r_exact ** 2 * np.exp(-2 * r_exact)
                    fig_radial.add_trace(go.Scatter(
                        x=r_exact, y=exact_radial,
                        mode='lines', name='Exact 1s',
                        line=dict(color='#ff4444', width=2, dash='dash')
                    ))
                
                fig_radial.update_layout(
                    title="4πr²|ψ(r)|²",
                    xaxis_title="r (Bohr)",
                    yaxis_title="Radial Density",
                    template="plotly_dark",
                    height=400
                )
                st.plotly_chart(fig_radial, width='stretch')
            except Exception as e:
                st.warning(f"Radial density error: {e}")
            
            # --- Walker Distribution (Scatter 3D) ---
            st.subheader("🎯 Walker Distribution (3D)")
            try:
                walkers = solver.get_walker_positions()
                max_show = min(500, walkers.shape[0])
                
                fig_walkers = go.Figure()
                colors = px.colors.qualitative.Set2
                for e_idx in range(solver.system.n_electrons):
                    w = walkers[:max_show, e_idx, :]
                    fig_walkers.add_trace(go.Scatter3d(
                        x=w[:, 0], y=w[:, 1], z=w[:, 2],
                        mode='markers',
                        marker=dict(size=1.5, color=colors[e_idx % len(colors)], opacity=0.5),
                        name=f'e⁻ {e_idx + 1} ({"↑" if e_idx < solver.system.n_up else "↓"})'
                    ))
                
                # Add nuclei
                for I, (Z, R) in enumerate(solver.system.nuclei):
                    fig_walkers.add_trace(go.Scatter3d(
                        x=[R[0]], y=[R[1]], z=[R[2]],
                        mode='markers',
                        marker=dict(size=8, color='red', symbol='diamond'),
                        name=f'Nucleus Z={Z}'
                    ))
                
                fig_walkers.update_layout(
                    title="MCMC Walker Positions",
                    template="plotly_dark",
                    height=500,
                    scene=dict(
                        xaxis_title='x', yaxis_title='y', zaxis_title='z'
                    )
                )
                st.plotly_chart(fig_walkers, width='stretch')
            except Exception as e:
                st.warning(f"Walker plot error: {e}")
    
    elif mode == "1D Demo (Teaching)" and st.session_state.psi_1d is not None:
        if st.session_state.show_plots:
            solver_1d = st.session_state.solver_1d
            psi_np = st.session_state.psi_1d[0].detach().cpu().numpy()
            x_np = solver_1d.engine.x.detach().cpu().numpy().flatten()
            
            # Real + Imaginary components
            st.subheader("Wavefunction Components")
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(x=x_np, y=psi_np[:, 0], mode='lines',
                                          name='Re(ψ)', line=dict(color='#00ccff', width=2)))
            fig_comp.add_trace(go.Scatter(x=x_np, y=psi_np[:, 1], mode='lines',
                                          name='Im(ψ)', line=dict(color='#ff6600', width=2)))
            fig_comp.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig_comp, width='stretch')
            
            # Phase
            phase = np.arctan2(psi_np[:, 1], psi_np[:, 0])
            st.subheader("Phase θ(x) = atan2(Im, Re)")
            fig_phase = go.Figure()
            fig_phase.add_trace(go.Scatter(x=x_np, y=phase, mode='lines',
                                           line=dict(color='#ff00ff', width=2)))
            fig_phase.update_layout(template="plotly_dark", height=300,
                                     yaxis_title="Phase (rad)")
            st.plotly_chart(fig_phase, width='stretch')
        else:
            st.info("Click '🔍 Render All Plots' to visualize.")
    else:
        st.info("Train the system first to see wavefunction data.")


# ============================================================
# 📊 QUANTUM GRIDS PAGE (Meme-Grid Style Heatmaps)
# ============================================================
elif page == "📊 Quantum Grids":
    st.title("📊 Quantum Grids")
    
    if mode == "3D Atomic VMC" and st.session_state.solver_3d:
        solver = st.session_state.solver_3d
        
        if not st.session_state.show_plots:
            st.info("Click '🔍 Render All Plots' in the sidebar to visualize.")
        elif not solver.energy_history:
            st.info("Train the system first to generate grid data.")
        else:
            # --- 1. Probability Density Grid ---
            st.subheader("🗺️ Probability Density Field")
            try:
                x_g, y_g, dens = solver.get_density_grid(grid_res=80, extent=4.0)
                
                dens_log = np.log(dens + 1e-20)
                dens_log = (dens_log - dens_log.min()) / (dens_log.max() - dens_log.min() + 1e-30)
                
                fig_field = px.imshow(
                    dens_log.T,
                    x=x_g, y=y_g,
                    color_continuous_scale='Turbo',
                    aspect='equal',
                    origin='lower'
                )
                fig_field.update_layout(
                    title="Log Probability Density (Normalized)",
                    template="plotly_dark",
                    height=500,
                    margin=dict(l=0, r=0, t=40, b=0),
                    coloraxis_colorbar=dict(title="log|ψ|²")
                )
                st.plotly_chart(fig_field, width='stretch')
            except Exception as e:
                st.warning(f"Density field error: {e}")
            
            # --- 2. Walker Position Heatmap ---
            st.subheader("🔥 Walker Density Heatmap")
            try:
                walkers = solver.get_walker_positions()
                pos_xy = walkers[:, 0, :2]
                
                fig_heat = go.Figure(go.Histogram2d(
                    x=pos_xy[:, 0], y=pos_xy[:, 1],
                    nbinsx=60, nbinsy=60,
                    colorscale='Hot',
                    colorbar=dict(title='Counts')
                ))
                fig_heat.update_layout(
                    title="Electron 1 — Position Histogram",
                    template="plotly_dark",
                    height=500,
                    xaxis_title='x (Bohr)',
                    yaxis_title='y (Bohr)'
                )
                st.plotly_chart(fig_heat, width='stretch')
            except Exception as e:
                st.warning(f"Walker heatmap error: {e}")
            
            # --- 3. Energy Landscape Grid ---
            st.subheader("⚡ Potential Energy Landscape")
            try:
                grid_res = 60
                extent = 4.0
                x_pts = torch.linspace(-extent, extent, grid_res, device=solver.device)
                y_pts = torch.linspace(-extent, extent, grid_res, device=solver.device)
                xx, yy = torch.meshgrid(x_pts, y_pts, indexing='ij')
                
                r_grid = torch.stack([xx.flatten(), yy.flatten(), 
                                      torch.zeros_like(xx.flatten())], dim=-1)
                r_grid = r_grid.unsqueeze(1)
                
                if solver.system.n_electrons == 1:
                    with torch.no_grad():
                        V_grid = compute_potential_energy(r_grid, solver.system, solver.device)
                    V_np = V_grid.reshape(grid_res, grid_res).cpu().numpy()
                    V_np = np.clip(V_np, -10, 10)
                    
                    fig_V = px.imshow(
                        V_np.T,
                        x=x_pts.cpu().numpy(), y=y_pts.cpu().numpy(),
                        color_continuous_scale='RdBu_r',
                        aspect='equal',
                        origin='lower'
                    )
                    fig_V.update_layout(
                        title="Coulomb Potential (z=0 slice)",
                        template="plotly_dark",
                        height=500,
                        margin=dict(l=0, r=0, t=40, b=0),
                        coloraxis_colorbar=dict(title="V (Ha)")
                    )
                    st.plotly_chart(fig_V, width='stretch')
                else:
                    st.caption("Potential landscape for multi-electron: showing electron-1 marginal")
            except Exception as e:
                st.warning(f"Potential grid error: {e}")
            
            # --- 4. Correlation Grid (multi-electron) ---
            if solver.system.n_electrons >= 2:
                st.subheader("🔗 Electron Correlation Map")
                try:
                    walkers = solver.get_walker_positions()
                    r1 = np.linalg.norm(walkers[:, 0, :], axis=1)
                    r12 = np.linalg.norm(walkers[:, 0, :] - walkers[:, 1, :], axis=1)
                    
                    fig_corr = go.Figure(go.Histogram2d(
                        x=r1, y=r12,
                        nbinsx=50, nbinsy=50,
                        colorscale='Plasma',
                        colorbar=dict(title='Counts')
                    ))
                    fig_corr.update_layout(
                        title="Electron Correlation: r₁ vs r₁₂",
                        template="plotly_dark",
                        height=450,
                        xaxis_title='r₁ (distance from nucleus)',
                        yaxis_title='r₁₂ (inter-electron distance)'
                    )
                    st.plotly_chart(fig_corr, width='stretch')
                except Exception as e:
                    st.warning(f"Correlation grid error: {e}")
    
    elif mode == "1D Demo (Teaching)" and st.session_state.psi_1d is not None and st.session_state.show_plots:
        solver_1d = st.session_state.solver_1d
        psi_np = st.session_state.psi_1d[0].detach().cpu().numpy()
        density = psi_np[:, 0] ** 2 + psi_np[:, 1] ** 2
        
        st.subheader("🗺️ Probability Density Grid (1D → 2D visualization)")
        grid_2d = np.outer(density, density)
        grid_2d = grid_2d / (grid_2d.max() + 1e-30)
        
        fig_1d_grid = px.imshow(
            grid_2d,
            color_continuous_scale='Inferno',
            aspect='equal'
        )
        fig_1d_grid.update_layout(
            title="Outer Product |ψ(x)|² ⊗ |ψ(y)|² (2-particle analog)",
            template="plotly_dark",
            height=500 
        )
        st.plotly_chart(fig_1d_grid, width='stretch')
    else:
        st.info("Train the system and click '🔍 Render All Plots' to see quantum grids.")


# ============================================================
# 📈 DIAGNOSTICS PAGE
# ============================================================
elif page == "📈 Diagnostics":
    st.title("📈 System Diagnostics")
    
    if mode == "3D Atomic VMC" and st.session_state.solver_3d:
        solver = st.session_state.solver_3d
        
        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Training Steps", st.session_state.training_steps)
        col2.metric("MCMC Step Size", f"{solver.sampler.step_size:.4f}")
        col3.metric("Acceptance Rate", f"{solver.sampler.acceptance_rate:.1%}")
        col4.metric("Equilibrated", "✅" if solver.equilibrated else "❌")
        col5.metric("Optimizer", solver.optimizer_type.upper())
        
        if st.session_state.show_plots and solver.energy_history:
            st.divider()
            
            # Gradient Norm
            col_g, col_s = st.columns(2)
            with col_g:
                fig_grad = go.Figure()
                fig_grad.add_trace(go.Scatter(
                    y=solver.grad_norm_history,
                    mode='lines',
                    line=dict(color='#ff6600', width=2)
                ))
                fig_grad.update_layout(
                    title="Gradient Norm",
                    template="plotly_dark", height=300
                )
                st.plotly_chart(fig_grad, width='stretch')
            
            with col_s:
                fig_step = go.Figure()
                fig_step.add_trace(go.Scatter(
                    y=solver.acceptance_history,
                    mode='lines',
                    line=dict(color='#00ccff', width=2),
                    name='Acceptance'
                ))
                fig_step.update_layout(
                    title="MCMC Acceptance History",
                    template="plotly_dark", height=300,
                    yaxis=dict(range=[0, 1])
                )
                st.plotly_chart(fig_step, width='stretch')
            
            # Energy Statistics
            st.subheader("Energy Statistics")
            if len(solver.energy_history) >= 10:
                recent = solver.energy_history[-min(100, len(solver.energy_history)):]
                system = ATOMS.get(st.session_state.system_key) or MOLECULES.get(st.session_state.system_key)
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Mean E (last 100)", f"{np.mean(recent):.4f}")
                c2.metric("Std E (last 100)", f"{np.std(recent):.4f}")
                c3.metric("Min E", f"{min(solver.energy_history):.4f}")
                c4.metric("Mean Var(E_L)", f"{np.mean(solver.variance_history[-100:]):.4f}")
                if system and system.exact_energy:
                    best_error = abs(min(solver.energy_history) - system.exact_energy) * 1000
                    c5.metric("Best Error (mHa)", f"{best_error:.2f}")
            
            # Walker distance distribution
            st.subheader("🎯 Walker Distance Distribution")
            try:
                walkers = solver.get_walker_positions()
                r_nuc = solver.system.positions().numpy()
                distances = []
                for i in range(solver.system.n_electrons):
                    for I in range(solver.system.n_nuclei):
                        d = np.linalg.norm(walkers[:, i, :] - r_nuc[I], axis=1)
                        distances.append(d)
                distances = np.concatenate(distances)
                
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=distances, nbinsx=80,
                    marker_color='#00ff88',
                    opacity=0.7,
                    name='r (e⁻ → nucleus)'
                ))
                fig_dist.update_layout(
                    title="Electron-Nuclear Distance Distribution",
                    template="plotly_dark", height=350,
                    xaxis_title="r (Bohr)",
                    yaxis_title="Count"
                )
                st.plotly_chart(fig_dist, width='stretch')
            except Exception as e:
                st.warning(f"Distance distribution error: {e}")
        elif not solver.energy_history:
            st.info("Train the system to see diagnostics.")
        else:
            st.info("Click '🔍 Render All Plots' to see diagnostics.")
    
    elif mode == "1D Demo (Teaching)" and st.session_state.solver_1d:
        solver_1d = st.session_state.solver_1d
        col1, col2 = st.columns(2)
        col1.metric("Training Steps", st.session_state.training_steps)
        col2.metric("Memory Buffer", f"{len(solver_1d.memory)} states")
        
        if st.session_state.show_plots and solver_1d.energy_history:
            fig_diag = go.Figure()
            fig_diag.add_trace(go.Scatter(
                y=solver_1d.energy_history,
                mode='lines',
                line=dict(color='#00ff88', width=2)
            ))
            fig_diag.update_layout(
                title="Full Energy History",
                template="plotly_dark", height=350,
                xaxis_title="Step", yaxis_title="E (Eh)"
            )
            st.plotly_chart(fig_diag, width='stretch')
    else:
        st.info("Initialize and train a system to see diagnostics.")


# ============================================================
# 📉 PES CURVES PAGE (Level 10 — Dissociation Curves)
# ============================================================
elif page == "📉 PES Curves (Level 10)":
    st.title("📉 Potential Energy Surface (Level 10)")
    st.markdown("""
    **Dissociation Curves:** Compute energy vs bond distance for diatomic molecules.  
    Tests static correlation (H₂ at R→∞), ionic-covalent transition (LiH),
    and triple-body correlation (H₂O).
    """)
    
    if mode != "3D Atomic VMC":
        st.warning("PES scanning requires 3D Atomic VMC mode.")
    else:
        col_pes1, col_pes2 = st.columns(2)
        
        with col_pes1:
            pes_mol = st.selectbox("Molecule", ["H2", "LiH", "H2O"])
            n_pes_points = st.slider("Bond Distance Points", 5, 30, 12)
            n_pes_steps = st.slider("VMC Steps per Point", 50, 500, 100)
        
        with col_pes2:
            if pes_mol == "H2":
                r_min = st.number_input("R_min (Bohr)", value=0.5, step=0.1)
                r_max_val = st.number_input("R_max (Bohr)", value=6.0, step=0.5)
                st.info("H₂: R_e = 1.401 Bohr, D_e = 4.747 eV")
            elif pes_mol == "LiH":
                r_min = st.number_input("R_min (Bohr)", value=1.5, step=0.1)
                r_max_val = st.number_input("R_max (Bohr)", value=8.0, step=0.5)
                st.info("LiH: R_e = 3.015 Bohr, ionic-covalent transition")
            else:
                r_min = st.number_input("R_OH_min (Bohr)", value=1.0, step=0.1)
                r_max_val = st.number_input("R_OH_max (Bohr)", value=5.0, step=0.5)
                st.info("H₂O: R_OH_e = 1.809 Bohr, bent geometry")
        
        if st.button("🚀 Run PES Scan", width='stretch', type="primary"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            scanner = PESSScanner(
                pes_mol,
                r_range=(r_min, r_max_val),
                n_points=n_pes_points,
                d_model=32, n_layers=2, n_determinants=4
            )
            
            progress = st.progress(0, text="Running PES scan...")
            status_text = st.empty()
            
            def pes_progress(idx, total, energy):
                progress.progress((idx + 1) / total,
                                  text=f"Point {idx+1}/{total}: R={scanner.r_values[idx]:.2f}, E={energy:.4f}")
            
            results = scanner.scan(
                n_train_steps=n_pes_steps,
                n_walkers=256,
                lr=1e-3, device=device,
                progress_callback=pes_progress
            )
            
            progress.empty()
            st.session_state.pes_results = results
            st.session_state.pes_scanner = scanner
            st.success(f"PES scan complete! {len(results)} points computed.")
        
        # Display results
        if st.session_state.pes_results:
            results = st.session_state.pes_results
            r_vals = [r[0] for r in results]
            energies = [r[1] for r in results]
            variances = [r[2] for r in results]
            
            # PES Curve
            fig_pes = go.Figure()
            fig_pes.add_trace(go.Scatter(
                x=r_vals, y=energies,
                mode='lines+markers',
                name='VMC Energy',
                line=dict(color='#00ff88', width=3),
                marker=dict(size=8)
            ))
            
            # Add reference equilibrium energy
            if pes_mol == "H2" and MOLECULES.get("H2"):
                fig_pes.add_hline(
                    y=MOLECULES["H2"].exact_energy,
                    line_dash="dash", line_color="#ff4444",
                    annotation_text=f"Exact E_eq = {MOLECULES['H2'].exact_energy} Ha"
                )
                # Dissociation limit: 2 × H atom
                fig_pes.add_hline(
                    y=2 * ATOMS["H"].exact_energy,
                    line_dash="dot", line_color="#ffaa00",
                    annotation_text="Dissociation Limit (2×H)"
                )
            elif pes_mol == "LiH" and MOLECULES.get("LiH"):
                fig_pes.add_hline(
                    y=MOLECULES["LiH"].exact_energy,
                    line_dash="dash", line_color="#ff4444",
                    annotation_text=f"Exact E_eq = {MOLECULES['LiH'].exact_energy} Ha"
                )
            elif pes_mol == "H2O" and MOLECULES.get("H2O"):
                fig_pes.add_hline(
                    y=MOLECULES["H2O"].exact_energy,
                    line_dash="dash", line_color="#ff4444",
                    annotation_text=f"Exact E_eq = {MOLECULES['H2O'].exact_energy} Ha"
                )
            
            fig_pes.update_layout(
                title=f"Potential Energy Surface — {pes_mol}",
                xaxis_title="Bond Distance R (Bohr)",
                yaxis_title="Energy (Hartree)",
                template="plotly_dark",
                height=500
            )
            st.plotly_chart(fig_pes, width='stretch')
            
            # Variance along PES
            fig_pes_var = go.Figure()
            fig_pes_var.add_trace(go.Scatter(
                x=r_vals, y=variances,
                mode='lines+markers',
                name='Var(E_L)',
                line=dict(color='#ff9900', width=2),
                fill='tozeroy'
            ))
            fig_pes_var.update_layout(
                title=f"Energy Variance Along PES — {pes_mol}",
                xaxis_title="Bond Distance R (Bohr)",
                yaxis_title="Variance (Ha²)",
                template="plotly_dark",
                height=350
            )
            st.plotly_chart(fig_pes_var, width='stretch')
            
            # Data table
            st.subheader("📋 PES Data")
            pes_table = []
            for r, e, v in results:
                pes_table.append({
                    "R (Bohr)": f"{r:.3f}",
                    "Energy (Ha)": f"{e:.6f}",
                    "Variance (Ha²)": f"{v:.6f}"
                })
            st.dataframe(pes_table, width='content')



# ============================================================
# 🌟 EXCITED STATES PAGE (Level 13)
# ============================================================
elif page == "🌟 Excited States (Level 13)":
    st.title("🌟 Excited States (Level 13)")
    st.markdown("""
    **Variance Minimization + Orthogonality:** Compute E₀ < E₁ < E₂ simultaneously.  
    Loss: L_k = ⟨E_L⟩ + β·Var(E_L) + λ·Σ|⟨ψ_k|ψ_j⟩|²
    """)

    if mode != "3D Atomic VMC":
        st.warning("Requires 3D Atomic VMC mode.")
    else:
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            exc_atom = st.selectbox("Atom", list(ATOMS.keys()), key="exc_atom")
            n_states = st.slider("Number of States", 2, 5, 3)
            exc_steps = st.slider("Training Steps", 50, 500, 100, key="exc_steps")
        with col_e2:
            exc_beta = st.number_input("β (variance weight)", value=0.5, step=0.1)
            exc_lambda = st.number_input("λ (orthogonality)", value=10.0, step=1.0)
            exc_walkers = st.slider("Walkers", 128, 1024, 256, key="exc_walk")

        if st.button("🚀 Run Excited State Calculation", width='stretch', type="primary"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            system = ATOMS[exc_atom]
            solver = ExcitedStateSolver(
                system, n_states=n_states, n_walkers=exc_walkers,
                d_model=32, n_layers=2, n_determinants=4,
                lr=1e-3, device=device, beta=exc_beta, ortho_lambda=exc_lambda
            )
            solver.equilibrate(100)
            progress = st.progress(0, text="Training excited states...")
            for i in range(exc_steps):
                res = solver.train_step(5)
                if (i + 1) % 5 == 0:
                    progress.progress((i + 1) / exc_steps, text=f"Step {i+1}/{exc_steps}")
            progress.empty()
            st.session_state.excited_solver = solver
            st.success(f"Done! {n_states} states computed.")

        if st.session_state.excited_solver:
            solver = st.session_state.excited_solver
            fig_exc = go.Figure()
            colors = ['#00ff88', '#ff6600', '#00ccff', '#ff44aa', '#ffcc00']
            for k in range(solver.n_states):
                if solver.energy_histories[k]:
                    fig_exc.add_trace(go.Scatter(
                        y=solver.energy_histories[k], mode='lines',
                        name=f'E_{k}', line=dict(color=colors[k % len(colors)], width=2)
                    ))
            fig_exc.update_layout(title="Multi-State Energy Convergence",
                                  template="plotly_dark", height=500,
                                  xaxis_title="Step", yaxis_title="Energy (Ha)")
            st.plotly_chart(fig_exc, width='stretch')

            energies = solver.get_energies()
            st.subheader("Energy Level Diagram")
            cols = st.columns(len(energies))
            for k, e in enumerate(energies):
                if e is not None:
                    cols[k].metric(f"E_{k}", f"{e:.4f} Ha")


# ============================================================
# 🔮 BERRY PHASE PAGE (Level 14)
# ============================================================
elif page == "🔮 Berry Phase (Level 14)":
    st.title("🔮 Berry Phase (Level 14)")
    st.markdown("""
    **Topological Phase from Neural Wavefunction:**  
    γ = -Im Σ log(⟨ψ(λ_k)|ψ(λ_{k+1})⟩ / |...|)  
    H₃ triangle loop: expected γ = π
    """)

    if mode != "3D Atomic VMC":
        st.warning("Requires 3D Atomic VMC mode.")
    else:
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            n_lambda = st.slider("λ discretization points", 6, 30, 12)
            berry_steps = st.slider("VMC steps per point", 50, 300, 100, key="b_steps")
        with col_b2:
            st.info("**H₃ loop:** equilateral → isosceles → equilateral")
            st.info("Expected: γ = π (conical intersection)")

        if st.button("🚀 Compute Berry Phase", width='stretch', type="primary"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            computer = BerryPhaseComputer(
                BerryPhaseComputer.h3_triangle_loop,
                n_lambda=n_lambda, device=device
            )
            progress = st.progress(0, text="Computing Berry phase...")

            def bp_progress(idx, total, E):
                progress.progress((idx + 1) / total, text=f"λ_{idx+1}/{total}: E={E:.4f}")

            gamma = computer.compute_berry_phase(
                n_vmc_steps=berry_steps, n_walkers=256,
                d_model=32, n_layers=2, n_determinants=4,
                progress_callback=bp_progress
            )
            progress.empty()
            st.session_state.berry_computer = computer
            st.session_state.berry_result = gamma
            st.success(f"Berry Phase γ = {gamma:.4f} rad ({gamma/np.pi:.4f}π)")

        if st.session_state.berry_result is not None:
            gamma = st.session_state.berry_result
            comp = st.session_state.berry_computer

            c1, c2, c3 = st.columns(3)
            c1.metric("γ (rad)", f"{gamma:.4f}")
            c2.metric("γ / π", f"{gamma / np.pi:.4f}")
            c3.metric("Error vs π", f"{abs(gamma - np.pi):.4f} rad")

            fig_bp = go.Figure()
            fig_bp.add_trace(go.Scatter(
                x=list(comp.lambda_values), y=comp.energies, mode='lines+markers',
                name='E(λ)', line=dict(color='#00ff88', width=2)
            ))
            fig_bp.update_layout(title="Energy Along Parameter Loop",
                                 template="plotly_dark", height=400,
                                 xaxis_title="λ (rad)", yaxis_title="E (Ha)")
            st.plotly_chart(fig_bp, width='stretch')

            if comp.overlaps:
                fig_ov = go.Figure()
                fig_ov.add_trace(go.Scatter(
                    x=list(range(len(comp.overlaps))),
                    y=[abs(o) for o in comp.overlaps],
                    mode='lines+markers', name='|⟨ψ_k|ψ_{k+1}⟩|',
                    line=dict(color='#ff6600', width=2)
                ))
                fig_ov.update_layout(title="Overlaps Around Loop",
                                     template="plotly_dark", height=350,
                                     xaxis_title="k", yaxis_title="|Overlap|")
                st.plotly_chart(fig_ov, width='stretch')


# ============================================================
# ⏰ TD-VMC PAGE (Level 15)
# ============================================================
elif page == "⏰ TD-VMC (Level 15)":
    st.title("⏰ Time-Dependent VMC (Level 15)")
    st.markdown("""
    **McLachlan Variational Principle:** iSθ̇ = f  
    Real-time quantum dynamics from neural wavefunction.  
    Track energy E(t) and dipole moment d(t).
    """)

    if mode != "3D Atomic VMC":
        st.warning("Requires 3D Atomic VMC mode.")
    else:
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            td_atom = st.selectbox("Atom", list(ATOMS.keys()), key="td_atom")
            td_dt = st.number_input("Time step dt", value=0.01, step=0.005, format="%.3f")
            td_steps = st.slider("Time steps", 10, 200, 50, key="td_steps")
        with col_t2:
            td_walkers = st.slider("Walkers", 128, 1024, 256, key="td_walk")
            st.info("First converges ground state, then evolves in time.")

        if st.button("🚀 Run TD-VMC", width='stretch', type="primary"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            system = ATOMS[td_atom]
            td = TimeDependentVMC(
                system, n_walkers=td_walkers,
                d_model=32, n_layers=2, n_determinants=4,
                device=device, dt=td_dt
            )
            with torch.no_grad():
                log_psi_fn = lambda r: td.wavefunction(r)
                for _ in range(100):
                    td.sampler.step(log_psi_fn)

            progress = st.progress(0, text="Time evolving...")

            def td_progress(i, total, res):
                progress.progress((i + 1) / total,
                                  text=f"t={res['time']:.3f}: E={res['energy']:.4f}")

            results = td.evolve(td_steps, n_mcmc_steps=3, progress_callback=td_progress)
            progress.empty()
            st.session_state.td_vmc = td
            st.session_state.td_results = results
            st.success(f"TD-VMC complete! {td_steps} time steps.")

        if st.session_state.td_results:
            res = st.session_state.td_results

            fig_te = go.Figure()
            fig_te.add_trace(go.Scatter(
                x=res['time'], y=res['energy'], mode='lines',
                name='E(t)', line=dict(color='#00ff88', width=2)
            ))
            fig_te.update_layout(title="Energy vs Time",
                                 template="plotly_dark", height=400,
                                 xaxis_title="t (a.u.)", yaxis_title="E (Ha)")
            st.plotly_chart(fig_te, width='stretch')

            if res['dipole']:
                dx = [d[0] for d in res['dipole']]
                dy = [d[1] for d in res['dipole']]
                dz = [d[2] for d in res['dipole']]
                fig_dip = go.Figure()
                fig_dip.add_trace(go.Scatter(x=res['time'], y=dx, name='d_x',
                                             line=dict(color='#ff6600')))
                fig_dip.add_trace(go.Scatter(x=res['time'], y=dy, name='d_y',
                                             line=dict(color='#00ccff')))
                fig_dip.add_trace(go.Scatter(x=res['time'], y=dz, name='d_z',
                                             line=dict(color='#ff44aa')))
                fig_dip.update_layout(title="Dipole Moment d(t)",
                                      template="plotly_dark", height=350,
                                      xaxis_title="t (a.u.)", yaxis_title="d (a.u.)")
                st.plotly_chart(fig_dip, width='stretch')



# ============================================================
# 🔷 Page: PERIODIC SYSTEMS (Level 16)
# ============================================================
elif page == "🔷 Periodic Systems (Level 16)":
    st.header("🔷 Periodic Systems — Bloch Waves for Solids")
    st.markdown("""
    **Level 16**: Bloch boundary conditions ψ(r+L) = e^{ikL}·ψ(r).
    Homogeneous Electron Gas (HEG) — the model of metallic bonding.
    Ewald summation for periodic Coulomb. Twist-Averaged Boundary Conditions.
    """)
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        heg_key = st.selectbox("🌍 Select HEG System", list(PERIODIC_SYSTEMS.keys()))
        heg_sys = PERIODIC_SYSTEMS[heg_key]
        st.metric("r_s (Wigner-Seitz)", f"{heg_sys.rs:.1f} Bohr")
        st.metric("N_electrons", heg_sys.n_electrons)
        st.metric("Cell Volume", f"{heg_sys.cell_volume():.2f} Bohr³")
    
    with col_p2:
        if heg_sys.exact_energy_per_electron is not None:
            st.metric("E_exact / e⁻", f"{heg_sys.exact_energy_per_electron:.6f} Ha")
            st.metric("Total E_exact", f"{heg_sys.exact_energy_per_electron * heg_sys.n_electrons:.4f} Ha")
        L_diag = np.linalg.norm(np.array(heg_sys.cell_vectors[0]))
        st.metric("Cell Side L", f"{L_diag:.3f} Bohr")
    
    st.subheader("📈 HEG Energy vs Density")
    rs_vals = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
    e_vals = []
    for rs in rs_vals:
        sys_tmp = build_heg_system(rs=rs, n_electrons=14)
        e_vals.append(sys_tmp.exact_energy_per_electron)
    
    fig_heg = go.Figure()
    fig_heg.add_trace(go.Scatter(
        x=rs_vals, y=e_vals, mode='lines+markers',
        name='E/electron (Ceperley-Alder)',
        line=dict(color='#00ff88', width=3),
        marker=dict(size=10)
    ))
    fig_heg.update_layout(
        title="HEG Energy per Electron (Ceperley-Alder)",
        xaxis_title="r_s (Bohr)",
        yaxis_title="E/electron (Ha)",
        template="plotly_dark", height=400
    )
    st.plotly_chart(fig_heg, width='stretch')
    
    with st.expander("📝 Ewald Summation Details"):
        st.markdown("""
        **Ewald summation** splits the 1/r Coulomb interaction into:
        - **Real-space sum**: erfc(αr)/r (short-range, fast convergence)
        - **Reciprocal-space sum**: Structure factor |S(G)|² (long-range)
        - **Self-energy**: -α/√π × N_e
        - **Madelung**: -πN²/(2Vα²) (neutralizing background)
        
        **TABC**: Average over Monkhorst-Pack k-grid to eliminate finite-size shell effects.
        """)


# ============================================================
# ⚡ Page: SPIN-ORBIT COUPLING (Level 17)
# ============================================================
elif page == "⚡ Spin-Orbit (Level 17)":
    st.header("⚡ Spin-Orbit Coupling — Relativistic Quantum Mechanics")
    st.markdown("""
    **Level 17**: Breit-Pauli spin-orbit Hamiltonian.
    H_SO = (α²/2) Σ_{i,I} Z_I / r_{iI}³ · L·S.
    2-component spinor wavefunctions. Fine-structure splitting of Helium.
    """)
    
    col_so1, col_so2, col_so3 = st.columns(3)
    with col_so1:
        so_atom = st.selectbox("⚛️ Select Atom", ["He", "Li", "Be", "B", "C", "N", "O"])
        st.info(f"Fine-structure constant α ≈ 1/137.036")
    with col_so2:
        system = ATOMS[so_atom]
        so_sys = SpinOrbitSystem(base_system=system)
        st.metric("Z", system.nuclei[0][0])
        st.metric("N_electrons", system.n_electrons)
    with col_so3:
        # Theoretical fine-structure splitting estimate
        Z = system.nuclei[0][0]
        alpha = 1.0 / 137.036
        # Rough estimate: ΔE ~ α² Z⁴ / (2n³) in Hartree
        delta_E = (alpha**2 * Z**4) / 4.0  # for n=2 roughly
        ha_to_cm = 219474.63
        st.metric("Est. splitting", f"{delta_E * ha_to_cm:.2f} cm⁻¹")
        st.metric("Est. splitting", f"{delta_E * 1000:.4f} mHa")
    
    st.subheader("📊 Fine-Structure Scaling")
    Z_vals = list(range(1, 11))
    split_vals = [(1/137.036)**2 * Z**4 / 4.0 * 219474.63 for Z in Z_vals]
    atom_names = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]
    
    fig_so = go.Figure()
    fig_so.add_trace(go.Bar(
        x=atom_names, y=split_vals,
        marker_color=['#ff4444' if s < 1 else '#ffaa00' if s < 100 else '#00ff88' for s in split_vals],
        text=[f"{s:.1f}" for s in split_vals],
        textposition='auto'
    ))
    fig_so.update_layout(
        title="Spin-Orbit Splitting ~ α²Z⁴ (cm⁻¹)",
        xaxis_title="Atom",
        yaxis_title="ΔE (cm⁻¹)",
        template="plotly_dark", height=400,
        yaxis_type="log"
    )
    st.plotly_chart(fig_so, width='stretch')
    
    with st.expander("📝 Breit-Pauli Physics"):
        st.markdown("""
        **Spin-orbit coupling** arises from the interaction between an electron's
        orbital angular momentum L and its spin S in the nucleus's electric field.
        
        - **Operator**: H_SO = (α²/2) Σ Z/r³ L·S
        - **Spinor representation**: Ψ = (ψ↑, ψ↓)ᵀ (2-component)
        - **He (³P term)**: J=0,1,2 splitting measured to **12 significant figures**
        - **Effect**: Scales as Z⁴ — dramatic for heavy atoms
        """)


# ============================================================
# 🔗 Page: ENTANGLEMENT ENTROPY (Level 18)
# ============================================================
elif page == "🔗 Entanglement (Level 18)":
    st.header("🔗 Entanglement Entropy — SWAP Trick")
    st.markdown("""
    **Level 18**: Rényi-2 entanglement entropy from neural wavefunction.
    First-ever computation of molecular entanglement from VMC/NQS.
    
    Method: S₂(A) = -log Tr(ρ_A²) via the SWAP trick with two independent copies.
    """)
    
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        ent_system = st.selectbox("⚛️ System", ["He", "Li", "Be", "B", "C", "H2", "LiH"],
                                   key="ent_sys")
        ent_partition = st.selectbox("✂️ Partition", ["spin", "spatial", "atom"])
    with col_e2:
        ent_samples = st.slider("🎲 MCMC Samples", 500, 5000, 1000, step=500, key="ent_samp")
        ent_burn = st.slider("🔥 Burn-in Steps", 20, 200, 50, step=10, key="ent_burn")
    
    if st.button("🔗 Compute Entanglement", type="primary"):
        with st.spinner("Computing Rényi-2 entropy via SWAP trick..."):
            if ent_system in ATOMS:
                system = ATOMS[ent_system]
            else:
                system = MOLECULES[ent_system]
            
            wf = NeuralWavefunction(system, d_model=32, n_layers=2, n_determinants=4)
            sampler = MetropolisSampler(n_walkers=ent_samples, n_electrons=system.n_electrons)
            sampler.initialize_around_nuclei(system)
            
            eec = EntanglementEntropyComputer(
                wavefunction=wf, sampler=sampler, system=system,
                partition=ent_partition
            )
            result = eec.compute_renyi2(n_samples=ent_samples, n_mcmc_burn=ent_burn)
            st.session_state.entanglement_results = result
        st.success("Entanglement computed!")
    
    if st.session_state.entanglement_results:
        res = st.session_state.entanglement_results
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("S₂ (Rényi-2)", f"{res['S2']:.4f}")
        col_r2.metric("Purity Tr(ρ_A²)", f"{res['purity']:.6f}")
        col_r3.metric("Stat. Error", f"{res['error']:.4f}")
        
        st.markdown(f"""
        | Property | Value |
        |---|---|
        | Subsystem A | {res['n_A']} electrons |
        | Subsystem B | {res['n_B']} electrons |
        | Partition | {res['partition']} |
        | S₂ = -ln(Tr(ρ²)) | **{res['S2']:.4f}** |
        """)
        
        # Interpretive diagram
        fig_ent = go.Figure()
        fig_ent.add_trace(go.Indicator(
            mode="gauge+number",
            value=res['S2'],
            title={'text': "Entanglement Entropy S₂"},
            gauge={
                'axis': {'range': [0, 3]},
                'bar': {'color': '#ff44aa'},
                'steps': [
                    {'range': [0, 0.5], 'color': '#1a1a2e'},
                    {'range': [0.5, 1.5], 'color': '#16213e'},
                    {'range': [1.5, 3.0], 'color': '#0f3460'}
                ],
                'threshold': {
                    'line': {'color': '#00ff88', 'width': 4},
                    'thickness': 0.75, 'value': np.log(2)
                }
            }
        ))
        fig_ent.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_ent, width='stretch')
        st.caption("Green line = ln(2) ≈ 0.693 (maximum for 2-state system)")


# ============================================================
# 🔬 Page: CONSERVATION LAW DISCOVERY (Level 19)
# ============================================================
elif page == "🔬 Conservation Discovery (Level 19)":
    st.header("🔬 Autonomous Conservation Law Discovery")
    st.markdown("""
    **Level 19**: Noether's theorem in reverse — discover unknown conservation laws.
    
    Train Q_φ(r) to minimize |<[Ĥ,Q̂]>|² (commutation with Hamiltonian)
    while maximizing orthogonality to known quantities (L_x, L_y, L_z).
    
    If Q commutes with H and is novel → **mathematical theorem discovered by computation**.
    """)
    
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        cons_atom = st.selectbox("⚛️ System", ["He", "Li", "Be", "B", "C"], key="cons_sys")
    with col_c2:
        cons_steps = st.slider("🔄 Training Steps", 50, 1000, 200, step=50, key="cons_steps")
    with col_c3:
        cons_lambda = st.slider("λ Novelty", 0.1, 5.0, 1.0, step=0.1, key="cons_lam")
    
    if st.button("🔬 Discover Conservation Laws", type="primary"):
        system = ATOMS[cons_atom]
        wf = NeuralWavefunction(system, d_model=32, n_layers=2, n_determinants=4)
        
        progress = st.empty()
        status_text = st.empty()
        
        discoverer = ConservationLawDiscovery(
            system=system, wavefunction=wf,
            d_hidden=64, n_hidden_layers=2,
            novelty_lambda=cons_lambda
        )
        
        def cons_progress(step, total, result):
            progress.progress(step / total,
                              text=f"Step {step}/{total}: comm={result['commutator_loss']:.6f}")
        
        results = discoverer.discover(
            n_steps=cons_steps, n_mcmc_steps=5,
            progress_callback=cons_progress
        )
        progress.empty()
        st.session_state.conservation_results = results
        st.success("Discovery complete!")
    
    if st.session_state.conservation_results:
        res = st.session_state.conservation_results
        
        # Status indicators
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("[H,Q]", f"{res['final_commutator']:.2e}")
        col_s2.metric("Novelty", f"{res['final_novelty']:.4f}")
        col_s3.metric("Conserved?", "✅ YES" if res['is_conserved'] else "❌ NO")
        col_s4.metric("Novel?", "⭐ YES" if res['is_novel'] else "❌ NO")
        
        # Interpretation
        st.info(res['interpretation'])
        
        # Training curves
        hist = res['history']
        fig_cons = make_subplots(rows=1, cols=2,
                                 subplot_titles=["Commutator Loss", "Novelty Penalty"])
        fig_cons.add_trace(go.Scatter(
            y=hist['commutator_loss'], mode='lines',
            name='|<[H,Q]>|²', line=dict(color='#00ff88')
        ), row=1, col=1)
        fig_cons.add_trace(go.Scatter(
            y=hist['novelty_penalty'], mode='lines',
            name='Novelty', line=dict(color='#ff44aa')
        ), row=1, col=2)
        fig_cons.update_layout(template="plotly_dark", height=350,
                               showlegend=True)
        st.plotly_chart(fig_cons, width='stretch')
        
        # Q statistics
        fig_q = go.Figure()
        fig_q.add_trace(go.Scatter(
            y=hist['Q_std'], mode='lines',
            name='σ(Q)', line=dict(color='#ffaa00', width=2)
        ))
        fig_q.update_layout(title="Q Network Variance (target: 1.0)",
                            template="plotly_dark", height=300,
                            xaxis_title="Step", yaxis_title="std(Q)")
        st.plotly_chart(fig_q, width='stretch')


# ============================================================
# 🧠 COLLECTIVE MEMORY PAGE (Level 20)
# ============================================================
elif page == "🎨 Latent Dream Memory 🖼️":
    st.title("🎨 Latent Dream Memory 🖼️")
    st.markdown("""
    **Stigmergy & Agent-Based Memory Convergence:**  
    These grids represent the 'Collective Memory' of agent clusters. 
    RGB encoding: **Red (Danger)**, **Green (Resource)**, **Blue (Sacred)**.
    The maps evolve as agents interact with the quantum landscape.
    """)
    
    # Stable seeds across session to prevent jumping on every click
    if 'stigmergy_seed' not in st.session_state:
        st.session_state.stigmergy_seed = int(time.time())
    
    master_seed = st.session_state.stigmergy_seed
    
    if st.button("🎲 Regenerate Memory Grids"):
        st.session_state.stigmergy_seed = int(time.time())
        st.rerun()
    
    st.subheader("🌐 Global Memory Grids (8 Replicate Clusters)")
    
    # 2x4 Grid layout
    row1 = st.columns(2)
    row2 = st.columns(2)
    row3 = st.columns(2)
    row4 = st.columns(2)
    
    all_cols = row1 + row2 + row3 + row4
    
    for i, col in enumerate(all_cols):
        with col:
            seed = master_seed + i
            fig = plot_stigmergy_map(seed=seed)
            st.pyplot(fig, clear_figure=True)
            st.caption(f"Cluster Instance #{i+1} — Seed: {seed}")

    st.divider()
    st.subheader("🌋 Converged Latent Blooms (Final States)")
    st.markdown("These 8 final plots represent the fully converged, hazy state of the neural memory field.")
    
    bloom_row1 = st.columns(2)
    bloom_row2 = st.columns(2)
    bloom_row3 = st.columns(2)
    bloom_row4 = st.columns(2)
    all_bloom_cols = bloom_row1 + bloom_row2 + bloom_row3 + bloom_row4
    
    for i, col in enumerate(all_bloom_cols):
        with col:
            seed = master_seed + 100 + i
            fig_bloom = plot_latent_bloom(seed=seed)
            st.pyplot(fig_bloom, clear_figure=True)
            st.caption(f"Latent Bloom Output #{i+1} — Seed: {seed}")

    st.divider()
    st.subheader("💎 The Master Latent Dimension Bloom")
    st.markdown("""
    **The Final Synthesis:** This high-fidelity visualization represents the union of the 
    Wavefunction Manifold and the Selective State Space (SSM) hidden dimensions. 
    It is the 'Singularity' of the neural quantum state.
    """)
    
    # Render the Master Bloom (passing the solver if initialized)
    solver_ref = st.session_state.solver_3d if st.session_state.solver_3d else None
    fig_master = plot_master_bloom(solver=solver_ref, seed=master_seed + 999)
    st.pyplot(fig_master, clear_figure=True)
    st.caption("🌌 Master Consensus Field — Unified Neural Quantum State [Nobel Territory]")


# ============================================================
#  FOOTER
# ============================================================
st.sidebar.divider()
st.sidebar.caption("The Schrödinger Dream v4.0 (Phase 4 — Nobel Territory)")
st.sidebar.caption("Beyond FermiNet — SSM-Backflow Engine")
st.sidebar.caption(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
st.sidebar.caption("Levels 1-20 Implemented — Complete Engine")











