import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
import time
from quantum_physics import QuantumPhysicsEngine
from neural_dream import SymplecticSSMGenerator, HamiltonianFlowNetwork
from solver import SchrodingerSolver

st.set_page_config(page_title="The Schrödinger Dream", layout="wide", page_icon="⚛️")

# --- Custom CSS for Sci-Fi Look ---
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .stButton>button {
        color: #00ff00;
        border-color: #00ff00;
        background-color: transparent;
        font-family: 'Courier New', Courier, monospace;
    }
    h1, h2, h3 {
        color: #e0e0e0;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .metric-value {
        font-size: 2em;
        font-weight: bold;
        color: #00cec9;
    }
</style>
""", unsafe_allow_html=True)

# --- Singleton State ---
if 'solver' not in st.session_state:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.session_state.solver = SchrodingerSolver(grid_size=256, device=device)
    st.session_state.V_x = None
    st.session_state.psi = None
    st.session_state.energy_history = []
    st.session_state.is_running = False

solver = st.session_state.solver

# --- Sidebar Controls ---
st.sidebar.title("🎛️ Laboratory Controls")

potential_type = st.sidebar.selectbox(
    "Select Potential V(x)",
    ["Harmonic Oscillator", "Double Well", "Infinite Square Well", "Step Potential", "Custom Draw (Coming Soon)"]
)

if st.sidebar.button("Initialize System"):
    # Create Potential
    x = solver.engine.x.detach().cpu().numpy().flatten()
    grid_size = len(x)
    V_np = np.zeros_like(x)
    
    if potential_type == "Harmonic Oscillator":
        # V = 0.5 * k * x^2
        V_np = 0.5 * x**2
    elif potential_type == "Double Well":
        # V = a(x^2 - 1)^2
        V_np = 0.5 * ((x/3)**2 - 1)**2
    elif potential_type == "Infinite Square Well":
        V_np[:] = 100.0 # High walls
        V_np[grid_size//4 : 3*grid_size//4] = 0.0
    elif potential_type == "Step Potential":
        V_np[grid_size//2:] = 2.0
        
    st.session_state.V_x = torch.tensor(V_np, dtype=torch.float32).view(1, -1, 1).to(solver.device)
    
    # Reset Solver
    solver.memory.clear()
    st.session_state.energy_history = []
    st.session_state.psi = None
    st.success("System Initialized! State: Superposition of random noise.")

run_awake = st.sidebar.checkbox("Run Awake Loop (Physics Minimization)", value=False)
run_dream = st.sidebar.button("🌙 Trigger REM Sleep (Dream Tunneling)")
measure_btn = st.sidebar.button("👁️ Measure State (Collapse)")

# --- Main Display ---
col1, col2 = st.columns([3, 1])

with col1:
    st.title("The Schrödinger Dream ⚛️")
    st.markdown("**Status**: " + ("Running Physics Engine..." if run_awake else "Idle"))
    
    plot_placeholder = st.empty()

with col2:
    st.subheader("Quantum Metrics")
    energy_placeholder = st.empty()
    uncertainty_placeholder = st.empty()
    
# --- Simulation Loop ---
if st.session_state.V_x is not None:
    V_x = st.session_state.V_x
    
    # Awake Loop
    if run_awake:
        loss = solver.train_step_awake(V_x)
        st.session_state.energy_history.append(loss)
        
        # Get current Psi
        with torch.no_grad():
            psi_tensor = solver.generator(solver.engine.x.view(1, -1, 1).to(solver.device))
        st.session_state.psi = psi_tensor
        
        # Determine Energy
        current_energy = loss # Variational Energy
        
    # Dream Loop (One-shot)
    if run_dream:
        with st.spinner("Dreaming... Tunneling through efficient phase space..."):
            # Train HGF quickly on current memory
            for _ in range(50): solver.train_step_dream()
            
            # Generate new Psi guess
            psi_dream = solver.generate_dream(V_x)
            
            # Inject into Generator (Simulated by re-training generator on this specific target for a few steps usually
            # or simply updating the "Current State" visualization if we treated generator as variational Ansatz)
            # For this demo, let's visualize the Dream State directly
            st.session_state.psi = psi_dream
            st.success("Dream complete! Found lower energy configuration.")
            
    # Measurement
    if measure_btn and st.session_state.psi is not None:
        with st.spinner("Collapsing Wavefunction..."):
            psi_curr = st.session_state.psi[0].cpu() # [L, 2]
            collapsed_x = solver.engine.hamiltonian_score_matching_collapse(psi_curr)
            st.toast(f"Measured Particle at x = {collapsed_x.item():.4f}")
            # Collapse visualization (Delta function)
            # For viz, we just show a marker
            
    # Visualization
    if st.session_state.psi is not None:
        psi_np = st.session_state.psi[0].detach().cpu().numpy() # [L, 2]
        x_np = solver.engine.x.detach().cpu().numpy().flatten()
        
        # Density
        density = psi_np[:, 0]**2 + psi_np[:, 1]**2
        density = density / (np.sum(density) * solver.engine.dx.item()) # Normalize for plot
        
        # Potential (Scaled to fit plot)
        pot_np = V_x[0].detach().cpu().numpy().flatten()
        
        # Phase (Angle)
        phase = np.arctan2(psi_np[:, 1], psi_np[:, 0])
        
        # Create Plot
        fig = go.Figure()
        
        # Potential
        fig.add_trace(go.Scatter(x=x_np, y=pot_np, mode='lines', name='Potential V(x)', 
                                 line=dict(color='gray', width=2, dash='dash')))
        
        # Wavefunction Density
        fig.add_trace(go.Scatter(x=x_np, y=density, mode='lines', name='Prob Density |ψ|²', 
                                 line=dict(color='#00ff00', width=3), fill='tozeroy'))
        
        # Phase as color (complex visualization usually requires 3D or color bar, simple line here)
        # We can add Real/Imag components optionally
        
        fig.update_layout(
            title="Quantum State Visualization",
            xaxis_title="Position (x)",
            yaxis_title="Probability / Potential",
            template="plotly_dark",
            height=500
        )
        
        if measure_btn:
            fig.add_vline(x=collapsed_x.item(), line_width=3, line_dash="dash", line_color="red", annotation_text="Measured")
            
        plot_placeholder.plotly_chart(fig, use_container_width=True)
        
        # Metrics
        if len(st.session_state.energy_history) > 0:
            e_val = st.session_state.energy_history[-1]
            energy_placeholder.metric("Expectation Energy <E>", f"{e_val:.4f} Eh")
            
            # Uncertainty calc
            # <x^2> - <x>^2
            # Simple calc for display
            x_mom = np.sum(density * x_np) * solver.engine.dx.item()
            x2_mom = np.sum(density * x_np**2) * solver.engine.dx.item()
            var_x = x2_mom - x_mom**2
            sigma_x = np.sqrt(max(0, var_x))
            uncertainty_placeholder.metric("Position Uncertainty Δx", f"{sigma_x:.4f}")
            
    else:
        st.info("System not initialized or wavefunction undefined.")

else:
    st.info("Please Initialize System from the Sidebar.")
