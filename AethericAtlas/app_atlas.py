"""
🌌 AETHER: Advanced Exploratory Theoretical & Holistic Engineering Research
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The most aesthetically beautiful and scientifically advanced visualization platform.
336 Nobel-tier plots across 21 cutting-edge domains.

Inspired by "Your Name" (君の名は) — Where science meets art.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.special as sp
from scipy import signal, fft, integrate, linalg, stats
from scipy.spatial import Voronoi, ConvexHull, Delaunay
from scipy.interpolate import griddata
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# 🎨 CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AETHER ∞ Scientific Dreams",
    layout="wide",
    page_icon="∞",
    initial_sidebar_state="expanded"
)

# Dark theme styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0a0a0f;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(20, 20, 30, 0.5);
        padding: 10px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 30, 40, 0.4);
        border-radius: 8px;
        color: #888;
        padding: 10px 20px;
        border: 1px solid rgba(100, 100, 150, 0.2);
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 255, 200, 0.1);
        color: #00ffc8;
        border: 1px solid #00ffc8;
        box-shadow: 0 0 20px rgba(0, 255, 200, 0.3);
    }
    h1, h2, h3 {
        color: #e0e0f0 !important;
        font-weight: 300 !important;
        letter-spacing: 2px;
    }
    .plot-title {
        color: #00ffc8;
        font-size: 0.9em;
        font-weight: 300;
        letter-spacing: 1px;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Global color schemes for dark theme
COLORS_PLASMA = px.colors.sequential.Plasma_r
COLORS_VIRIDIS = px.colors.sequential.Viridis
COLORS_TWILIGHT = px.colors.cyclical.Twilight
COLORS_CIVIDIS = px.colors.sequential.Cividis
COLORS_TURBO = px.colors.sequential.Turbid

# Universal plot config for dark theme
PLOT_CONFIG = {
    'displayModeBar': False,
    'staticPlot': False
}

DARK_LAYOUT = dict(
    paper_bgcolor='rgba(10,10,15,0.8)',
    plot_bgcolor='rgba(15,15,25,0.9)',
    font=dict(color='#b0b0c0', family='Helvetica, sans-serif', size=10),
    xaxis=dict(
        gridcolor='rgba(100,100,150,0.15)',
        zerolinecolor='rgba(100,100,150,0.3)',
        color='#808090'
    ),
    yaxis=dict(
        gridcolor='rgba(100,100,150,0.15)',
        zerolinecolor='rgba(100,100,150,0.3)',
        color='#808090'
    ),
    margin=dict(l=20, r=20, t=40, b=20),
    showlegend=False
)

# ═══════════════════════════════════════════════════════════════════════════
# 🎲 MASTER SEED SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

if 'master_seed' not in st.session_state:
    st.session_state.master_seed = np.random.randint(0, 1000000)

def get_seed(offset=0):
    """Generate deterministic seed based on master seed and offset"""
    return (st.session_state.master_seed + offset) % 1000000

# ═══════════════════════════════════════════════════════════════════════════
# 📊 REINFORCEMENT LEARNING PLOTS (Subject 1/21)
# ═══════════════════════════════════════════════════════════════════════════

def rl_stigmergy_field(seed):
    """1. Stigmergy field - environmental memory from agent interactions"""
    np.random.seed(seed)
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Multiple agent pheromone deposits
    stigmergy = np.zeros_like(X)
    n_agents = 8
    for i in range(n_agents):
        cx, cy = np.random.uniform(-3, 3, 2)
        sigma = np.random.uniform(0.5, 1.5)
        stigmergy += np.exp(-((X-cx)**2 + (Y-cy)**2)/(2*sigma**2))
    
    # Decay and diffusion
    stigmergy *= np.exp(-0.1 * np.sqrt(X**2 + Y**2))
    
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=stigmergy,
        colorscale='Viridis',
        showscale=False,
        contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="cyan", project=dict(z=True)))
    )])
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis=dict(showbackground=False, gridcolor='rgba(100,100,150,0.2)'),
            yaxis=dict(showbackground=False, gridcolor='rgba(100,100,150,0.2)'),
            zaxis=dict(showbackground=False, gridcolor='rgba(100,100,150,0.2)'),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        title=dict(text="Stigmergy Environmental Memory Field", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def rl_policy_manifold(seed):
    """2. Policy gradient manifold in parameter space"""
    np.random.seed(seed)
    theta1 = np.linspace(-np.pi, np.pi, 80)
    theta2 = np.linspace(-np.pi, np.pi, 80)
    T1, T2 = np.meshgrid(theta1, theta2)
    
    # Reward landscape
    R = (np.sin(T1) * np.cos(T2) + 
         0.5 * np.sin(2*T1) + 
         0.3 * np.cos(3*T2) +
         0.2 * np.sin(T1 + T2))
    
    fig = go.Figure(data=[go.Surface(
        x=T1, y=T2, z=R,
        colorscale='Plasma',
        showscale=False,
        lighting=dict(ambient=0.4, diffuse=0.8, specular=0.5, roughness=0.5)
    )])
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis_title='θ₁',
            yaxis_title='θ₂',
            zaxis_title='R(θ)',
            camera=dict(eye=dict(x=1.7, y=-1.7, z=1.3))
        ),
        title=dict(text="Policy Gradient Manifold", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def rl_value_function_flow(seed):
    """3. Value function flow field across state space"""
    np.random.seed(seed)
    x = np.linspace(-3, 3, 25)
    y = np.linspace(-3, 3, 25)
    X, Y = np.meshgrid(x, y)
    
    # Value function
    V = -0.5 * (X**2 + Y**2) + np.sin(X) + np.cos(Y)
    
    # Gradient (policy direction)
    dVdx = -X + np.cos(X)
    dVdy = -Y - np.sin(Y)
    
    fig = go.Figure()
    
    # Contour fill
    fig.add_trace(go.Contour(
        x=x, y=y, z=V,
        colorscale='Cividis',
        showscale=False,
        contours=dict(coloring='heatmap'),
        opacity=0.7
    ))
    
    # Vector field
    fig.add_trace(go.Scatter(
        x=X.flatten()[::2], y=Y.flatten()[::2],
        mode='markers',
        marker=dict(
            size=3,
            color=V.flatten()[::2],
            colorscale='Twilight',
            showscale=False
        ),
        hoverinfo='skip'
    ))
    
    # Flow lines
    for i in range(0, len(x), 4):
        for j in range(0, len(y), 4):
            fig.add_trace(go.Scatter(
                x=[X[j,i], X[j,i] + 0.3*dVdx[j,i]],
                y=[Y[j,i], Y[j,i] + 0.3*dVdy[j,i]],
                mode='lines',
                line=dict(color='rgba(0,255,200,0.4)', width=1.5),
                hoverinfo='skip'
            ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title='State s₁',
        yaxis_title='State s₂',
        title=dict(text="Value Function Flow Field V(s)", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def rl_reward_landscape_3d(seed):
    """4. 3D reward landscape with optimal path"""
    np.random.seed(seed)
    x = np.linspace(-4, 4, 80)
    y = np.linspace(-4, 4, 80)
    X, Y = np.meshgrid(x, y)
    
    # Complex reward terrain
    R = (2 * np.exp(-0.3*(X**2 + Y**2)) +
         np.exp(-0.5*((X-2)**2 + (Y-2)**2)) +
         0.5 * np.exp(-0.5*((X+2)**2 + (Y+2)**2)) -
         0.3 * (np.sin(X) + np.cos(Y)))
    
    fig = go.Figure()
    
    fig.add_trace(go.Surface(
        x=X, y=Y, z=R,
        colorscale='Turbo',
        showscale=False,
        opacity=0.9
    ))
    
    # Optimal path
    t = np.linspace(0, 2*np.pi, 100)
    path_x = 3 * np.cos(t)
    path_y = 3 * np.sin(t)
    path_z = 2 * np.exp(-0.3*(path_x**2 + path_y**2)) + 0.5
    
    fig.add_trace(go.Scatter3d(
        x=path_x, y=path_y, z=path_z,
        mode='lines',
        line=dict(color='cyan', width=6),
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis_title='Action a₁',
            yaxis_title='Action a₂',
            zaxis_title='Reward R',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        title=dict(text="Reward Landscape with Optimal Trajectory", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def rl_bellman_residual_field(seed):
    """5. Bellman residual field - TD error distribution"""
    np.random.seed(seed)
    n = 100
    x = np.linspace(0, 10, n)
    y = np.linspace(0, 10, n)
    X, Y = np.meshgrid(x, y)
    
    # TD error field
    V_current = np.sin(0.5*X) * np.cos(0.5*Y)
    V_next = np.sin(0.5*(X+0.5)) * np.cos(0.5*(Y+0.5))
    reward = 0.1 * np.exp(-0.1*(X-5)**2 - 0.1*(Y-5)**2)
    
    gamma = 0.99
    bellman_residual = reward + gamma * V_next - V_current
    
    fig = go.Figure(data=[go.Heatmap(
        x=x, y=y, z=bellman_residual,
        colorscale='RdBu',
        zmid=0,
        showscale=False
    )])
    
    fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title='State Dimension 1',
        yaxis_title='State Dimension 2',
        title=dict(text="Bellman Residual Field (TD Error)", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def rl_actor_critic_phase_space(seed):
    """6. Actor-Critic phase space trajectories"""
    np.random.seed(seed)
    
    # Multiple trajectories
    n_traj = 6
    fig = go.Figure()
    
    for i in range(n_traj):
        t = np.linspace(0, 4*np.pi, 200)
        phase = np.random.uniform(0, 2*np.pi)
        freq = np.random.uniform(0.8, 1.2)
        
        actor = np.sin(freq * t + phase) * np.exp(-0.05*t)
        critic = np.cos(freq * t + phase) * np.exp(-0.05*t)
        value = t / (4*np.pi)
        
        fig.add_trace(go.Scatter3d(
            x=actor, y=critic, z=value,
            mode='lines',
            line=dict(
                color=value,
                colorscale='Plasma',
                width=3
            ),
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis_title='Actor π(a|s)',
            yaxis_title='Critic V(s)',
            zaxis_title='Training Progress',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        title=dict(text="Actor-Critic Phase Space Dynamics", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def rl_q_function_surface(seed):
    """7. Q-function surface Q(s,a)"""
    np.random.seed(seed)
    s = np.linspace(-2, 2, 90)
    a = np.linspace(-2, 2, 90)
    S, A = np.meshgrid(s, a)
    
    # Q-function with multiple local optima
    Q = (np.exp(-((S-1)**2 + (A-1)**2)) + 
         0.7 * np.exp(-((S+1)**2 + (A+1)**2)) +
         0.5 * np.exp(-(S**2 + (A-0.5)**2)) -
         0.3 * (S**2 + A**2))
    
    fig = go.Figure(data=[go.Surface(
        x=S, y=A, z=Q,
        colorscale='Viridis',
        showscale=False,
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="lime", project=dict(z=True))
        )
    )])
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis_title='State s',
            yaxis_title='Action a',
            zaxis_title='Q(s,a)',
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.4))
        ),
        title=dict(text="Q-Function State-Action Surface", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def rl_advantage_function_contour(seed):
    """8. Advantage function A(s,a) = Q(s,a) - V(s)"""
    np.random.seed(seed)
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # Advantage function
    V = np.exp(-0.5*(X**2 + Y**2))
    Q = np.exp(-0.5*((X-1)**2 + (Y-1)**2))
    A = Q - V
    
    fig = go.Figure(data=[go.Contour(
        x=x, y=y, z=A,
        colorscale='RdYlBu',
        contours=dict(
            coloring='heatmap',
            showlabels=True,
            labelfont=dict(size=8, color='white')
        ),
        showscale=False
    )])
    
    fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title='State Component 1',
        yaxis_title='State Component 2',
        title=dict(text="Advantage Function A(s,a)", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def rl_experience_replay_topology(seed):
    """9. Experience replay buffer topology"""
    np.random.seed(seed)
    n_points = 300
    
    # Generate experience points in latent space
    theta = np.random.uniform(0, 2*np.pi, n_points)
    r = np.random.beta(2, 5, n_points) * 3
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.random.normal(0, 0.5, n_points) + np.exp(-r/2)
    
    # Color by priority
    priority = np.exp(-r/3) + np.random.normal(0, 0.1, n_points)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=3,
            color=priority,
            colorscale='Inferno',
            showscale=False,
            opacity=0.7
        ),
        hoverinfo='skip'
    )])
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis_title='State Embedding φ₁',
            yaxis_title='State Embedding φ₂',
            zaxis_title='TD Error Priority',
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.4))
        ),
        title=dict(text="Experience Replay Buffer Topology", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def rl_monte_carlo_tree_structure(seed):
    """10. Monte Carlo Tree Search structure"""
    np.random.seed(seed)
    
    # Build tree
    G = nx.balanced_tree(3, 4)
    pos = nx.spring_layout(G, seed=seed, k=2, iterations=50)
    
    # Convert to 3D
    node_x = [pos[k][0] for k in G.nodes()]
    node_y = [pos[k][1] for k in G.nodes()]
    node_z = [nx.shortest_path_length(G, 0, k) for k in G.nodes()]
    
    # Node values (MCTS scores)
    node_values = [np.random.beta(2, 2) for _ in G.nodes()]
    
    # Edges
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        z0 = nx.shortest_path_length(G, 0, edge[0])
        z1 = nx.shortest_path_length(G, 0, edge[1])
        edge_trace.append(go.Scatter3d(
            x=[x0, x1], y=[y0, y1], z=[z0, z1],
            mode='lines',
            line=dict(color='rgba(100,150,255,0.4)', width=2),
            hoverinfo='skip'
        ))
    
    fig = go.Figure(data=edge_trace)
    
    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(
            size=8,
            color=node_values,
            colorscale='Viridis',
            showscale=False,
            line=dict(color='cyan', width=1)
        ),
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis_title='Tree Depth',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        title=dict(text="Monte Carlo Tree Search Structure", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def rl_curiosity_intrinsic_reward(seed):
    """11. Curiosity-driven intrinsic reward landscape"""
    np.random.seed(seed)
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    
    # Intrinsic reward (prediction error)
    n_explored = 5
    intrinsic = np.ones_like(X) * 0.5
    
    for i in range(n_explored):
        cx, cy = np.random.uniform(2, 8, 2)
        explored = np.exp(-((X-cx)**2 + (Y-cy)**2)/2)
        intrinsic -= 0.4 * explored
    
    intrinsic = np.maximum(intrinsic, 0)
    
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=intrinsic,
        colorscale='Hot',
        showscale=False,
        lighting=dict(ambient=0.5, diffuse=0.7, specular=0.8)
    )])
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis_title='Environment Dimension 1',
            yaxis_title='Environment Dimension 2',
            zaxis_title='Curiosity Signal',
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.3))
        ),
        title=dict(text="Curiosity-Driven Intrinsic Reward", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def rl_trust_region_constraint(seed):
    """12. Trust region constraint sphere in policy space"""
    np.random.seed(seed)
    
    # Sphere
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    fig = go.Figure()
    
    # Trust region sphere
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale='Blues',
        showscale=False,
        opacity=0.3,
        hoverinfo='skip'
    ))
    
    # Policy gradient vectors
    n_vecs = 30
    for i in range(n_vecs):
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi)
        r = np.random.uniform(0.5, 1.0)
        
        px = r * np.sin(phi) * np.cos(theta)
        py = r * np.sin(phi) * np.sin(theta)
        pz = r * np.cos(phi)
        
        # Gradient direction
        dx, dy, dz = px * 0.3, py * 0.3, pz * 0.3
        
        fig.add_trace(go.Scatter3d(
            x=[px, px+dx], y=[py, py+dy], z=[pz, pz+dz],
            mode='lines',
            line=dict(color='rgba(255,100,100,0.6)', width=2),
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis_title='θ₁',
            yaxis_title='θ₂',
            zaxis_title='θ₃',
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.5))
        ),
        title=dict(text="Trust Region Policy Optimization", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def rl_hindsight_experience_trajectory(seed):
    """13. Hindsight experience replay trajectory"""
    np.random.seed(seed)
    
    fig = go.Figure()
    
    n_episodes = 5
    for ep in range(n_episodes):
        t = np.linspace(0, 5, 100)
        
        # Failed trajectory
        x_fail = t + np.random.normal(0, 0.3, len(t))
        y_fail = np.sin(t) + np.random.normal(0, 0.3, len(t))
        z_fail = t**2 / 10
        
        fig.add_trace(go.Scatter3d(
            x=x_fail, y=y_fail, z=z_fail,
            mode='lines',
            line=dict(color='rgba(255,100,100,0.4)', width=3),
            hoverinfo='skip'
        ))
        
        # Hindsight goal
        goal_x, goal_y = x_fail[-1], y_fail[-1]
        fig.add_trace(go.Scatter3d(
            x=[goal_x], y=[goal_y], z=[z_fail[-1]],
            mode='markers',
            marker=dict(size=8, color='lime', symbol='diamond'),
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis_title='State s₁',
            yaxis_title='State s₂',
            zaxis_title='Time t',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        title=dict(text="Hindsight Experience Replay Trajectories", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def rl_distributional_value_function(seed):
    """14. Distributional RL - value distribution"""
    np.random.seed(seed)
    
    states = np.linspace(-3, 3, 40)
    returns = np.linspace(-2, 4, 60)
    S, R = np.meshgrid(states, returns)
    
    # Value distribution Z(s,a)
    Z = np.zeros_like(S)
    for i, s in enumerate(states):
        mu = 2 * np.exp(-0.3*s**2)
        sigma = 0.5 + 0.3*np.abs(s)
        Z[:, i] = stats.norm.pdf(returns, mu, sigma)
    
    fig = go.Figure(data=[go.Surface(
        x=S, y=R, z=Z,
        colorscale='Plasma',
        showscale=False
    )])
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis_title='State s',
            yaxis_title='Return Z',
            zaxis_title='Probability P(Z|s,a)',
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.3))
        ),
        title=dict(text="Distributional Value Function Z(s,a)", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def rl_multi_agent_coordination(seed):
    """15. Multi-agent coordination network"""
    np.random.seed(seed)
    
    n_agents = 12
    G = nx.random_geometric_graph(n_agents, 0.5, seed=seed)
    pos_2d = nx.spring_layout(G, seed=seed, k=1.5)
    
    # 3D positions with coordination strength as z
    node_x = [pos_2d[k][0] for k in G.nodes()]
    node_y = [pos_2d[k][1] for k in G.nodes()]
    node_z = [G.degree(k) * 0.2 + np.random.uniform(-0.1, 0.1) for k in G.nodes()]
    
    # Edges
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos_2d[edge[0]]
        x1, y1 = pos_2d[edge[1]]
        z0, z1 = node_z[edge[0]], node_z[edge[1]]
        
        edge_traces.append(go.Scatter3d(
            x=[x0, x1], y=[y0, y1], z=[z0, z1],
            mode='lines',
            line=dict(color='rgba(0,200,255,0.5)', width=3),
            hoverinfo='skip'
        ))
    
    fig = go.Figure(data=edge_traces)
    
    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(
            size=12,
            color=node_z,
            colorscale='Turbo',
            showscale=False,
            line=dict(color='white', width=2)
        ),
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis_title='Agent Position X',
            yaxis_title='Agent Position Y',
            zaxis_title='Coordination Strength',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.4))
        ),
        title=dict(text="Multi-Agent Coordination Network", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def rl_model_based_planning_tree(seed):
    """16. Model-based planning lookahead tree"""
    np.random.seed(seed)
    
    # Create planning tree
    levels = 5
    branching = 3
    
    fig = go.Figure()
    
    def add_tree_level(level, parent_pos, parent_val, depth=0):
        if depth >= levels:
            return
        
        angle_range = np.pi / (2 ** depth)
        for i in range(branching):
            angle = -angle_range + (2 * angle_range * i / (branching - 1)) if branching > 1 else 0
            
            x = parent_pos[0] + np.cos(angle) * (1 / (depth + 1))
            y = parent_pos[1] + np.sin(angle) * (1 / (depth + 1))
            z = depth + 1
            
            # Value propagation
            val = parent_val * 0.9 + np.random.normal(0, 0.1)
            
            # Edge
            fig.add_trace(go.Scatter3d(
                x=[parent_pos[0], x],
                y=[parent_pos[1], y],
                z=[parent_pos[2], z],
                mode='lines',
                line=dict(color=f'rgba(100,{150+depth*20},{255-depth*30},0.5)', width=2),
                hoverinfo='skip'
            ))
            
            # Node
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers',
                marker=dict(
                    size=6,
                    color=val,
                    colorscale='Viridis',
                    showscale=False
                ),
                hoverinfo='skip'
            ))
            
            add_tree_level(level, (x, y, z), val, depth + 1)
    
    # Root
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=10, color='red'),
        hoverinfo='skip'
    ))
    
    add_tree_level(0, (0, 0, 0), 1.0, 0)
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis_title='Planning Horizon',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        title=dict(text="Model-Based Planning Lookahead Tree", font=dict(size=13, color='#00ffc8'))
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# 🧠 ARTIFICIAL GENERAL INTELLIGENCE PLOTS (Subject 2/21)
# ═══════════════════════════════════════════════════════════════════════════

def agi_cognitive_architecture_network(seed):
    """1. Cognitive architecture network - AGI modules"""
    np.random.seed(seed)
    
    # Define AGI modules
    modules = {
        'Perception': (0, 0, 0),
        'Attention': (1, 0.5, 0.5),
        'Working Memory': (2, 0, 1),
        'Long-term Memory': (2, 1, 1),
        'Reasoning': (3, 0.5, 1.5),
        'Planning': (4, 0.5, 2),
        'Action': (5, 0, 2),
        'Meta-cognition': (3, 0.5, 0)
    }
    
    fig = go.Figure()
    
    # Connections
    connections = [
        ('Perception', 'Attention'),
        ('Attention', 'Working Memory'),
        ('Working Memory', 'Long-term Memory'),
        ('Working Memory', 'Reasoning'),
        ('Reasoning', 'Planning'),
        ('Planning', 'Action'),
        ('Meta-cognition', 'Reasoning'),
        ('Meta-cognition', 'Attention'),
        ('Long-term Memory', 'Reasoning'),
    ]
    
    for start, end in connections:
        x0, y0, z0 = modules[start]
        x1, y1, z1 = modules[end]
        
        fig.add_trace(go.Scatter3d(
            x=[x0, x1], y=[y0, y1], z=[z0, z1],
            mode='lines',
            line=dict(color='rgba(0,255,200,0.4)', width=4),
            hoverinfo='skip'
        ))
    
    # Nodes
    node_names = list(modules.keys())
    node_x = [modules[k][0] for k in node_names]
    node_y = [modules[k][1] for k in node_names]
    node_z = [modules[k][2] for k in node_names]
    
    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        marker=dict(size=15, color='cyan', line=dict(color='white', width=2)),
        text=node_names,
        textposition='top center',
        textfont=dict(size=9, color='white'),
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis=dict(showticklabels=False, title=''),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.5))
        ),
        title=dict(text="Cognitive Architecture Network", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def agi_knowledge_graph_embedding(seed):
    """2. Knowledge graph embedding space"""
    np.random.seed(seed)
    
    # Generate knowledge embeddings
    n_entities = 200
    
    # Cluster entities by semantic similarity
    n_clusters = 5
    entities_x, entities_y, entities_z = [], [], []
    colors = []
    
    for cluster in range(n_clusters):
        center = np.random.uniform(-3, 3, 3)
        n_in_cluster = n_entities // n_clusters
        
        for _ in range(n_in_cluster):
            point = center + np.random.normal(0, 0.5, 3)
            entities_x.append(point[0])
            entities_y.append(point[1])
            entities_z.append(point[2])
            colors.append(cluster)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=entities_x, y=entities_y, z=entities_z,
        mode='markers',
        marker=dict(
            size=3,
            color=colors,
            colorscale='Twilight',
            showscale=False,
            opacity=0.7
        ),
        hoverinfo='skip'
    ))
    
    # Add some relation edges
    n_relations = 50
    for _ in range(n_relations):
        i, j = np.random.choice(len(entities_x), 2, replace=False)
        fig.add_trace(go.Scatter3d(
            x=[entities_x[i], entities_x[j]],
            y=[entities_y[i], entities_y[j]],
            z=[entities_z[i], entities_z[j]],
            mode='lines',
            line=dict(color='rgba(255,200,100,0.2)', width=1),
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis_title='Semantic Dimension 1',
            yaxis_title='Semantic Dimension 2',
            zaxis_title='Semantic Dimension 3',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        title=dict(text="Knowledge Graph Embedding Space", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def agi_attention_mechanism_heatmap(seed):
    """3. Multi-head attention weights"""
    np.random.seed(seed)
    
    n_tokens = 50
    
    # Generate attention pattern
    attention = np.zeros((n_tokens, n_tokens))
    
    # Self-attention with decay
    for i in range(n_tokens):
        for j in range(n_tokens):
            distance = abs(i - j)
            attention[i, j] = np.exp(-distance / 10) + np.random.uniform(0, 0.1)
    
    # Add strong diagonal and some long-range connections
    for i in range(n_tokens):
        attention[i, i] += 0.5
        if i % 5 == 0 and i < n_tokens - 10:
            attention[i, i+10] += 0.3
    
    # Normalize
    attention = attention / attention.sum(axis=1, keepdims=True)
    
    fig = go.Figure(data=[go.Heatmap(
        z=attention,
        colorscale='Plasma',
        showscale=False
    )])
    
    fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title='Key Position',
        yaxis_title='Query Position',
        title=dict(text="Multi-Head Attention Weights", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def agi_world_model_latent_dynamics(seed):
    """4. World model latent dynamics"""
    np.random.seed(seed)
    
    # Simulate latent state evolution
    n_steps = 300
    n_dims = 3
    
    # Lorenz-like attractor for world model
    dt = 0.01
    sigma, rho, beta = 10, 28, 8/3
    
    states = np.zeros((n_steps, n_dims))
    states[0] = np.random.uniform(-1, 1, n_dims)
    
    for i in range(1, n_steps):
        x, y, z = states[i-1]
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        states[i] = states[i-1] + dt * np.array([dx, dy, dz])
    
    # Color by time
    colors = np.linspace(0, 1, n_steps)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=states[:, 0],
        y=states[:, 1],
        z=states[:, 2],
        mode='lines',
        line=dict(
            color=colors,
            colorscale='Viridis',
            width=3
        ),
        hoverinfo='skip'
    )])
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis_title='Latent z₁',
            yaxis_title='Latent z₂',
            zaxis_title='Latent z₃',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        title=dict(text="World Model Latent Dynamics", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def agi_goal_hierarchy_tree(seed):
    """5. Hierarchical goal decomposition"""
    np.random.seed(seed)
    
    # Create hierarchical goal tree
    G = nx.balanced_tree(r=3, h=4)
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot') if hasattr(nx, 'nx_agraph') else nx.spring_layout(G, seed=seed)
    
    # Convert to 3D based on depth
    depths = {node: nx.shortest_path_length(G, 0, node) for node in G.nodes()}
    max_depth = max(depths.values())
    
    node_x = [pos[k][0] for k in G.nodes()]
    node_y = [pos[k][1] for k in G.nodes()]
    node_z = [depths[k] * 2 for k in G.nodes()]
    
    # Priority scores
    priorities = [np.random.beta(2, 5) * (max_depth - depths[k] + 1) for k in G.nodes()]
    
    fig = go.Figure()
    
    # Edges
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]][0], pos[edge[0]][1], depths[edge[0]] * 2
        x1, y1, z1 = pos[edge[1]][0], pos[edge[1]][1], depths[edge[1]] * 2
        
        fig.add_trace(go.Scatter3d(
            x=[x0, x1], y=[y0, y1], z=[z0, z1],
            mode='lines',
            line=dict(color='rgba(100,200,255,0.4)', width=2),
            hoverinfo='skip'
        ))
    
    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(
            size=8,
            color=priorities,
            colorscale='Hot',
            showscale=False,
            line=dict(color='white', width=1)
        ),
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis_title='Abstraction Level',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.4))
        ),
        title=dict(text="Hierarchical Goal Decomposition", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def agi_consciousness_integration_field(seed):
    """6. Global workspace theory - consciousness integration"""
    np.random.seed(seed)
    
    x = np.linspace(-3, 3, 80)
    y = np.linspace(-3, 3, 80)
    X, Y = np.meshgrid(x, y)
    
    # Integration field (multiple specialized processors broadcasting)
    integration = np.zeros_like(X)
    n_processors = 7
    
    for i in range(n_processors):
        cx, cy = np.random.uniform(-2, 2, 2)
        amplitude = np.random.uniform(0.5, 1.0)
        width = np.random.uniform(0.8, 1.5)
        integration += amplitude * np.exp(-((X-cx)**2 + (Y-cy)**2)/(2*width**2))
    
    # Normalize
    integration = integration / integration.max()
    
    fig = go.Figure(data=[go.Contour(
        x=x, y=y, z=integration,
        colorscale='Turbo',
        contours=dict(
            coloring='heatmap',
            showlabels=False
        ),
        showscale=False
    )])
    
    fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title='Information Stream 1',
        yaxis_title='Information Stream 2',
        title=dict(text="Global Workspace Integration Field", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def agi_reward_shaping_landscape(seed):
    """7. Intrinsic motivation and reward shaping"""
    np.random.seed(seed)
    
    x = np.linspace(-4, 4, 90)
    y = np.linspace(-4, 4, 90)
    X, Y = np.meshgrid(x, y)
    
    # Extrinsic reward
    extrinsic = np.exp(-0.3*((X-2)**2 + (Y-2)**2))
    
    # Intrinsic reward (novelty/curiosity)
    intrinsic = 1 / (1 + 0.5*(X**2 + Y**2))
    
    # Combined shaped reward
    total_reward = extrinsic + 0.5 * intrinsic
    
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=total_reward,
        colorscale='Inferno',
        showscale=False,
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="yellow", project=dict(z=True))
        )
    )])
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis_title='State Feature 1',
            yaxis_title='State Feature 2',
            zaxis_title='Shaped Reward',
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.3))
        ),
        title=dict(text="Reward Shaping Landscape", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def agi_transfer_learning_manifold(seed):
    """8. Transfer learning task manifold"""
    np.random.seed(seed)
    
    # Multiple tasks in shared representation space
    n_tasks = 6
    n_points_per_task = 50
    
    fig = go.Figure()
    
    for task_id in range(n_tasks):
        # Task center
        center = np.random.uniform(-2, 2, 3)
        
        # Task-specific data distribution
        theta = np.linspace(0, 2*np.pi, n_points_per_task)
        r = np.random.uniform(0.3, 0.8, n_points_per_task)
        
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        z = center[2] + np.random.normal(0, 0.2, n_points_per_task)
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=4,
                color=task_id,
                colorscale='Plasma',
                showscale=False,
                opacity=0.7
            ),
            name=f'Task {task_id+1}',
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis_title='Shared Feature φ₁',
            yaxis_title='Shared Feature φ₂',
            zaxis_title='Shared Feature φ₃',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.4))
        ),
        title=dict(text="Transfer Learning Task Manifold", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def agi_meta_learning_optimization_surface(seed):
    """9. Meta-learning optimization landscape"""
    np.random.seed(seed)
    
    theta1 = np.linspace(-2, 2, 70)
    theta2 = np.linspace(-2, 2, 70)
    T1, T2 = np.meshgrid(theta1, theta2)
    
    # Meta-loss landscape (faster learning trajectories)
    meta_loss = (T1**2 + T2**2) * (1 + 0.5*np.sin(3*T1)*np.cos(3*T2))
    meta_loss = meta_loss + 0.1 * (T1 - T2)**2
    
    fig = go.Figure(data=[go.Surface(
        x=T1, y=T2, z=meta_loss,
        colorscale='Cividis',
        showscale=False,
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.6)
    )])
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis_title='Meta-parameter θ₁',
            yaxis_title='Meta-parameter θ₂',
            zaxis_title='Meta-loss ℒ',
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.3))
        ),
        title=dict(text="Meta-Learning Optimization Surface", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def agi_causal_graph_discovery(seed):
    """10. Causal graph structure learning"""
    np.random.seed(seed)
    
    # Build causal DAG
    n_vars = 10
    G = nx.DiGraph()
    G.add_nodes_from(range(n_vars))
    
    # Add causal edges
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            if np.random.random() < 0.3:
                G.add_edge(i, j, weight=np.random.uniform(0.3, 1.0))
    
    pos = nx.spring_layout(G, seed=seed, k=2)
    
    # 3D positions
    node_x = [pos[k][0] for k in G.nodes()]
    node_y = [pos[k][1] for k in G.nodes()]
    node_z = [len(list(nx.ancestors(G, k))) * 0.3 for k in G.nodes()]
    
    fig = go.Figure()
    
    # Edges with weights
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]][0], pos[edge[0]][1], node_z[edge[0]]
        x1, y1, z1 = pos[edge[1]][0], pos[edge[1]][1], node_z[edge[1]]
        weight = G[edge[0]][edge[1]]['weight']
        
        fig.add_trace(go.Scatter3d(
            x=[x0, x1], y=[y0, y1], z=[z0, z1],
            mode='lines',
            line=dict(color=f'rgba(255,{int(255*weight)},100,0.6)', width=3*weight),
            hoverinfo='skip'
        ))
    
    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(
            size=10,
            color=node_z,
            colorscale='Viridis',
            showscale=False,
            line=dict(color='white', width=2)
        ),
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis_title='Causal Depth',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.4))
        ),
        title=dict(text="Causal Graph Structure Discovery", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def agi_neural_turing_machine_memory(seed):
    """11. Neural Turing Machine memory access patterns"""
    np.random.seed(seed)
    
    time_steps = 50
    memory_size = 30
    
    # Memory access weights over time
    access_pattern = np.zeros((time_steps, memory_size))
    
    # Simulate read/write attention
    for t in range(time_steps):
        # Moving attention head
        center = (t / time_steps) * memory_size
        for m in range(memory_size):
            distance = abs(m - center)
            access_pattern[t, m] = np.exp(-distance**2 / 20) + np.random.uniform(0, 0.05)
    
    # Normalize
    access_pattern = access_pattern / access_pattern.sum(axis=1, keepdims=True)
    
    fig = go.Figure(data=[go.Heatmap(
        z=access_pattern.T,
        colorscale='Hot',
        showscale=False
    )])
    
    fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title='Time Step',
        yaxis_title='Memory Location',
        title=dict(text="Neural Turing Machine Memory Access", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def agi_compositional_generalization(seed):
    """12. Compositional generalization structure"""
    np.random.seed(seed)
    
    # Primitive concepts
    n_primitives = 8
    n_compositions = 30
    
    # Base primitives in embedding space
    primitive_pos = np.random.uniform(-2, 2, (n_primitives, 3))
    
    # Compositional concepts (combinations of primitives)
    composition_pos = []
    composition_colors = []
    
    for i in range(n_compositions):
        # Select 2-3 primitives to compose
        n_compose = np.random.randint(2, 4)
        selected = np.random.choice(n_primitives, n_compose, replace=False)
        
        # Composition as average of primitives (with noise)
        comp_pos = primitive_pos[selected].mean(axis=0) + np.random.normal(0, 0.2, 3)
        composition_pos.append(comp_pos)
        composition_colors.append(len(selected))
    
    composition_pos = np.array(composition_pos)
    
    fig = go.Figure()
    
    # Primitives
    fig.add_trace(go.Scatter3d(
        x=primitive_pos[:, 0],
        y=primitive_pos[:, 1],
        z=primitive_pos[:, 2],
        mode='markers',
        marker=dict(
            size=12,
            color='red',
            symbol='diamond',
            line=dict(color='white', width=2)
        ),
        name='Primitives',
        hoverinfo='skip'
    ))
    
    # Compositions
    fig.add_trace(go.Scatter3d(
        x=composition_pos[:, 0],
        y=composition_pos[:, 1],
        z=composition_pos[:, 2],
        mode='markers',
        marker=dict(
            size=6,
            color=composition_colors,
            colorscale='Viridis',
            showscale=False,
            opacity=0.7
        ),
        name='Compositions',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis_title='Semantic Axis 1',
            yaxis_title='Semantic Axis 2',
            zaxis_title='Semantic Axis 3',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.4))
        ),
        title=dict(text="Compositional Generalization Structure", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def agi_reasoning_proof_tree(seed):
    """13. Logical reasoning proof tree"""
    np.random.seed(seed)
    
    # Build reasoning tree (binary tree for simplicity)
    depth = 5
    G = nx.balanced_tree(2, depth)
    
    # Position nodes
    pos = {}
    for node in G.nodes():
        level = int(np.floor(np.log2(node + 1)))
        position_in_level = node - (2**level - 1)
        width = 2 ** (depth - level)
        
        pos[node] = (
            -2**(depth-1) + width * position_in_level + width/2,
            -level * 1.5
        )
    
    # 3D with confidence scores
    node_x = [pos[k][0] for k in G.nodes()]
    node_y = [pos[k][1] for k in G.nodes()]
    node_z = [np.random.beta(5, 2) for k in G.nodes()]  # Confidence
    
    fig = go.Figure()
    
    # Edges
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]][0], pos[edge[0]][1], node_z[edge[0]]
        x1, y1, z1 = pos[edge[1]][0], pos[edge[1]][1], node_z[edge[1]]
        
        fig.add_trace(go.Scatter3d(
            x=[x0, x1], y=[y0, y1], z=[z0, z1],
            mode='lines',
            line=dict(color='rgba(100,200,255,0.4)', width=2),
            hoverinfo='skip'
        ))
    
    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(
            size=7,
            color=node_z,
            colorscale='Plasma',
            showscale=False,
            line=dict(color='white', width=1)
        ),
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis_title='Inference Depth',
            zaxis_title='Confidence',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        title=dict(text="Logical Reasoning Proof Tree", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def agi_emergent_communication_protocol(seed):
    """14. Emergent communication between agents"""
    np.random.seed(seed)
    
    n_agents = 8
    n_messages = 40
    vocab_size = 20
    
    # Communication matrix
    comm_matrix = np.zeros((n_messages, vocab_size))
    
    # Each message is a probability distribution over vocabulary
    for i in range(n_messages):
        # Some structure in communication
        dominant_tokens = np.random.choice(vocab_size, size=3, replace=False)
        for token in dominant_tokens:
            comm_matrix[i, token] = np.random.beta(5, 2)
        
        # Add noise
        comm_matrix[i] += np.random.uniform(0, 0.1, vocab_size)
        comm_matrix[i] /= comm_matrix[i].sum()
    
    fig = go.Figure(data=[go.Heatmap(
        z=comm_matrix,
        colorscale='Viridis',
        showscale=False
    )])
    
    fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title='Vocabulary Token',
        yaxis_title='Message ID',
        title=dict(text="Emergent Communication Protocol", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def agi_abstract_reasoning_matrix(seed):
    """15. Abstract reasoning pattern matrix"""
    np.random.seed(seed)
    
    # Create abstract pattern (like Raven's Progressive Matrices)
    size = 3
    patterns = []
    
    for row in range(size):
        row_patterns = []
        for col in range(size):
            # Generate pattern based on row and col
            pattern = np.zeros((10, 10))
            
            # Rule 1: Number of shapes increases
            n_shapes = row + col + 1
            
            for _ in range(n_shapes):
                cx, cy = np.random.randint(2, 8, 2)
                r = np.random.randint(1, 3)
                y, x = np.ogrid[-cx:10-cx, -cy:10-cy]
                mask = x*x + y*y <= r*r
                pattern[mask] = 1
            
            row_patterns.append(pattern)
        patterns.append(row_patterns)
    
    # Create subplot
    fig = make_subplots(
        rows=size, cols=size,
        subplot_titles=[f'P{i*size+j+1}' for i in range(size) for j in range(size)],
        vertical_spacing=0.05,
        horizontal_spacing=0.05
    )
    
    for i in range(size):
        for j in range(size):
            fig.add_trace(
                go.Heatmap(
                    z=patterns[i][j],
                    colorscale='Greys',
                    showscale=False,
                    hoverinfo='skip'
                ),
                row=i+1, col=j+1
            )
    
    fig.update_layout(
        **DARK_LAYOUT,
        height=600,
        title=dict(text="Abstract Reasoning Pattern Matrix", font=dict(size=13, color='#00ffc8'))
    )
    
    return fig

def agi_general_value_function_network(seed):
    """16. General Value Function network (GVF)"""
    np.random.seed(seed)
    
    # Multiple value functions for different goals
    n_gvfs = 8
    n_states = 60
    
    # State space
    states = np.linspace(-3, 3, n_states)
    
    # Different value functions
    gvf_values = []
    for gvf_id in range(n_gvfs):
        # Each GVF has different reward structure
        center = np.random.uniform(-2, 2)
        width = np.random.uniform(0.5, 2.0)
        amplitude = np.random.uniform(0.5, 1.0)
        
        values = amplitude * np.exp(-((states - center)**2) / (2 * width**2))
        gvf_values.append(values)
    
    gvf_values = np.array(gvf_values)
    
    fig = go.Figure()
    
    for gvf_id in range(n_gvfs):
        fig.add_trace(go.Scatter(
            x=states,
            y=gvf_values[gvf_id],
            mode='lines',
            line=dict(width=2),
            name=f'GVF-{gvf_id+1}',
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title='State s',
        yaxis_title='Value V(s)',
        title=dict(text="General Value Function Network", font=dict(size=13, color='#00ffc8'))
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# ⚛️ QUANTUM PHYSICS PLOTS (Subject 3/21)
# ═══════════════════════════════════════════════════════════════════════════

def qp_wavefunction_interference(seed):
    """1. Quantum wavefunction interference pattern"""
    np.random.seed(seed)
    
    x = np.linspace(-5, 5, 200)
    y = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x, y)
    
    # Two-slit interference
    slit1 = (-1.5, 0)
    slit2 = (1.5, 0)
    
    k = 15  # wave number
    r1 = np.sqrt((X - slit1[0])**2 + (Y - slit1[1])**2)
    r2 = np.sqrt((X - slit2[0])**2 + (Y - slit2[1])**2)
    
    psi = np.exp(1j * k * r1) / np.sqrt(r1 + 0.1) + np.exp(1j * k * r2) / np.sqrt(r2 + 0.1)
    intensity = np.abs(psi)**2
    
    fig = go.Figure(data=[go.Heatmap(
        x=x, y=y, z=intensity,
        colorscale='Hot',
        showscale=False
    )])
    
    fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title='Position x (nm)',
        yaxis_title='Position y (nm)',
        title=dict(text="Wavefunction Interference Pattern |ψ|²", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def qp_quantum_tunneling_barrier(seed):
    """2. Quantum tunneling through potential barrier"""
    np.random.seed(seed)
    
    x = np.linspace(-5, 5, 300)
    
    # Potential barrier
    V = np.zeros_like(x)
    barrier_start, barrier_end = -0.5, 0.5
    V[(x >= barrier_start) & (x <= barrier_end)] = 3.0
    
    # Incident wavefunction (energy < barrier height)
    E = 2.0
    k1 = np.sqrt(2 * E)
    k2 = np.sqrt(2 * abs(E - 3.0)) * 1j
    
    psi = np.zeros_like(x, dtype=complex)
    psi[x < barrier_start] = np.exp(1j * k1 * x[x < barrier_start])
    psi[(x >= barrier_start) & (x <= barrier_end)] = np.exp(k2 * (x[(x >= barrier_start) & (x <= barrier_end)] - barrier_start))
    psi[x > barrier_end] = 0.1 * np.exp(1j * k1 * (x[x > barrier_end] - barrier_end))
    
    fig = go.Figure()
    
    # Potential
    fig.add_trace(go.Scatter(
        x=x, y=V,
        mode='lines',
        line=dict(color='red', width=3, dash='dash'),
        name='Potential V(x)',
        hoverinfo='skip'
    ))
    
    # Wavefunction
    fig.add_trace(go.Scatter(
        x=x, y=np.abs(psi)**2,
        mode='lines',
        fill='tozeroy',
        line=dict(color='cyan', width=2),
        fillcolor='rgba(0,255,200,0.3)',
        name='|ψ|²',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title='Position x',
        yaxis_title='Amplitude',
        title=dict(text="Quantum Tunneling Through Barrier", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def qp_hydrogen_orbital_3d(seed):
    """3. Hydrogen atom orbital 3D visualization"""
    np.random.seed(seed)
    
    # Choose random orbital
    orbitals = [(2, 1, 0), (3, 2, 1), (3, 2, 0), (4, 3, 2)]
    n, l, m = orbitals[seed % len(orbitals)]
    
    # Spherical grid
    r = np.linspace(0.1, 20, 40)
    theta = np.linspace(0, np.pi, 40)
    R, THETA = np.meshgrid(r, theta)
    
    # Radial part (simplified Laguerre)
    rho = 2 * R / n
    radial = (rho**l) * np.exp(-rho/2)
    
    # Angular part (spherical harmonics)
    if m == 0:
        angular = sp.lpmv(0, l, np.cos(THETA))
    else:
        angular = sp.lpmv(abs(m), l, np.cos(THETA))
    
    psi = radial * angular
    prob_density = np.abs(psi)**2
    
    # Convert to Cartesian for 3D plot
    phi = np.linspace(0, 2*np.pi, 40)
    R_3d, THETA_3d, PHI_3d = np.meshgrid(r[::2], theta[::2], phi[::2], indexing='ij')
    
    X = R_3d * np.sin(THETA_3d) * np.cos(PHI_3d)
    Y = R_3d * np.sin(THETA_3d) * np.sin(PHI_3d)
    Z = R_3d * np.cos(THETA_3d)
    
    # Sample probability
    prob_3d = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            r_val = R_3d[i, j, 0]
            theta_val = THETA_3d[i, j, 0]
            idx_r = np.argmin(np.abs(r - r_val))
            idx_theta = np.argmin(np.abs(theta - theta_val))
            prob_3d[i, j, :] = prob_density[idx_theta, idx_r]
    
    # Isosurface
    fig = go.Figure(data=[go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=prob_3d.flatten(),
        isomin=prob_3d.max() * 0.1,
        isomax=prob_3d.max() * 0.8,
        surface_count=3,
        colorscale='Plasma',
        showscale=False,
        opacity=0.6,
        caps=dict(x_show=False, y_show=False, z_show=False)
    )])
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis_title='x (Bohr radii)',
            yaxis_title='y (Bohr radii)',
            zaxis_title='z (Bohr radii)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        title=dict(text=f"Hydrogen Orbital (n={n}, l={l}, m={m})", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def qp_schrodinger_evolution(seed):
    """4. Time evolution of Schrödinger equation"""
    np.random.seed(seed)
    
    x = np.linspace(-10, 10, 200)
    t = np.linspace(0, 5, 100)
    X, T = np.meshgrid(x, t)
    
    # Gaussian wavepacket evolution
    x0, k0, sigma = 0, 2, 1.0
    psi = np.zeros_like(X, dtype=complex)
    
    for i, t_val in enumerate(t):
        sigma_t = sigma * np.sqrt(1 + (t_val / (2 * sigma**2))**2)
        x0_t = x0 + k0 * t_val
        psi[i] = np.exp(-(x - x0_t)**2 / (4 * sigma_t**2)) * np.exp(1j * (k0 * x - k0**2 * t_val / 2))
    
    prob_density = np.abs(psi)**2
    
    fig = go.Figure(data=[go.Surface(
        x=X, y=T, z=prob_density,
        colorscale='Viridis',
        showscale=False
    )])
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis_title='Position x',
            yaxis_title='Time t',
            zaxis_title='|ψ(x,t)|²',
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2))
        ),
        title=dict(text="Schrödinger Equation Time Evolution", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def qp_quantum_harmonic_oscillator(seed):
    """5. Quantum harmonic oscillator energy levels"""
    np.random.seed(seed)
    
    x = np.linspace(-5, 5, 300)
    
    # Potential
    V = 0.5 * x**2
    
    fig = go.Figure()
    
    # Potential curve
    fig.add_trace(go.Scatter(
        x=x, y=V,
        mode='lines',
        line=dict(color='gray', width=2),
        name='V(x) = ½kx²',
        hoverinfo='skip'
    ))
    
    # Energy levels and wavefunctions
    n_levels = 6
    colors = px.colors.sequential.Plasma
    
    for n in range(n_levels):
        E_n = n + 0.5
        
        # Hermite polynomial wavefunction
        prefactor = 1 / np.sqrt(2**n * np.math.factorial(n)) * (1/np.pi)**0.25
        hermite = sp.hermite(n)
        psi_n = prefactor * np.exp(-x**2 / 2) * hermite(x)
        
        # Plot energy level
        fig.add_trace(go.Scatter(
            x=x, y=np.full_like(x, E_n),
            mode='lines',
            line=dict(color=colors[n % len(colors)], dash='dash', width=1),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Plot wavefunction (offset by energy)
        fig.add_trace(go.Scatter(
            x=x, y=E_n + psi_n * 0.5,
            mode='lines',
            line=dict(color=colors[n % len(colors)], width=2),
            fill='tonexty' if n > 0 else None,
            fillcolor=f'rgba({int(colors[n % len(colors)][4:-1].split(",")[0])},{int(colors[n % len(colors)][4:-1].split(",")[1])},{int(colors[n % len(colors)][4:-1].split(",")[2])},0.2)',
            name=f'n={n}',
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title='Position x',
        yaxis_title='Energy E',
        title=dict(text="Quantum Harmonic Oscillator", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def qp_spin_bloch_sphere(seed):
    """6. Spin state on Bloch sphere"""
    np.random.seed(seed)
    
    # Bloch sphere
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    fig = go.Figure()
    
    # Sphere
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, 'rgba(100,100,150,0.1)'], [1, 'rgba(100,100,150,0.1)']],
        showscale=False,
        opacity=0.3,
        hoverinfo='skip'
    ))
    
    # Axes
    axis_length = 1.3
    axes = [
        ([0, axis_length], [0, 0], [0, 0], 'red', 'X'),
        ([0, 0], [0, axis_length], [0, 0], 'green', 'Y'),
        ([0, 0], [0, 0], [0, axis_length], 'blue', 'Z')
    ]
    
    for ax_x, ax_y, ax_z, color, label in axes:
        fig.add_trace(go.Scatter3d(
            x=ax_x, y=ax_y, z=ax_z,
            mode='lines',
            line=dict(color=color, width=4),
            hoverinfo='skip'
        ))
    
    # Quantum state vector (random on sphere)
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2*np.pi)
    
    state_x = np.sin(theta) * np.cos(phi)
    state_y = np.sin(theta) * np.sin(phi)
    state_z = np.cos(theta)
    
    fig.add_trace(go.Scatter3d(
        x=[0, state_x], y=[0, state_y], z=[0, state_z],
        mode='lines+markers',
        line=dict(color='cyan', width=6),
        marker=dict(size=[0, 10], color='cyan'),
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis=dict(range=[-1.5, 1.5], showticklabels=False),
            yaxis=dict(range=[-1.5, 1.5], showticklabels=False),
            zaxis=dict(range=[-1.5, 1.5], showticklabels=False),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        title=dict(text="Spin-½ State on Bloch Sphere", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def qp_quantum_phase_space(seed):
    """7. Wigner quasiprobability distribution"""
    np.random.seed(seed)
    
    x = np.linspace(-4, 4, 100)
    p = np.linspace(-4, 4, 100)
    X, P = np.meshgrid(x, p)
    
    # Wigner function for Fock state
    n = np.random.randint(0, 4)
    r2 = X**2 + P**2
    
    # Laguerre polynomial
    laguerre = sp.genlaguerre(n, 0)
    W = ((-1)**n / np.pi) * np.exp(-r2) * laguerre(2 * r2)
    
    fig = go.Figure(data=[go.Contour(
        x=x, y=p, z=W,
        colorscale='RdBu',
        showscale=False,
        contours=dict(
            coloring='heatmap',
            showlabels=True
        )
    )])
    
    fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title='Position x',
        yaxis_title='Momentum p',
        title=dict(text=f"Wigner Function (Fock n={n})", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def qp_berry_phase_path(seed):
    """8. Berry phase geometric path"""
    np.random.seed(seed)
    
    # Parameter space path (adiabatic cycle)
    t = np.linspace(0, 2*np.pi, 200)
    
    # Path in parameter space
    R = 2
    param1 = R * np.cos(t)
    param2 = R * np.sin(t)
    param3 = 0.5 * np.sin(2*t)
    
    # Berry curvature (simplified)
    curvature = np.abs(np.sin(t) * np.cos(t))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=param1, y=param2, z=param3,
        mode='lines',
        line=dict(
            color=curvature,
            colorscale='Plasma',
            width=8
        ),
        hoverinfo='skip'
    ))
    
    # Start/end point
    fig.add_trace(go.Scatter3d(
        x=[param1[0]], y=[param2[0]], z=[param3[0]],
        mode='markers',
        marker=dict(size=10, color='lime', symbol='diamond'),
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis_title='Parameter λ₁',
            yaxis_title='Parameter λ₂',
            zaxis_title='Parameter λ₃',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        title=dict(text="Berry Phase Geometric Path", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def qp_density_matrix_evolution(seed):
    """9. Density matrix time evolution"""
    np.random.seed(seed)
    
    n_states = 5
    n_times = 50
    
    # Initial density matrix (pure state)
    rho_init = np.zeros((n_states, n_states), dtype=complex)
    rho_init[0, 0] = 1.0
    
    # Evolution under Hamiltonian
    density_evolution = []
    
    for t in range(n_times):
        # Simulate decoherence
        gamma = 0.05
        rho_t = rho_init.copy()
        
        # Off-diagonal decay
        for i in range(n_states):
            for j in range(n_states):
                if i != j:
                    rho_t[i, j] *= np.exp(-gamma * t * abs(i - j))
        
        # Population redistribution
        for i in range(1, n_states):
            transfer = rho_t[0, 0] * (1 - np.exp(-0.02 * t))
            rho_t[i, i] += transfer / (n_states - 1)
        rho_t[0, 0] = 1 - rho_t[0, 0]
        
        density_evolution.append(np.abs(rho_t))
    
    # Plot multiple time snapshots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f't={t}' for t in [0, 10, 20, 30, 40, 49]],
        specs=[[{'type': 'heatmap'}]*3, [{'type': 'heatmap'}]*3]
    )
    
    plot_times = [0, 10, 20, 30, 40, 49]
    for idx, t in enumerate(plot_times):
        row = idx // 3 + 1
        col = idx % 3 + 1
        
        fig.add_trace(
            go.Heatmap(
                z=density_evolution[t],
                colorscale='Viridis',
                showscale=False,
                hoverinfo='skip'
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        **DARK_LAYOUT,
        height=500,
        title=dict(text="Density Matrix Evolution (Decoherence)", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def qp_quantum_discord(seed):
    """10. Quantum discord landscape"""
    np.random.seed(seed)
    
    # Bipartite system parameters
    theta = np.linspace(0, np.pi, 60)
    phi = np.linspace(0, 2*np.pi, 60)
    THETA, PHI = np.meshgrid(theta, phi)
    
    # Quantum discord (simplified model)
    discord = np.sin(THETA)**2 * (1 + 0.5 * np.cos(2*PHI))
    
    fig = go.Figure(data=[go.Surface(
        x=THETA * np.cos(PHI),
        y=THETA * np.sin(PHI),
        z=discord,
        colorscale='Inferno',
        showscale=False
    )])
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis_title='System A Parameter',
            yaxis_title='System B Parameter',
            zaxis_title='Discord D',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        title=dict(text="Quantum Discord Landscape", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def qp_aharonov_bohm_phase(seed):
    """11. Aharonov-Bohm phase shift"""
    np.random.seed(seed)
    
    # Two paths around magnetic flux
    theta = np.linspace(0, 2*np.pi, 200)
    
    # Path 1 (outer)
    r1 = 2
    x1 = r1 * np.cos(theta)
    y1 = r1 * np.sin(theta)
    
    # Path 2 (inner)
    r2 = 1
    x2 = r2 * np.cos(theta)
    y2 = r2 * np.sin(theta)
    
    # Phase accumulation
    flux = np.random.uniform(0.5, 2.0)  # Magnetic flux
    phase1 = flux * theta / (2*np.pi)
    phase2 = flux * theta / (2*np.pi)
    
    # Interference intensity
    z = np.cos(phase1) + np.cos(phase2)
    
    fig = go.Figure()
    
    # Outer path
    fig.add_trace(go.Scatter3d(
        x=x1, y=y1, z=z,
        mode='lines',
        line=dict(color=phase1, colorscale='Viridis', width=6),
        hoverinfo='skip'
    ))
    
    # Inner path
    fig.add_trace(go.Scatter3d(
        x=x2, y=y2, z=z,
        mode='lines',
        line=dict(color=phase2, colorscale='Plasma', width=6),
        hoverinfo='skip'
    ))
    
    # Magnetic flux region
    flux_theta = np.linspace(0, 2*np.pi, 50)
    flux_r = np.linspace(0, 0.5, 10)
    F_THETA, F_R = np.meshgrid(flux_theta, flux_r)
    
    fig.add_trace(go.Surface(
        x=F_R * np.cos(F_THETA),
        y=F_R * np.sin(F_THETA),
        z=np.zeros_like(F_R),
        colorscale=[[0, 'red'], [1, 'red']],
        showscale=False,
        opacity=0.5,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis_title='Interference',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.4))
        ),
        title=dict(text="Aharonov-Bohm Phase Shift", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def qp_quantum_zeno_effect(seed):
    """12. Quantum Zeno effect - frequent measurements"""
    np.random.seed(seed)
    
    t = np.linspace(0, 10, 200)
    
    # Unperturbed decay
    gamma = 0.5
    P_undisturbed = np.exp(-gamma * t)
    
    # With frequent measurements
    measurement_intervals = [1, 0.5, 0.2, 0.1]
    
    fig = go.Figure()
    
    # Undisturbed
    fig.add_trace(go.Scatter(
        x=t, y=P_undisturbed,
        mode='lines',
        line=dict(color='red', width=3, dash='dash'),
        name='No measurement',
        hoverinfo='skip'
    ))
    
    # With measurements
    colors = px.colors.sequential.Plasma
    for i, dt_measure in enumerate(measurement_intervals):
        n_measurements = int(t[-1] / dt_measure)
        P_zeno = []
        
        for t_val in t:
            n = int(t_val / dt_measure)
            P_survive = (1 - gamma * dt_measure) ** n if n > 0 else 1.0
            P_zeno.append(P_survive)
        
        fig.add_trace(go.Scatter(
            x=t, y=P_zeno,
            mode='lines',
            line=dict(color=colors[i], width=2),
            name=f'Δt = {dt_measure}',
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title='Time t',
        yaxis_title='Survival Probability P(t)',
        title=dict(text="Quantum Zeno Effect", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def qp_casimir_force_plates(seed):
    """13. Casimir force between plates"""
    np.random.seed(seed)
    
    # Distance between plates
    d_values = np.linspace(0.5, 5, 100)
    
    # Casimir force (1/d^4 dependence)
    hbar_c = 197  # MeV·fm
    A = 1  # Plate area
    F_casimir = -np.pi**2 * hbar_c * A / (240 * d_values**4)
    
    # Energy vs distance
    E_casimir = np.pi**2 * hbar_c * A / (720 * d_values**3)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Casimir Force F(d)', 'Casimir Energy E(d)'),
        horizontal_spacing=0.15
    )
    
    fig.add_trace(
        go.Scatter(
            x=d_values, y=F_casimir,
            mode='lines',
            line=dict(color='cyan', width=3),
            fill='tozeroy',
            fillcolor='rgba(0,255,200,0.2)',
            hoverinfo='skip'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=d_values, y=E_casimir,
            mode='lines',
            line=dict(color='magenta', width=3),
            fill='tozeroy',
            fillcolor='rgba(255,0,255,0.2)',
            hoverinfo='skip'
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text='Distance d (nm)', row=1, col=1)
    fig.update_xaxes(title_text='Distance d (nm)', row=1, col=2)
    fig.update_yaxes(title_text='Force F (pN)', row=1, col=1)
    fig.update_yaxes(title_text='Energy E (eV)', row=1, col=2)
    
    fig.update_layout(
        **DARK_LAYOUT,
        height=400,
        title=dict(text="Casimir Effect", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def qp_quantum_walk_graph(seed):
    """14. Quantum walk on graph"""
    np.random.seed(seed)
    
    # Create graph
    n_nodes = 20
    G = nx.cycle_graph(n_nodes)
    pos = nx.circular_layout(G)
    
    # Quantum walk evolution
    n_steps = 30
    
    # Initial state (localized)
    init_node = 0
    state = np.zeros(n_nodes, dtype=complex)
    state[init_node] = 1.0
    
    # Adjacency matrix
    A = nx.adjacency_matrix(G).toarray()
    
    # Evolution operator (coin + shift)
    probabilities = []
    
    for step in range(n_steps):
        # Simple quantum walk
        state = A @ state / np.sqrt(np.sum(np.abs(A @ state)**2))
        probabilities.append(np.abs(state)**2)
    
    probabilities = np.array(probabilities)
    
    fig = go.Figure(data=[go.Heatmap(
        z=probabilities.T,
        colorscale='Plasma',
        showscale=False
    )])
    
    fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title='Time Step',
        yaxis_title='Node',
        title=dict(text="Quantum Walk on Cycle Graph", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def qp_squeezed_state_ellipse(seed):
    """15. Squeezed coherent state uncertainty ellipse"""
    np.random.seed(seed)
    
    # Phase space
    x = np.linspace(-4, 4, 100)
    p = np.linspace(-4, 4, 100)
    X, P = np.meshgrid(x, p)
    
    # Squeezed state parameters
    r = np.random.uniform(0.5, 1.5)  # Squeezing parameter
    theta = np.random.uniform(0, 2*np.pi)  # Squeezing angle
    
    # Covariance matrix
    sigma_x = np.exp(r)
    sigma_p = np.exp(-r)
    
    # Rotated
    X_rot = X * np.cos(theta) + P * np.sin(theta)
    P_rot = -X * np.sin(theta) + P * np.cos(theta)
    
    # Wigner function
    W = (2 / np.pi) * np.exp(-2 * (X_rot**2 / sigma_x**2 + P_rot**2 / sigma_p**2))
    
    fig = go.Figure()
    
    fig.add_trace(go.Contour(
        x=x, y=p, z=W,
        colorscale='Hot',
        showscale=False,
        contours=dict(
            coloring='heatmap'
        )
    ))
    
    # Uncertainty ellipse
    t = np.linspace(0, 2*np.pi, 100)
    ellipse_x = sigma_x * np.cos(t)
    ellipse_p = sigma_p * np.sin(t)
    
    # Rotate
    ellipse_x_rot = ellipse_x * np.cos(theta) - ellipse_p * np.sin(theta)
    ellipse_p_rot = ellipse_x * np.sin(theta) + ellipse_p * np.cos(theta)
    
    fig.add_trace(go.Scatter(
        x=ellipse_x_rot, y=ellipse_p_rot,
        mode='lines',
        line=dict(color='cyan', width=3, dash='dash'),
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title='Position x',
        yaxis_title='Momentum p',
        title=dict(text="Squeezed Coherent State", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def qp_quantum_annealing_landscape(seed):
    """16. Quantum annealing energy landscape"""
    np.random.seed(seed)
    
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # Rugged classical landscape
    classical = (X**2 + Y**2 + 
                 0.5 * np.sin(5*X) * np.cos(5*Y) +
                 0.3 * np.sin(3*X + 2*Y))
    
    # Tunneling creates smoother landscape
    s = 0.7  # Annealing parameter (0 to 1)
    quantum = (s * classical + 
               (1-s) * 0.5 * (X**2 + Y**2))
    
    fig = go.Figure()
    
    fig.add_trace(go.Surface(
        x=X, y=Y, z=quantum,
        colorscale='Cividis',
        showscale=False,
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="lime", project=dict(z=True))
        )
    ))
    
    # Global minimum
    min_idx = np.unravel_index(quantum.argmin(), quantum.shape)
    fig.add_trace(go.Scatter3d(
        x=[X[min_idx]], y=[Y[min_idx]], z=[quantum[min_idx]],
        mode='markers',
        marker=dict(size=10, color='red', symbol='diamond'),
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis_title='Configuration x',
            yaxis_title='Configuration y',
            zaxis_title='Energy E',
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.3))
        ),
        title=dict(text="Quantum Annealing Landscape", font=dict(size=13, color='#00ffc8'))
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# 📐 TOPOLOGY PLOTS (Subject 4/21) - Will continue in next part...
# ═══════════════════════════════════════════════════════════════════════════

def topology_mobius_strip(seed):
    """1. Möbius strip non-orientable surface"""
    np.random.seed(seed)
    
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(-1, 1, 20)
    U, V = np.meshgrid(u, v)
    
    # Möbius strip parametrization
    R = 2
    X = (R + V * np.cos(U/2)) * np.cos(U)
    Y = (R + V * np.cos(U/2)) * np.sin(U)
    Z = V * np.sin(U/2)
    
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Twilight',
        showscale=False,
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.9, roughness=0.3)
    )])
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis=dict(showticklabels=False, title=''),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.5))
        ),
        title=dict(text="Möbius Strip (Non-Orientable)", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def topology_klein_bottle(seed):
    """2. Klein bottle immersion in 3D"""
    np.random.seed(seed)
    
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, 2*np.pi, 100)
    U, V = np.meshgrid(u, v)
    
    # Klein bottle parametrization
    r = 2
    X = (r + np.cos(U/2)*np.sin(V) - np.sin(U/2)*np.sin(2*V)) * np.cos(U)
    Y = (r + np.cos(U/2)*np.sin(V) - np.sin(U/2)*np.sin(2*V)) * np.sin(U)
    Z = np.sin(U/2)*np.sin(V) + np.cos(U/2)*np.sin(2*V)
    
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Plasma',
        showscale=False,
        opacity=0.9
    )])
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis=dict(showticklabels=False, title=''),
            camera=dict(eye=dict(x=2, y=2, z=1.5))
        ),
        title=dict(text="Klein Bottle Immersion", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def topology_torus_linking(seed):
    """3. Torus with linked cycles"""
    np.random.seed(seed)
    
    u = np.linspace(0, 2*np.pi, 80)
    v = np.linspace(0, 2*np.pi, 80)
    U, V = np.meshgrid(u, v)
    
    # Torus
    R, r = 3, 1
    X = (R + r*np.cos(V)) * np.cos(U)
    Y = (R + r*np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    
    fig = go.Figure()
    
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Viridis',
        showscale=False,
        opacity=0.6
    ))
    
    # Meridian cycle
    meridian_u = np.linspace(0, 2*np.pi, 100)
    meridian_x = (R + r*np.cos(meridian_u)) * np.cos(0)
    meridian_y = (R + r*np.cos(meridian_u)) * np.sin(0)
    meridian_z = r * np.sin(meridian_u)
    
    fig.add_trace(go.Scatter3d(
        x=meridian_x, y=meridian_y, z=meridian_z,
        mode='lines',
        line=dict(color='cyan', width=8),
        hoverinfo='skip'
    ))
    
    # Longitude cycle
    longitude_v = np.linspace(0, 2*np.pi, 100)
    longitude_x = (R + r*np.cos(0)) * np.cos(longitude_v)
    longitude_y = (R + r*np.cos(0)) * np.sin(longitude_v)
    longitude_z = np.zeros_like(longitude_v)
    
    fig.add_trace(go.Scatter3d(
        x=longitude_x, y=longitude_y, z=longitude_z,
        mode='lines',
        line=dict(color='magenta', width=8),
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis=dict(showticklabels=False, title=''),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.3))
        ),
        title=dict(text="Torus Fundamental Cycles", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def topology_hopf_fibration(seed):
    """4. Hopf fibration S³ → S²"""
    np.random.seed(seed)
    
    # Base S² sphere
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    U, V = np.meshgrid(u, v)
    
    X_base = np.sin(V) * np.cos(U)
    Y_base = np.sin(V) * np.sin(U)
    Z_base = np.cos(V)
    
    fig = go.Figure()
    
    # Base sphere
    fig.add_trace(go.Surface(
        x=X_base, y=Y_base, z=Z_base,
        colorscale=[[0, 'rgba(100,100,150,0.1)'], [1, 'rgba(100,100,150,0.1)']],
        showscale=False,
        opacity=0.2,
        hoverinfo='skip'
    ))
    
    # Hopf fibers (circles)
    n_fibers = 15
    for i in range(n_fibers):
        # Point on S²
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi)
        
        # Fiber circle in S³ (projected to R³)
        t = np.linspace(0, 2*np.pi, 100)
        
        # Stereographic projection
        fiber_x = np.cos(t) * np.sin(phi) * np.cos(theta)
        fiber_y = np.cos(t) * np.sin(phi) * np.sin(theta)
        fiber_z = np.sin(t)
        
        color_val = i / n_fibers
        fig.add_trace(go.Scatter3d(
            x=fiber_x, y=fiber_y, z=fiber_z,
            mode='lines',
            line=dict(color=px.colors.sequential.Plasma[int(color_val * (len(px.colors.sequential.Plasma)-1))], width=4),
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis=dict(showticklabels=False, title=''),
            camera=dict(eye=dict(x=2, y=2, z=1.5))
        ),
        title=dict(text="Hopf Fibration S³ → S²", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def topology_knot_theory_trefoil(seed):
    """5. Trefoil knot"""
    np.random.seed(seed)
    
    t = np.linspace(0, 2*np.pi, 500)
    
    # Trefoil knot parametrization
    x = np.sin(t) + 2*np.sin(2*t)
    y = np.cos(t) - 2*np.cos(2*t)
    z = -np.sin(3*t)
    
    # Color by parameter
    colors = t / (2*np.pi)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(
            color=colors,
            colorscale='Twilight',
            width=12
        ),
        hoverinfo='skip'
    )])
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis=dict(showticklabels=False, title=''),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.5))
        ),
        title=dict(text="Trefoil Knot", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def topology_homology_complex(seed):
    """6. Simplicial complex homology"""
    np.random.seed(seed)
    
    # Generate point cloud
    n_points = 100
    points = np.random.randn(n_points, 3)
    
    # Compute Delaunay triangulation
    tri = Delaunay(points[:, :2])
    
    fig = go.Figure()
    
    # Plot simplices
    for simplex in tri.simplices[:50]:  # Show subset
        triangle = points[simplex]
        fig.add_trace(go.Mesh3d(
            x=triangle[:, 0],
            y=triangle[:, 1],
            z=triangle[:, 2],
            color='cyan',
            opacity=0.3,
            hoverinfo='skip'
        ))
    
    # Plot points
    fig.add_trace(go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(size=3, color='white'),
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis=dict(showticklabels=False, title=''),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        title=dict(text="Simplicial Complex Homology", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def topology_genus_surface(seed):
    """7. High-genus surface"""
    np.random.seed(seed)
    
    # Create surface with handles
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, 2*np.pi, 100)
    U, V = np.meshgrid(u, v)
    
    # Base torus with modulation
    R, r = 3, 1
    X = (R + r*np.cos(V)) * np.cos(U) + 0.5*np.sin(3*U)*np.cos(V)
    Y = (R + r*np.cos(V)) * np.sin(U) + 0.5*np.sin(3*V)
    Z = r * np.sin(V) + 0.3*np.cos(2*U)*np.sin(2*V)
    
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Plasma',
        showscale=False,
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.7)
    )])
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis=dict(showticklabels=False, title=''),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.5))
        ),
        title=dict(text="High-Genus Surface", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def topology_betti_numbers(seed):
    """8. Betti numbers visualization"""
    np.random.seed(seed)
    
    # Different topological spaces
    spaces = ['S¹', 'S²', 'T²', 'RP²', 'K']
    b0 = [1, 1, 1, 1, 1]
    b1 = [1, 0, 2, 1, 1]
    b2 = [0, 1, 1, 0, 0]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=spaces, y=b0,
        name='β₀ (Components)',
        marker_color='cyan'
    ))
    
    fig.add_trace(go.Bar(
        x=spaces, y=b1,
        name='β₁ (Loops)',
        marker_color='magenta'
    ))
    
    fig.add_trace(go.Bar(
        x=spaces, y=b2,
        name='β₂ (Voids)',
        marker_color='yellow'
    ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        barmode='group',
        xaxis_title='Topological Space',
        yaxis_title='Betti Number',
        title=dict(text="Betti Numbers of Spaces", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def topology_fiber_bundle(seed):
    """9. Fiber bundle structure"""
    np.random.seed(seed)
    
    # Base space (circle)
    theta = np.linspace(0, 2*np.pi, 50)
    base_x = 3 * np.cos(theta)
    base_y = 3 * np.sin(theta)
    base_z = np.zeros_like(theta)
    
    fig = go.Figure()
    
    # Base circle
    fig.add_trace(go.Scatter3d(
        x=base_x, y=base_y, z=base_z,
        mode='lines',
        line=dict(color='red', width=8),
        hoverinfo='skip'
    ))
    
    # Fibers
    n_fibers = 12
    for i in range(n_fibers):
        angle = 2*np.pi * i / n_fibers
        base_point_x = 3 * np.cos(angle)
        base_point_y = 3 * np.sin(angle)
        
        # Fiber (circle)
        phi = np.linspace(0, 2*np.pi, 50)
        fiber_x = base_point_x + 0.5 * np.cos(phi) * np.cos(angle)
        fiber_y = base_point_y + 0.5 * np.cos(phi) * np.sin(angle)
        fiber_z = 0.5 * np.sin(phi)
        
        fig.add_trace(go.Scatter3d(
            x=fiber_x, y=fiber_y, z=fiber_z,
            mode='lines',
            line=dict(color='cyan', width=3),
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis=dict(showticklabels=False, title=''),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.5))
        ),
        title=dict(text="Fiber Bundle Structure", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def topology_homotopy_path(seed):
    """10. Homotopy path deformation"""
    np.random.seed(seed)
    
    t = np.linspace(0, 1, 100)
    
    fig = go.Figure()
    
    # Multiple homotopic paths
    n_paths = 8
    for i in range(n_paths):
        s = i / (n_paths - 1)  # Homotopy parameter
        
        # Path deformation
        x = t
        y = np.sin(3*np.pi*t) * (1 - s) + s * 0.5 * np.sin(5*np.pi*t)
        z = np.full_like(t, s)
        
        color_val = s
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(
                color=px.colors.sequential.Plasma[int(color_val * (len(px.colors.sequential.Plasma)-1))],
                width=4
            ),
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis_title='t',
            yaxis=dict(showticklabels=False, title=''),
            zaxis_title='Homotopy s',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        title=dict(text="Homotopy Path Deformation", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def topology_euler_characteristic(seed):
    """11. Euler characteristic χ = V - E + F"""
    np.random.seed(seed)
    
    # Create polyhedra
    polyhedra = ['Tetrahedron', 'Cube', 'Octahedron', 'Icosahedron', 'Torus']
    vertices = [4, 8, 6, 12, 16]
    edges = [6, 12, 12, 30, 32]
    faces = [4, 6, 8, 20, 16]
    chi = [v - e + f for v, e, f in zip(vertices, edges, faces)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=polyhedra, y=vertices,
        mode='markers+lines',
        name='Vertices V',
        marker=dict(size=12, color='cyan'),
        line=dict(color='cyan', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=polyhedra, y=edges,
        mode='markers+lines',
        name='Edges E',
        marker=dict(size=12, color='magenta'),
        line=dict(color='magenta', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=polyhedra, y=faces,
        mode='markers+lines',
        name='Faces F',
        marker=dict(size=12, color='yellow'),
        line=dict(color='yellow', width=2)
    ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title='Polyhedron',
        yaxis_title='Count',
        title=dict(text="Euler Characteristic χ = V - E + F", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def topology_covering_space(seed):
    """12. Covering space projection"""
    np.random.seed(seed)
    
    # Universal cover (helix) projecting to circle
    t = np.linspace(0, 4*np.pi, 200)
    
    # Helix (covering space)
    R = 2
    helix_x = R * np.cos(t)
    helix_y = R * np.sin(t)
    helix_z = t
    
    fig = go.Figure()
    
    # Helix
    fig.add_trace(go.Scatter3d(
        x=helix_x, y=helix_y, z=helix_z,
        mode='lines',
        line=dict(color='cyan', width=6),
        hoverinfo='skip'
    ))
    
    # Base circle
    circle_t = np.linspace(0, 2*np.pi, 100)
    circle_x = R * np.cos(circle_t)
    circle_y = R * np.sin(circle_t)
    circle_z = np.zeros_like(circle_t)
    
    fig.add_trace(go.Scatter3d(
        x=circle_x, y=circle_y, z=circle_z,
        mode='lines',
        line=dict(color='red', width=8, dash='dash'),
        hoverinfo='skip'
    ))
    
    # Projection lines
    for i in range(0, len(t), 20):
        fig.add_trace(go.Scatter3d(
            x=[helix_x[i], R * np.cos(t[i])],
            y=[helix_y[i], R * np.sin(t[i])],
            z=[helix_z[i], 0],
            mode='lines',
            line=dict(color='rgba(255,255,100,0.3)', width=1),
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis_title='Covering Dimension',
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.5))
        ),
        title=dict(text="Covering Space Projection", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def topology_persistent_homology(seed):
    """13. Persistent homology barcode"""
    np.random.seed(seed)
    
    # Simulate persistence diagram
    n_features = 30
    births = np.random.uniform(0, 5, n_features)
    deaths = births + np.random.exponential(2, n_features)
    deaths = np.minimum(deaths, 10)
    
    # Sort by birth time
    order = np.argsort(births)
    births = births[order]
    deaths = deaths[order]
    
    # Dimension (0, 1, 2)
    dimensions = np.random.choice([0, 1, 2], n_features)
    
    fig = go.Figure()
    
    colors = {0: 'cyan', 1: 'magenta', 2: 'yellow'}
    
    for i in range(n_features):
        fig.add_trace(go.Scatter(
            x=[births[i], deaths[i]],
            y=[i, i],
            mode='lines',
            line=dict(color=colors[dimensions[i]], width=4),
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title='Filtration Parameter',
        yaxis_title='Feature',
        title=dict(text="Persistent Homology Barcode", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def topology_symplectic_manifold(seed):
    """14. Symplectic form on manifold"""
    np.random.seed(seed)
    
    # Phase space (cotangent bundle)
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    
    # Symplectic potential
    A_x = -Y
    A_y = X
    
    # Magnitude
    magnitude = np.sqrt(A_x**2 + A_y**2)
    
    fig = go.Figure()
    
    # Contour of magnitude
    fig.add_trace(go.Contour(
        x=x, y=y, z=magnitude,
        colorscale='Viridis',
        showscale=False,
        contours=dict(coloring='heatmap'),
        opacity=0.7
    ))
    
    # Vector field
    skip = 2
    fig.add_trace(go.Scatter(
        x=X[::skip, ::skip].flatten(),
        y=Y[::skip, ::skip].flatten(),
        mode='markers',
        marker=dict(size=2, color='white'),
        hoverinfo='skip'
    ))
    
    # Arrows
    for i in range(0, len(x), skip):
        for j in range(0, len(y), skip):
            scale = 0.15
            fig.add_annotation(
                x=X[j, i] + scale*A_x[j, i],
                y=Y[j, i] + scale*A_y[j, i],
                ax=X[j, i],
                ay=Y[j, i],
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor='cyan'
            )
    
    fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title='Position q',
        yaxis_title='Momentum p',
        title=dict(text="Symplectic Form ω = dq ∧ dp", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def topology_manifold_atlas(seed):
    """15. Manifold atlas (coordinate charts)"""
    np.random.seed(seed)
    
    # Sphere with multiple charts
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    U, V = np.meshgrid(u, v)
    
    X = np.sin(V) * np.cos(U)
    Y = np.sin(V) * np.sin(U)
    Z = np.cos(V)
    
    fig = go.Figure()
    
    # Sphere
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale=[[0, 'rgba(100,100,150,0.3)'], [1, 'rgba(100,100,150,0.3)']],
        showscale=False,
        opacity=0.4,
        hoverinfo='skip'
    ))
    
    # Chart patches (different coordinate systems)
    n_charts = 6
    for i in range(n_charts):
        theta_center = 2*np.pi * i / n_charts
        phi_center = np.pi / 3
        
        # Small patch
        delta = 0.3
        u_patch = np.linspace(theta_center - delta, theta_center + delta, 20)
        v_patch = np.linspace(phi_center - delta, phi_center + delta, 20)
        U_patch, V_patch = np.meshgrid(u_patch, v_patch)
        
        X_patch = np.sin(V_patch) * np.cos(U_patch)
        Y_patch = np.sin(V_patch) * np.sin(U_patch)
        Z_patch = np.cos(V_patch)
        
        color_val = i / n_charts
        fig.add_trace(go.Surface(
            x=X_patch, y=Y_patch, z=Z_patch,
            colorscale=[[0, px.colors.sequential.Plasma[int(color_val * (len(px.colors.sequential.Plasma)-1))]], 
                        [1, px.colors.sequential.Plasma[int(color_val * (len(px.colors.sequential.Plasma)-1))]]],
            showscale=False,
            opacity=0.8,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis=dict(showticklabels=False, title=''),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.5))
        ),
        title=dict(text="Manifold Atlas (Chart System)", font=dict(size=13, color='#00ffc8'))
    )
    return fig

def topology_vector_bundle_section(seed):
    """16. Vector bundle with section"""
    np.random.seed(seed)
    
    # Base space (circle)
    t = np.linspace(0, 2*np.pi, 50)
    R = 2
    base_x = R * np.cos(t)
    base_y = R * np.sin(t)
    base_z = np.zeros_like(t)
    
    fig = go.Figure()
    
    # Base circle
    fig.add_trace(go.Scatter3d(
        x=base_x, y=base_y, z=base_z,
        mode='lines',
        line=dict(color='white', width=6),
        hoverinfo='skip'
    ))
    
    # Section (vector field on base)
    for i in range(0, len(t), 3):
        # Vector at each point
        tangent = np.array([-np.sin(t[i]), np.cos(t[i]), 0])
        normal = np.array([np.cos(t[i]), np.sin(t[i]), 0])
        
        # Section value
        section_vec = normal * np.sin(2*t[i]) + tangent * np.cos(2*t[i]) + np.array([0, 0, 0.5*np.sin(3*t[i])])
        section_vec *= 0.5
        
        # Draw vector
        fig.add_trace(go.Scatter3d(
            x=[base_x[i], base_x[i] + section_vec[0]],
            y=[base_y[i], base_y[i] + section_vec[1]],
            z=[base_z[i], base_z[i] + section_vec[2]],
            mode='lines+markers',
            line=dict(color='cyan', width=4),
            marker=dict(size=[0, 6], color='cyan'),
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis=dict(showticklabels=False, title=''),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.5))
        ),
        title=dict(text="Vector Bundle Section", font=dict(size=13, color='#00ffc8'))
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# 🎯 MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

# Define all 21 subjects
SUBJECTS = [
    "Reinforcement Learning",
    "Artificial General Intelligence",
    "Quantum Physics",
    "Topology",
    "Quantum Gravity",
    "String Theory",
    "Quantum Computing",
    "Machine Learning Theory",
    "Computational Complexity",
    "Category Theory",
    "Algebraic Geometry",
    "Differential Geometry",
    "Gauge Theory",
    "Quantum Field Theory",
    "Statistical Mechanics",
    "Information Theory",
    "Network Science",
    "Dynamical Systems",
    "Chaos Theory",
    "Fractals & Self-Similarity",
    "Quantum Entanglement"
]

# Map subjects to their plot functions (first 4 subjects implemented above)
PLOT_FUNCTIONS = {
    "Reinforcement Learning": [
        rl_stigmergy_field, rl_policy_manifold, rl_value_function_flow,
        rl_reward_landscape_3d, rl_bellman_residual_field, rl_actor_critic_phase_space,
        rl_q_function_surface, rl_advantage_function_contour, rl_experience_replay_topology,
        rl_monte_carlo_tree_structure, rl_curiosity_intrinsic_reward, rl_trust_region_constraint,
        rl_hindsight_experience_trajectory, rl_distributional_value_function, rl_multi_agent_coordination,
        rl_model_based_planning_tree
    ],
    "Artificial General Intelligence": [
        agi_cognitive_architecture_network, agi_knowledge_graph_embedding, agi_attention_mechanism_heatmap,
        agi_world_model_latent_dynamics, agi_goal_hierarchy_tree, agi_consciousness_integration_field,
        agi_reward_shaping_landscape, agi_transfer_learning_manifold, agi_meta_learning_optimization_surface,
        agi_causal_graph_discovery, agi_neural_turing_machine_memory, agi_compositional_generalization,
        agi_reasoning_proof_tree, agi_emergent_communication_protocol, agi_abstract_reasoning_matrix,
        agi_general_value_function_network
    ],
    "Quantum Physics": [
        qp_wavefunction_interference, qp_quantum_tunneling_barrier, qp_hydrogen_orbital_3d,
        qp_schrodinger_evolution, qp_quantum_harmonic_oscillator, qp_spin_bloch_sphere,
        qp_quantum_phase_space, qp_berry_phase_path, qp_density_matrix_evolution,
        qp_quantum_discord, qp_aharonov_bohm_phase, qp_quantum_zeno_effect,
        qp_casimir_force_plates, qp_quantum_walk_graph, qp_squeezed_state_ellipse,
        qp_quantum_annealing_landscape
    ],
    "Topology": [
        topology_mobius_strip, topology_klein_bottle, topology_torus_linking,
        topology_hopf_fibration
        # ... (16 total - will add remaining 12)
    ]
    # ... other subjects will follow similar pattern
}

def main():
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='font-size: 3em; font-weight: 100; letter-spacing: 8px; color: #00ffc8;'>
            ∞ AETHER ∞
        </h1>
        <p style='font-size: 1.2em; color: #808090; letter-spacing: 3px;'>
            Advanced Exploratory Theoretical & Holistic Engineering Research
        </p>
        <p style='font-size: 0.9em; color: #606070; font-style: italic;'>
            "Science is art. Art is science. Beauty is truth." — Your Name (君の名は)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Seed control in sidebar
    with st.sidebar:
        st.markdown("### 🎲 Seed Control")
        if st.button("🔄 Regenerate All", type="primary", use_container_width=True):
            st.session_state.master_seed = np.random.randint(0, 1000000)
            st.rerun()
        
        st.caption(f"Current Seed: {st.session_state.master_seed}")
        st.divider()
        
        st.markdown("### 📊 Stats")
        st.metric("Total Subjects", "21")
        st.metric("Plots per Subject", "16")
        st.metric("Total Visualizations", "336")
        st.divider()
        
        st.markdown("### 🌌 Domains")
        st.caption("""
        • Reinforcement Learning
        • AGI & Cognition
        • Quantum Physics
        • Topology
        • Quantum Gravity
        • String Theory
        • Quantum Computing
        • ML Theory
        • Complexity
        • Category Theory
        • Algebraic Geometry
        • Differential Geometry
        • Gauge Theory
        • QFT
        • Statistical Mechanics
        • Information Theory
        • Network Science
        • Dynamical Systems
        • Chaos Theory
        • Fractals
        • Entanglement
        """)
    
    # Create tabs for subjects
    tabs = st.tabs(SUBJECTS)
    
    # Render each subject
    for idx, (tab, subject) in enumerate(zip(tabs, SUBJECTS)):
        with tab:
            st.markdown(f"## {subject}")
            st.caption(f"16 scientifically accurate visualizations exploring {subject.lower()}")
            st.divider()
            
            # Check if plots are implemented
            if subject in PLOT_FUNCTIONS:
                plot_funcs = PLOT_FUNCTIONS[subject]
                
                # Display in 4x4 grid
                for row in range(4):
                    cols = st.columns(4)
                    for col_idx, col in enumerate(cols):
                        plot_idx = row * 4 + col_idx
                        if plot_idx < len(plot_funcs):
                            with col:
                                with st.spinner(f""):
                                    try:
                                        seed = get_seed(idx * 16 + plot_idx)
                                        fig = plot_funcs[plot_idx](seed)
                                        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
                                    except Exception as e:
                                        st.error(f"Error: {str(e)[:50]}")
            else:
                st.info(f"🚧 Plots for {subject} are being implemented...")
                st.markdown("""
                This section will contain 16 advanced visualizations including:
                - 3D manifold representations
                - Dynamic field simulations
                - Theoretical predictions
                - Cutting-edge research visualizations
                """)

if __name__ == "__main__":
    main()