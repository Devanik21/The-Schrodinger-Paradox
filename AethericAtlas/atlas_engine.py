import numpy as np
import matplotlib.pyplot as plt
import io
import time
from scipy.fft import fft2, fftshift

class AethericEngine:
    """
    16-Lens Observational System for 21 STEM Vectors.
    Generates 336 unique, physically grounded latent projections.
    """
    
    def __init__(self):
        self.subjects = [
            "Hamiltonian Potential Well", "MCMC Walker Topology", "Hutchinson Laplacian Curvature",
            "Log-Domain Slater Nodes", "Kato Cusp Enforcement", "Fisher Information Manifold",
            "Atomic Shell Structure", "Molecular PES Landscape", "SSM-Backflow Data Channels",
            "Flow-VMC Acceptance Field", "Excited State Orthogonality", "Topological Berry Flow",
            "TD-VMC Quantum Dynamics", "Bloch Periodic Lattice", "Relativistic Spin-Orbit Split",
            "Entanglement Entropy Mesh", "Noether Discovery Landscape", "Stigmergic Latent Bloom",
            "Global Knowledge Meme Grid", "N-Body Correlation Mesh", "Master Consensus Field"
        ]

    def apply_stigmergy_effect(self, grid, blur_iterations=3, bloom_intensity=1.2):
        for _ in range(blur_iterations):
            grid = (grid + 
                    np.roll(grid, 1, axis=0) * 0.2 + np.roll(grid, -1, axis=0) * 0.2 + 
                    np.roll(grid, 1, axis=1) * 0.2 + np.roll(grid, -1, axis=1) * 0.2) / 1.8
        grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
        return np.clip(grid * bloom_intensity, 0, 1) ** 1.3

    def get_base_field(self, subject_idx, seed=42):
        """Generates a high-res base physical tensor (Z) for the given subject."""
        np.random.seed(seed)
        res = 120
        x = np.linspace(-4, 4, res); y = np.linspace(-4, 4, res)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2) + 0.05
        
        # Subject-specific base physics
        if subject_idx == 0: # Hamiltonian Well
            Z = -2.0/R + 0.3*np.sin(R*3) * np.exp(-R*0.5)
        elif subject_idx == 1: # MCMC Walker
            Z = np.zeros_like(X)
            for _ in range(5): 
                cx, cy = np.random.randn(2)*1.5
                Z += np.exp(-((X-cx)**2 + (Y-cy)**2)/0.4)
        elif subject_idx == 2: # Laplacian
            Z = np.exp(-(X**2 + Y**2)*0.3) * np.sin(X*3) * np.cos(Y*3)
        elif subject_idx == 3: # Slater Nodes
            Z = np.sin(X*2)*np.cos(Y*1.5) - np.sin(Y*2)*np.cos(X*1.5)
        elif subject_idx == 4: # Kato Cusp
            Z = np.exp(-2.0*R) * (1 - R*0.5) + 0.3*np.exp(-((R-1.5)**2)/0.1)
        elif subject_idx == 5: # Fisher Manifold
            Z = np.sin(X*2) * np.sin(Y*2) + np.cos((X+Y)*1.5)
        elif subject_idx == 6: # Atomic Shells
            theta = np.arctan2(Y, X)
            Z = np.exp(-R)*4 + np.exp(-(R-1.5)**2)*2*np.cos(theta)**2
        elif subject_idx == 7: # PES Landscape
            Z = 5*(1-np.exp(-1.2*(np.sqrt((X-0.7)**2+Y**2)-1.4)))**2
        elif subject_idx == 8: # SSM Dataflow
            Z = np.exp(-0.5*np.abs(X-Y)) * np.sin(1.5*X)
        elif subject_idx == 9: # Flow Acceptance
            Z = 0.5*np.exp(-R*0.5) + 0.5*np.exp(-((X-1.5)**2+Y**2)*0.8)
        elif subject_idx == 10: # Ortho Pressure
            Z = np.exp(-(R-1.5)**2 / 0.2) + np.exp(-(R-0.5)**2 / 0.1)
        elif subject_idx == 11: # Berry Flow
            Z = np.arctan2(Y, X) # Phase field
        elif subject_idx == 12: # TD-VMC
            Z = np.exp(-R*0.3) * np.cos(X*2)
        elif subject_idx == 13: # Bloch Lattice
            Z = -(np.cos(np.pi*X) + np.cos(np.pi*Y))
        elif subject_idx == 14: # Spin-Orbit
            Z = np.exp(-R*1.2) * (np.cos(np.arctan2(Y,X)) + np.sin(np.arctan2(Y,X)))
        elif subject_idx == 15: # Entanglement
            Z = np.sin(X*3)**2 * np.cos(Y*3)**2 + np.exp(-R**2)
        elif subject_idx == 16: # Noether
            Z = np.abs(np.sin(X*Y)*0.5 + 0.5)
        elif subject_idx == 17: # Latent Bloom
            Z = np.exp(-R**2/2) * (1 + 0.5*np.sin(X*4)*np.cos(Y*4))
        elif subject_idx == 18: # Meme Grid
            Z = (np.random.rand(res, res) > 0.95).astype(float)
        elif subject_idx == 19: # Correlation Mesh
            Z = 1.0 - (np.exp(-1.5*(X-1)**2 - 1.5*(Y-1)**2) + np.exp(-1.5*(X+1)**2 - 1.5*(Y+1)**2))
        else: # Master Synth.
            Z = np.sin(X*1.5)*np.cos(Y*1.5) + np.cos(np.sqrt(X**2+Y**2)*3.0)*0.3
            
        return X, Y, Z

    def apply_lens(self, X, Y, Z, lens_idx, subject_name, seed=42):
        """Transforms a physical tensor into one of 16 'Observational Lenses'."""
        fig = plt.figure(figsize=(7, 7), facecolor='#0e1117')
        color_maps = ['magma', 'inferno', 'viridis', 'plasma', 'twilight', 'cool', 'hot', 'ocean', 'cividis', 'gnuplot2', 'hsv', 'terrain', 'cubehelix_r', 'RdBu_r', 'Spectral', 'prism']
        cmap = color_maps[lens_idx % len(color_maps)]
        
        if lens_idx == 0: # Primary Proj.
            ax = fig.add_subplot(111)
            grid = plt.get_cmap(cmap)((Z - Z.min()) / (Z.max() - Z.min() + 1e-8))[:,:,:3]
            ax.imshow(self.apply_stigmergy_effect(grid), interpolation='bicubic')
            ax.axis('off')
        
        elif lens_idx == 1: # 3D Manifold
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Z, cmap=cmap, antialiased=True, alpha=0.9)
            ax.set_facecolor('#0e1117'); ax.axis('off')
            
        elif lens_idx == 2: # Phase Flow
            ax = fig.add_subplot(111)
            U, V = np.gradient(Z)
            ax.streamplot(X, Y, V, -U, color='#00ff88', linewidth=0.8, density=1.5)
            ax.set_facecolor('#0e1117'); ax.axis('off')

        elif lens_idx == 3: # Spectral Map
            ax = fig.add_subplot(111)
            F = fftshift(fft2(Z))
            spec = np.log(np.abs(F) + 1)
            ax.imshow(spec, cmap='hot', interpolation='bilinear'); ax.axis('off')

        elif lens_idx == 4: # Stigmergy (High Diffusion)
            ax = fig.add_subplot(111)
            grid = plt.get_cmap('twilight')((Z - Z.min()) / (Z.max() - Z.min() + 1e-8))[:,:,:3]
            ax.imshow(self.apply_stigmergy_effect(grid, blur_iterations=8, bloom_intensity=1.6))
            ax.axis('off')

        elif lens_idx == 5: # Interference
            ax = fig.add_subplot(111)
            Z2 = np.sin(X*5) * np.cos(Y*5)
            inter = np.abs(Z + Z2)
            ax.imshow(inter, cmap='coolwarm', interpolation='bicubic'); ax.axis('off')
            
        elif lens_idx == 6: # Entropy Density
            ax = fig.add_subplot(111)
            E = -Z**2 * np.log(np.abs(Z)**2 + 1e-8)
            ax.imshow(E, cmap='inferno', interpolation='gaussian'); ax.axis('off')

        elif lens_idx == 7: # Metric Curvature
            ax = fig.add_subplot(111)
            curv = np.abs(np.gradient(np.gradient(Z)[0])[0])
            ax.imshow(curv, cmap='magma', interpolation='bilinear'); ax.axis('off')
            
        elif lens_idx == 8: # Vorticity
            ax = fig.add_subplot(111)
            U, V = np.gradient(Z)
            vort = np.gradient(V)[1] - np.gradient(U)[0]
            ax.imshow(vort, cmap='RdBu_r', interpolation='bicubic'); ax.axis('off')

        elif lens_idx == 9: # Quantum Jitter
            ax = fig.add_subplot(111)
            jitter = Z + np.random.randn(*Z.shape) * 0.15
            ax.imshow(jitter, cmap='twilight_shifted', interpolation='nearest'); ax.axis('off')

        elif lens_idx == 10: # Non-Euclidean Voids
            ax = fig.add_subplot(111)
            voids = np.where(Z > np.median(Z), Z, 0)
            ax.imshow(voids, cmap='gnuplot2', interpolation='bilinear'); ax.axis('off')

        elif lens_idx == 11: # Lie Slice
            ax = fig.add_subplot(111, projection='3d')
            ax.contour(X, Y, Z, levels=15, cmap='spring')
            ax.set_facecolor('#0e1117'); ax.axis('off')

        elif lens_idx == 12: # Nodal Surface
            ax = fig.add_subplot(111)
            nodes = np.abs(np.sign(Z))
            ax.imshow(nodes, cmap='bone', interpolation='nearest'); ax.axis('off')

        elif lens_idx == 13: # Fisher Metric
            ax = fig.add_subplot(111)
            F = Z**2 / (np.var(Z) + 1e-8)
            ax.imshow(F, cmap='YlOrRd', interpolation='gaussian'); ax.axis('off')

        elif lens_idx == 14: # Kinetic Elev.
            ax = fig.add_subplot(111, projection='3d')
            K = np.sqrt(np.gradient(Z)[0]**2 + np.gradient(Z)[1]**2)
            ax.plot_surface(X, Y, K, cmap='ocean', alpha=0.8); ax.axis('off')

        else: # Master Synth.
            ax = fig.add_subplot(111)
            grid = plt.get_cmap('prism')((Z - Z.min()) / (Z.max() - Z.min() + 1e-8))[:,:,:3]
            ax.imshow(self.apply_stigmergy_effect(grid, bloom_intensity=1.8))
            ax.axis('off')

        ax.set_title(f"V-{self.subjects.index(subject_name)+1:02d} | L-{lens_idx:02d} PROJECTION", color='#ff44aa', fontsize=10, family='monospace', pad=2)
        plt.subplots_adjust(top=0.9, bottom=0, right=1, left=0)
        return fig

    def get_simulation(self, subject_index, seed=42, lens=0):
        X, Y, Z = self.get_base_field(subject_index, seed)
        return self.apply_lens(X, Y, Z, lens, self.subjects[subject_index], seed)
