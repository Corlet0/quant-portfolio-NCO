# quant-portfolio-NCO
# Nested Clustered Optimization (NCO) Quantitative Portfolio Engine

An institutional-grade portfolio allocation framework designed to overcome Markowitz's Curse (estimation error amplification and ill-conditioned covariance inversion). 

The engine combines:
- **Pre-Processing & Covariance Denoising:** Random Matrix Theory (Marčenko–Pastur clipping) and market mode detoning.
- **Unsupervised Machine Learning:** Metric space conversion ($D = \sqrt{0.5(1 - \rho)}$) and Ward's hierarchical agglomerative tree clustering.
- **Dual-Loop Allocation:**
  - **Inner Loop (Intra-Cluster):** Minimum Variance / Equal Risk Contribution solvers to eliminate local idiosyncratic risk.
  - **Outer Loop (Inter-Cluster):** Risk-adjusted optimization (Max Sortino / Max Sharpe) across synthetic meta-clusters with portfolio weight constraints.
```[cite: 1, 3, 11]
