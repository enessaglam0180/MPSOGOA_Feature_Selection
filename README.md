```markdown
# bMPSOGOA: Binary Modified Particle Swarm Optimization Grasshopper Optimization Algorithm for Feature Selection

This repository contains a high-performance Python implementation of the **Binary Modified Particle Swarm Optimization Grasshopper Optimization Algorithm (bMPSOGOA)**. This hybrid metaheuristic algorithm is specifically designed to solve high-dimensional **Feature Selection** problems by combining the robust global exploration of Particle Swarm Optimization (PSO) with the refined local exploitation of the Grasshopper Optimization Algorithm (GOA).

---

## Theoretical Background

### 1. Hybridization Concept (MPSOGOA)
Individually, both PSO and GOA are outstanding optimizers, but they have distinct strengths and limitations:
- **PSO** is highly effective at global exploration but can easily get trapped in local optima due to premature convergence.
- **GOA** mimics the cooperative behavior of grasshopper swarms. It adjusts attraction and repulsion dynamics, which provides superb local exploitation but can suffer from slow convergence speed.

**MPSOGOA** merges these approaches. It enhances the classical GOA formulation by integrating the **velocity/inertia update mechanism of PSO**, allowing agents to dynamically adjust their trajectories based on:
1. Social interactions among grasshoppers (attraction/repulsion forces).
2. Their personal historical best position ($p_{best}$).
3. The global swarm's best position ($g_{best}$).

### 2. Binary Feature Selection Wrapper
Feature selection is a discrete optimization problem where the goal is to choose a subset of features that minimizes classification error and reduces data dimensionality. The fitness function balances these two objectives:

$$\text{Fitness} = \alpha \times \text{Error Rate} + \beta \times \left( \frac{\text{Selected Features}}{\text{Total Features}} \right)$$

where $\alpha \in [0.9, 0.99]$ and $\beta = 1 - \alpha$.

To map the continuous coordinates of MPSOGOA into a binary search space $\{0, 1\}$, transfer functions are used:
- **Sigmoid (S-shaped) Transfer Function:** $$T(x) = \frac{1}{1 + e^{-x}}$$
- **V-shaped Transfer Function:** $$T(x) = \left| \text{erf}\left(\frac{\sqrt{\pi}}{2} x\right) \right| \quad \text{or} \quad \left| \frac{x}{\sqrt{1+x^2}} \right|$$

---

## Repository Structure

* **`bMPSOGOA_final.py` / `bMPSOGOA_final.ipynb`**: The main standalone implementation of the proposed binary hybrid algorithm with integrated dataset pipelines and evaluation loops.
* **`binary_benchmark.py`**: Comparative analysis script that evaluates **bMPSOGOA** against standard metaheuristic algorithms (such as Binary PSO, Binary GOA, and Genetic Algorithms).
* **`comparison.png`**: Plot comparing the convergence rates and fitness performance of the tested algorithms.
* **`output sigmoid.png` / `output_v_shaped.png`**: Visual results comparing the effect of S-shaped vs. V-shaped transfer functions on the feature selection process.

---

## Installation & Setup

Ensure you have a Python 3.8+ environment. You can install all necessary dependencies with:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn

```

---

## How to Run

### 1. Run the Main Optimization Model

Execute the final production script to run bMPSOGOA on the default dataset:

```bash
python bMPSOGOA_final.py

```

### 2. Run the Multi-Algorithm Comparative Benchmark

To run a comprehensive benchmark comparison comparing bMPSOGOA with standard algorithms:

```bash
python binary_benchmark.py

```

---

## Key Algorithm Hyperparameters

You can tune the core parameters inside `bMPSOGOA_final.py` or the benchmark script to match your hardware and dataset requirements:

| Parameter | Description | Recommended Range |
| --- | --- | --- |
| `N_AGENTS` | Swarm / Population size (Number of search agents) | $15 - 40$ |
| `MAX_ITER` | Maximum number of optimization cycles | $20 - 100$ |
| `cMax` | Max parameter controlling GOA's comfort zone decay | $1.0$ |
| `cMin` | Min parameter controlling GOA's comfort zone decay | $0.00001$ |
| `w_inertia` | PSO inertia weight factor | $0.4 - 0.9$ |
| `c1`, `c2` | Cognitive and Social acceleration factors | $1.5 - 2.0$ |

---

## Analysis & Output

Upon completion, the scripts will generate several logs and visualizations:

1. **Console Summary**: Accurate details on the baseline accuracy (using all features) compared to the post-optimization performance, highlighting the reduction in feature dimensionality.
2. **Convergence Charts (`comparison.png`)**: Shows how quickly bMPSOGOA converges to high-accuracy states compared to pure PSO and GOA.
3. **Transfer Function Performance Plots**: Visual verification of S-shaped vs. V-shaped binarization performance in terms of exploration stability and feature preservation.

---

## License

This project is open-source. Feel free to use and modify it for your research and application developments.

```

```