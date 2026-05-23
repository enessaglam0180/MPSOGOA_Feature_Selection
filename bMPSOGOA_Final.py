"""
=============================================================================
Binary MPSOGOA Feature Selection — Production Version
=============================================================================
Multi-Strategy Particle Swarm Gazelle Optimization Algorithm (MPSOGOA)
applied to the "Appliances Energy Prediction" dataset.

Enhancements over the baseline version:
  1. RandomForestRegressor as the wrapper model (replaces DecisionTreeRegressor)
  2. Configurable transfer function: Sigmoid (S-shaped) or V-shaped
  3. Dynamic alpha/beta weight optimisation via automated grid search
  4. Convergence curve plot + selected feature names printed at the end
  5. Extended evaluation metrics: RMSE, MAE, R²

Author : Enes Saglam
Date   : 2026-05-23
Python : >= 3.9
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import time
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================================
# 1. TRANSFER FUNCTIONS  (Continuous → Binary)
# ============================================================================

def sigmoid_transfer(x: np.ndarray) -> np.ndarray:
    """S-shaped transfer function.
    Maps continuous values to a probability in [0, 1] via the logistic sigmoid.
    """
    x = np.clip(x, -10, 10)  # Prevent exp overflow
    return 1.0 / (1.0 + np.exp(-x))


def v_shaped_transfer(x: np.ndarray) -> np.ndarray:
    """V-shaped transfer function.
    Uses the hyperbolic tangent to map continuous values to [0, 1].
    Unlike sigmoid, this function is symmetric around zero and tends to
    preserve current bit values rather than resetting them.
    Formula: T(x) = |tanh(x)|
    """
    return np.abs(np.tanh(x))


def get_transfer_function(transfer_type: str):
    """Return the appropriate transfer function based on the type string.

    Parameters
    ----------
    transfer_type : str
        Either 'sigmoid' (S-shaped) or 'v_shaped' (V-shaped).

    Returns
    -------
    callable
        The transfer function.
    """
    if transfer_type == "sigmoid":
        return sigmoid_transfer
    elif transfer_type == "v_shaped":
        return v_shaped_transfer
    else:
        raise ValueError(
            f"Unknown transfer type '{transfer_type}'. "
            "Choose 'sigmoid' or 'v_shaped'."
        )


# ============================================================================
# 2. PIECEWISE CHAOS MAPPING  (Population Initialisation)
# ============================================================================

def piecewise_chaos_initialization(pop_size: int, dim: int) -> np.ndarray:
    """Generate an initial population using Piecewise Chaos Mapping.

    This technique improves population diversity compared to uniform random
    initialisation, helping the optimiser explore the search space better.

    Parameters
    ----------
    pop_size : int
        Number of individuals in the population.
    dim : int
        Dimensionality (number of features).

    Returns
    -------
    np.ndarray of shape (pop_size, dim)
        Initial population matrix scaled to [-5, 5].
    """
    P = 0.4
    X_chaos = np.zeros((pop_size, dim))
    val = np.random.rand(dim)

    for i in range(pop_size):
        for j in range(dim):
            if 0 <= val[j] < P:
                val[j] = val[j] / P
            elif P <= val[j] < 0.5:
                val[j] = (val[j] - P) / (0.5 - P)
            elif 0.5 <= val[j] < 1 - P:
                val[j] = (1 - P - val[j]) / (0.5 - P)
            else:
                val[j] = (1 - val[j]) / P
        X_chaos[i, :] = val * 10 - 5  # Scale to [-5, 5]

    return X_chaos


# ============================================================================
# 3. FITNESS (OBJECTIVE) FUNCTION
# ============================================================================

def evaluate_fitness(
    selected_indices: np.ndarray,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    alpha: float = 0.99,
    beta: float = 0.01,
    n_estimators: int = 50,
) -> tuple:
    """Evaluate a candidate solution's fitness.

    Fitness = alpha * RMSE  +  beta * (selected_count / total_features)

    Parameters
    ----------
    selected_indices : np.ndarray
        Binary vector indicating selected (1) or excluded (0) features.
    X_train, X_test : pd.DataFrame
        Training and test feature matrices.
    y_train, y_test : pd.Series
        Training and test target vectors.
    alpha : float
        Weight for prediction error (RMSE).
    beta : float
        Weight for dimensionality reduction ratio.
    n_estimators : int
        Number of trees in the RandomForestRegressor ensemble.

    Returns
    -------
    fitness : float
        Combined fitness score.
    rmse : float
        Root Mean Squared Error on the test set.
    """
    if np.sum(selected_indices) == 0:
        return float("inf"), float("inf")

    mask = selected_indices == 1
    X_train_sel = X_train.iloc[:, mask]
    X_test_sel = X_test.iloc[:, mask]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,  # Use all CPU cores for speed
    )
    model.fit(X_train_sel, y_train)
    predictions = model.predict(X_test_sel)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    total_features = X_train.shape[1]
    selected_count = np.sum(selected_indices)

    fitness = (alpha * rmse) + (beta * (selected_count / total_features))
    return fitness, rmse


# ============================================================================
# 4. BINARY MPSOGOA ALGORITHM
# ============================================================================

def run_bMPSOGOA(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    pop_size: int = 10,
    max_iter: int = 20,
    alpha: float = 0.99,
    beta: float = 0.01,
    transfer_type: str = "sigmoid",
    n_estimators: int = 50,
    verbose: bool = False,
) -> dict:
    """Run the Binary MPSOGOA feature selection algorithm.

    Combines PSO velocity updates with GOA-inspired exploitation/exploration
    dynamics, using Piecewise Chaos Mapping for initialisation and a
    configurable transfer function for binarisation.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Training and test feature matrices.
    y_train, y_test : pd.Series
        Training and test target vectors.
    pop_size : int
        Population size.
    max_iter : int
        Maximum number of iterations.
    alpha : float
        Fitness weight for prediction error.
    beta : float
        Fitness weight for dimensionality reduction.
    transfer_type : str
        'sigmoid' for S-shaped or 'v_shaped' for V-shaped transfer function.
    n_estimators : int
        Number of trees in the RandomForestRegressor.
    verbose : bool
        If True, print per-iteration information.

    Returns
    -------
    dict with keys:
        'selected_count'     : int   — number of selected features
        'selected_mask'      : np.ndarray — binary mask of selected features
        'rmse'               : float — best RMSE achieved
        'fitness'            : float — best fitness achieved
        'convergence_curve'  : list  — fitness at each iteration
    """
    total_features = X_train.shape[1]
    transfer_fn = get_transfer_function(transfer_type)

    # Step 1 — Initialise population via Piecewise Chaos Mapping
    X = piecewise_chaos_initialization(pop_size, total_features)
    V = np.zeros((pop_size, total_features))  # PSO velocity matrix

    pbest_X = np.copy(X)
    pbest_fitness = np.full(pop_size, float("inf"))

    gbest_X = np.zeros(total_features)
    gbest_fitness = float("inf")
    gbest_rmse = float("inf")

    # PSO coefficients
    w_max, w_min = 0.9, 0.4
    c1, c2 = 1.5, 1.5

    convergence_curve = []

    for t in range(max_iter):
        w = w_max - t * ((w_max - w_min) / max_iter)  # Adaptive inertia weight

        for i in range(pop_size):
            # Step 2 — Transfer function → binary conversion
            prob = transfer_fn(X[i])

            if transfer_type == "sigmoid":
                # S-shaped: use probability threshold
                X_bin = (prob > 0.5).astype(int)
            else:
                # V-shaped: flip current bit with probability = transfer value
                current_bin = (sigmoid_transfer(X[i]) > 0.5).astype(int)
                rand = np.random.rand(total_features)
                X_bin = np.where(rand < prob, 1 - current_bin, current_bin)

            # Ensure at least one feature is selected
            if np.sum(X_bin) == 0:
                X_bin[np.random.randint(0, total_features)] = 1

            # Evaluate fitness
            fitness, rmse = evaluate_fitness(
                X_bin, X_train, X_test, y_train, y_test,
                alpha=alpha, beta=beta, n_estimators=n_estimators,
            )

            # Update personal best
            if fitness < pbest_fitness[i]:
                pbest_fitness[i] = fitness
                pbest_X[i] = np.copy(X[i])

            # Update global best
            if fitness < gbest_fitness:
                gbest_fitness = fitness
                gbest_X = np.copy(X[i])
                gbest_rmse = rmse

        # Record convergence
        convergence_curve.append(gbest_fitness)

        if verbose:
            print(
                f"  Iter {t + 1:>3}/{max_iter}  |  "
                f"Best Fitness: {gbest_fitness:.4f}  |  "
                f"Best RMSE: {gbest_rmse:.4f}"
            )

        # Step 3 — MPSOGOA position update
        for i in range(pop_size):
            r1 = np.random.rand()
            r2 = np.random.rand()
            r3 = np.random.rand()

            # PSO velocity update
            V[i] = (
                w * V[i]
                + c1 * r1 * (pbest_X[i] - X[i])
                + c2 * r2 * (gbest_X - X[i])
            )

            # GOA hybrid dynamics
            if r3 < 0.5:
                # Exploitation: grazing behaviour
                X[i] = X[i] + V[i] * np.random.normal(0, 1, total_features)
            else:
                # Exploration: Lévy-walk-like jumping
                levy_step = np.random.standard_cauchy(total_features)
                X[i] = X[i] + V[i] + (gbest_X - X[i]) * levy_step

            # Step 4 — Global perturbation (local-minimum escape)
            if np.random.rand() < 0.1:
                X[i] = X[i] + np.random.uniform(-1, 1, total_features)

    # --- Extract final binary solution ---
    best_prob = transfer_fn(gbest_X)
    if transfer_type == "sigmoid":
        best_bin = (best_prob > 0.5).astype(int)
    else:
        current_bin = (sigmoid_transfer(gbest_X) > 0.5).astype(int)
        rand = np.random.rand(total_features)
        best_bin = np.where(rand < best_prob, 1 - current_bin, current_bin)

    if np.sum(best_bin) == 0:
        best_bin[0] = 1

    return {
        "selected_count": int(np.sum(best_bin)),
        "selected_mask": best_bin,
        "rmse": gbest_rmse,
        "fitness": gbest_fitness,
        "convergence_curve": convergence_curve,
    }


# ============================================================================
# 5. EXPANDED METRICS  (RMSE, MAE, R²)
# ============================================================================

def evaluate_final_metrics(
    selected_mask: np.ndarray,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    n_estimators: int = 50,
) -> dict:
    """Train a final RandomForestRegressor on the selected features and
    compute RMSE, MAE, and R² on the test set.

    Returns
    -------
    dict with keys 'rmse', 'mae', 'r2', 'predictions'.
    """
    mask = selected_mask == 1
    X_train_sel = X_train.iloc[:, mask]
    X_test_sel = X_test.iloc[:, mask]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_sel, y_train)
    predictions = model.predict(X_test_sel)

    return {
        "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
        "predictions": predictions,
    }


# ============================================================================
# 6. VISUALISATION  (Convergence Curve + Feature Names)
# ============================================================================

def plot_convergence_curve(
    convergence_curve: list,
    title: str = "MPSOGOA Convergence Curve",
    save_path: str | None = None,
) -> None:
    """Plot the fitness convergence curve across iterations."""
    sns.set_theme(style="whitegrid", palette="deep")
    fig, ax = plt.subplots(figsize=(10, 5))

    iterations = range(1, len(convergence_curve) + 1)
    ax.plot(
        iterations,
        convergence_curve,
        marker="o",
        markersize=5,
        linewidth=2,
        color="#2563EB",
        label="Best Fitness",
    )
    ax.fill_between(
        iterations,
        convergence_curve,
        alpha=0.15,
        color="#2563EB",
    )

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Fitness Value", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()

    if save_path:
        abs_save = os.path.abspath(save_path)
        fig.savefig(abs_save, dpi=150, bbox_inches="tight")
        print(f"  [INFO] Convergence plot saved to: {abs_save}")

    plt.show()


def print_selected_features(
    selected_mask: np.ndarray,
    feature_names: list,
) -> list:
    """Print and return the real column names of the selected features."""
    selected = [
        name for name, flag in zip(feature_names, selected_mask) if flag == 1
    ]
    print(f"\n  Selected Features ({len(selected)}/{len(feature_names)}):")
    for idx, name in enumerate(selected, 1):
        print(f"    {idx:>2}. {name}")
    return selected


# ============================================================================
# 7. ALPHA / BETA WEIGHT GRID SEARCH
# ============================================================================

def weight_grid_search(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    weight_pairs: list[tuple[float, float]] | None = None,
    transfer_type: str = "sigmoid",
    pop_size: int = 10,
    max_iter: int = 20,
    n_estimators: int = 50,
) -> pd.DataFrame:
    """Run the MPSOGOA algorithm across several (alpha, beta) combinations
    and return a summary DataFrame ranking them by RMSE.

    Parameters
    ----------
    weight_pairs : list of (alpha, beta) tuples, optional
        If None, defaults to [(0.99, 0.01), (0.95, 0.05), (0.90, 0.10),
        (0.85, 0.15), (0.80, 0.20)].

    Returns
    -------
    pd.DataFrame
        Columns: alpha, beta, selected_features, rmse, mae, r2, fitness
        Sorted by RMSE ascending.
    """
    if weight_pairs is None:
        weight_pairs = [
            (0.99, 0.01),
            (0.95, 0.05),
            (0.90, 0.10),
            (0.85, 0.15),
            (0.80, 0.20),
        ]

    records = []

    print("\n" + "=" * 80)
    print("  ALPHA / BETA WEIGHT GRID SEARCH")
    print("=" * 80)
    header = (
        f"  {'Alpha':>6}  {'Beta':>6}  |  "
        f"{'Features':>8}  {'RMSE':>10}  {'MAE':>10}  {'R²':>8}  {'Fitness':>10}"
    )
    print(header)
    print("  " + "-" * 76)

    for alpha, beta in weight_pairs:
        result = run_bMPSOGOA(
            X_train, X_test, y_train, y_test,
            pop_size=pop_size,
            max_iter=max_iter,
            alpha=alpha,
            beta=beta,
            transfer_type=transfer_type,
            n_estimators=n_estimators,
            verbose=False,
        )

        metrics = evaluate_final_metrics(
            result["selected_mask"],
            X_train, X_test, y_train, y_test,
            n_estimators=n_estimators,
        )

        row = {
            "alpha": alpha,
            "beta": beta,
            "selected_features": result["selected_count"],
            "rmse": round(metrics["rmse"], 4),
            "mae": round(metrics["mae"], 4),
            "r2": round(metrics["r2"], 4),
            "fitness": round(result["fitness"], 4),
        }
        records.append(row)

        print(
            f"  {alpha:>6.2f}  {beta:>6.2f}  |  "
            f"{row['selected_features']:>8}  {row['rmse']:>10.4f}  "
            f"{row['mae']:>10.4f}  {row['r2']:>8.4f}  {row['fitness']:>10.4f}"
        )

    df_grid = pd.DataFrame(records).sort_values("rmse").reset_index(drop=True)

    best = df_grid.iloc[0]
    print("  " + "-" * 76)
    print(
        f"  ★ Best weights: alpha={best['alpha']}, beta={best['beta']}  "
        f"→  RMSE={best['rmse']:.4f}, MAE={best['mae']:.4f}, "
        f"R²={best['r2']:.4f}"
    )
    print("=" * 80)

    return df_grid


# ============================================================================
# 8. MAIN EXECUTION PIPELINE
# ============================================================================

def main() -> None:
    """End-to-end pipeline: load data → run simulations → visualise → grid search."""

    # ------------------------------------------------------------------
    # A.  DATA LOADING
    # ------------------------------------------------------------------
    file_path = "energydata_complete.csv"
    df = pd.read_csv(file_path)

    X = df.drop(columns=["date", "Appliances"])
    y = df["Appliances"]
    feature_names = list(X.columns)
    total_features = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ------------------------------------------------------------------
    # B.  CONFIGURATION
    # ------------------------------------------------------------------
    N_RUNS         = 10        # Number of independent simulation runs
    POP_SIZE       = 10        # Population size per run
    MAX_ITER       = 20        # Iterations per run
    N_ESTIMATORS   = 50        # Trees in RandomForest (speed / accuracy trade-off)
    TRANSFER_TYPE  = "sigmoid" # 'sigmoid' or 'v_shaped'
    ALPHA          = 0.99      # Fitness weight for RMSE
    BETA           = 0.01      # Fitness weight for feature ratio

    # ------------------------------------------------------------------
    # C.  MULTI-RUN SIMULATION
    # ------------------------------------------------------------------
    print(f"\n{'=' * 85}")
    print(f"  Binary MPSOGOA Feature Selection — Production Run")
    print(f"  Transfer Function : {TRANSFER_TYPE.upper()}")
    print(f"  Wrapper Model     : RandomForestRegressor (n_estimators={N_ESTIMATORS})")
    print(f"  Alpha / Beta      : {ALPHA} / {BETA}")
    print(f"  Runs: {N_RUNS}  |  Pop: {POP_SIZE}  |  Iter: {MAX_ITER}")
    print(f"{'=' * 85}\n")

    header = (
        f"  {'Run':>3}  |  {'Original':>8}  {'Selected':>8}  |  "
        f"{'RMSE':>10}  {'MAE':>10}  {'R²':>8}  |  {'Time(s)':>8}"
    )
    print(header)
    print("  " + "-" * 78)

    results = []
    best_overall = {"rmse": float("inf")}

    for run_idx in range(1, N_RUNS + 1):
        start_time = time.time()

        result = run_bMPSOGOA(
            X_train, X_test, y_train, y_test,
            pop_size=POP_SIZE,
            max_iter=MAX_ITER,
            alpha=ALPHA,
            beta=BETA,
            transfer_type=TRANSFER_TYPE,
            n_estimators=N_ESTIMATORS,
            verbose=False,
        )

        # Compute full metrics for this run's best solution
        metrics = evaluate_final_metrics(
            result["selected_mask"],
            X_train, X_test, y_train, y_test,
            n_estimators=N_ESTIMATORS,
        )

        elapsed = round(time.time() - start_time, 2)

        row = {
            "run": run_idx,
            "original": total_features,
            "selected": result["selected_count"],
            "rmse": round(metrics["rmse"], 4),
            "mae": round(metrics["mae"], 4),
            "r2": round(metrics["r2"], 4),
            "fitness": round(result["fitness"], 4),
            "time_s": elapsed,
            "convergence": result["convergence_curve"],
            "mask": result["selected_mask"],
        }
        results.append(row)

        print(
            f"  {run_idx:>3}  |  {total_features:>8}  {row['selected']:>8}  |  "
            f"{row['rmse']:>10.4f}  {row['mae']:>10.4f}  {row['r2']:>8.4f}  |  "
            f"{elapsed:>8.2f}"
        )

        if metrics["rmse"] < best_overall["rmse"]:
            best_overall = {**row, "metrics": metrics, "result": result}

    # ------------------------------------------------------------------
    # D.  SUMMARY STATISTICS
    # ------------------------------------------------------------------
    df_runs = pd.DataFrame(results)
    print("  " + "-" * 78)
    print(
        f"  AVG  |  {total_features:>8}  {df_runs['selected'].mean():>8.1f}  |  "
        f"{df_runs['rmse'].mean():>10.4f}  {df_runs['mae'].mean():>10.4f}  "
        f"{df_runs['r2'].mean():>8.4f}  |  {df_runs['time_s'].mean():>8.2f}"
    )
    print(
        f"  STD  |  {'':>8}  {df_runs['selected'].std():>8.2f}  |  "
        f"{df_runs['rmse'].std():>10.4f}  {df_runs['mae'].std():>10.4f}  "
        f"{df_runs['r2'].std():>8.4f}  |  {df_runs['time_s'].std():>8.2f}"
    )
    print(f"{'=' * 85}")

    # ------------------------------------------------------------------
    # E.  BEST RUN — DETAILED OUTPUT
    # ------------------------------------------------------------------
    print(f"\n  ★ BEST RUN: #{best_overall['run']}")
    print(f"    RMSE : {best_overall['rmse']:.4f}")
    print(f"    MAE  : {best_overall['mae']:.4f}")
    print(f"    R²   : {best_overall['r2']:.4f}")

    # Print selected feature names
    selected_names = print_selected_features(
        best_overall["mask"], feature_names
    )

    # ------------------------------------------------------------------
    # F.  CONVERGENCE CURVE PLOT  (best run)
    # ------------------------------------------------------------------
    plot_convergence_curve(
        best_overall["convergence"],
        title=f"MPSOGOA Convergence Curve (Best Run #{best_overall['run']})",
        save_path="convergence_curve.png",
    )

    # ------------------------------------------------------------------
    # G.  TRANSFER FUNCTION COMPARISON  (Sigmoid vs V-shaped)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 85}")
    print("  TRANSFER FUNCTION COMPARISON:  Sigmoid  vs  V-shaped")
    print(f"{'=' * 85}")

    tf_results = {}
    for tf_name in ["sigmoid", "v_shaped"]:
        tf_rmses = []
        tf_features = []
        for _ in range(3):  # 3 runs per transfer function for a quick comparison
            res = run_bMPSOGOA(
                X_train, X_test, y_train, y_test,
                pop_size=POP_SIZE,
                max_iter=MAX_ITER,
                alpha=ALPHA,
                beta=BETA,
                transfer_type=tf_name,
                n_estimators=N_ESTIMATORS,
            )
            m = evaluate_final_metrics(
                res["selected_mask"],
                X_train, X_test, y_train, y_test,
                n_estimators=N_ESTIMATORS,
            )
            tf_rmses.append(m["rmse"])
            tf_features.append(res["selected_count"])

        tf_results[tf_name] = {
            "avg_rmse": np.mean(tf_rmses),
            "std_rmse": np.std(tf_rmses),
            "avg_features": np.mean(tf_features),
        }

    print(
        f"  {'Type':>10}  |  {'Avg RMSE':>10}  {'Std RMSE':>10}  {'Avg Features':>12}"
    )
    print("  " + "-" * 52)
    for name, vals in tf_results.items():
        print(
            f"  {name:>10}  |  {vals['avg_rmse']:>10.4f}  "
            f"{vals['std_rmse']:>10.4f}  {vals['avg_features']:>12.1f}"
        )
    winner = min(tf_results, key=lambda k: tf_results[k]["avg_rmse"])
    print(f"\n  ★ Recommended transfer function: {winner.upper()}")
    print(f"{'=' * 85}")

    # ------------------------------------------------------------------
    # H.  ALPHA / BETA WEIGHT GRID SEARCH
    # ------------------------------------------------------------------
    df_grid = weight_grid_search(
        X_train, X_test, y_train, y_test,
        weight_pairs=None,  # Uses default set
        transfer_type=TRANSFER_TYPE,
        pop_size=POP_SIZE,
        max_iter=MAX_ITER,
        n_estimators=N_ESTIMATORS,
    )

    # ------------------------------------------------------------------
    # I.  FINAL SUMMARY
    # ------------------------------------------------------------------
    print(f"\n{'=' * 85}")
    print("  ✅  ALL TASKS COMPLETED SUCCESSFULLY")
    print(f"{'=' * 85}")
    print(f"  • Wrapper model         : RandomForestRegressor (n_estimators={N_ESTIMATORS})")
    print(f"  • Transfer functions     : Sigmoid & V-shaped compared")
    print(f"  • Weight grid search     : {len(df_grid)} combinations tested")
    print(f"  • Best run RMSE          : {best_overall['rmse']:.4f}")
    print(f"  • Best run MAE           : {best_overall['mae']:.4f}")
    print(f"  • Best run R²            : {best_overall['r2']:.4f}")
    print(f"  • Selected features      : {', '.join(selected_names)}")
    print(f"  • Convergence plot saved : convergence_curve.png")
    print(f"{'=' * 85}\n")


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()
