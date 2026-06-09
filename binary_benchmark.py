import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split


try:
    from bMPSOGOA_final import evaluate_fitness, sigmoid_transfer, run_bMPSOGOA
except ModuleNotFoundError:
    print("HATA: bMPSOGOA_final.py dosyası aynı klasörde bulunamadı!")
    
    exit()

# =====================================================================================
# STANDARD BINARY PSO (bPSO)
# =====================================================================================
def run_bpso(X_train, X_test, y_train, y_test, pop_size=10, max_iter=15):
    total_features = X_train.shape[1]
    # Rastgele 0 ve 1'lerden oluşan başlangıç pozisyonu
    X = np.random.randint(2, size=(pop_size, total_features))
    for i in range(pop_size):
        if np.sum(X[i]) == 0: X[i, np.random.randint(total_features)] = 1
            
    V = np.zeros((pop_size, total_features))
    
    pbest_X = np.copy(X)
    pbest_fitness = np.full(pop_size, float('inf'))
    gbest_X = np.zeros(total_features)
    gbest_fitness = float('inf')
    gbest_metrics = {}
    
    convergence = []
    w, c1, c2 = 0.9, 1.5, 1.5
    
    for t in range(max_iter):
        for i in range(pop_size):
            fit, rmse, mae, r2 = evaluate_fitness(X[i], X_train, X_test, y_train, y_test, alpha=0.99, beta=0.01)
            
            if fit < pbest_fitness[i]:
                pbest_fitness[i] = fit
                pbest_X[i] = np.copy(X[i])
            if fit < gbest_fitness:
                gbest_fitness = fit
                gbest_X = np.copy(X[i])
                gbest_metrics = {'rmse': rmse, 'mae': mae, 'r2': r2, 'count': np.sum(X[i])}
                
        convergence.append(gbest_metrics.get('rmse', gbest_fitness))
        w = 0.9 - t * (0.5 / max_iter) # Atalet ağırlığını düşür
        
        for i in range(pop_size):
            r1, r2 = np.random.rand(total_features), np.random.rand(total_features)
            # Sürekli Hız Güncellemesi
            V[i] = w * V[i] + c1 * r1 * (pbest_X[i] - X[i]) + c2 * r2 * (gbest_X - X[i])
            # Sigmoid Transferi ile Binary Güncelleme (bPSO kuralı)
            prob = sigmoid_transfer(V[i])
            X[i] = (np.random.rand(total_features) < prob).astype(int)
            
            if np.sum(X[i]) == 0: X[i, np.random.randint(total_features)] = 1
                
    return gbest_metrics, convergence

# =====================================================================================
# STANDARD BINARY GWO (bGWO)
# =====================================================================================
def run_bgwo(X_train, X_test, y_train, y_test, pop_size=10, max_iter=15):
    total_features = X_train.shape[1]
    X = np.random.randint(2, size=(pop_size, total_features))
    for i in range(pop_size):
        if np.sum(X[i]) == 0: X[i, np.random.randint(total_features)] = 1
            
    alpha_pos, beta_pos, delta_pos = np.zeros(total_features), np.zeros(total_features), np.zeros(total_features)
    alpha_fit, beta_fit, delta_fit = float("inf"), float("inf"), float("inf")
    alpha_metrics = {}
    
    convergence = []

    for t in range(max_iter):
        for i in range(pop_size):
            fit, rmse, mae, r2 = evaluate_fitness(X[i], X_train, X_test, y_train, y_test, alpha=0.99, beta=0.01)
            
            if fit < alpha_fit:
                delta_fit, delta_pos = beta_fit, np.copy(beta_pos)
                beta_fit, beta_pos = alpha_fit, np.copy(alpha_pos)
                alpha_fit, alpha_pos = fit, np.copy(X[i])
                alpha_metrics = {'rmse': rmse, 'mae': mae, 'r2': r2, 'count': np.sum(X[i])}
            elif fit < beta_fit:
                delta_fit, delta_pos = beta_fit, np.copy(beta_pos)
                beta_fit, beta_pos = fit, np.copy(X[i])
            elif fit < delta_fit:
                delta_fit, delta_pos = fit, np.copy(X[i])

        convergence.append(alpha_metrics.get('rmse', alpha_fit))
        a = 2 - t * (2 / max_iter)
        
        for i in range(pop_size):
            r1, r2 = np.random.rand(total_features), np.random.rand(total_features)
            A1, C1 = 2 * a * r1 - a, 2 * r2
            d_alpha = np.abs(C1 * alpha_pos - X[i])
            X1 = alpha_pos - A1 * d_alpha

            r1, r2 = np.random.rand(total_features), np.random.rand(total_features)
            A2, C2 = 2 * a * r1 - a, 2 * r2
            d_beta = np.abs(C2 * beta_pos - X[i])
            X2 = beta_pos - A2 * d_beta

            r1, r2 = np.random.rand(total_features), np.random.rand(total_features)
            A3, C3 = 2 * a * r1 - a, 2 * r2
            d_delta = np.abs(C3 * delta_pos - X[i])
            X3 = delta_pos - A3 * d_delta

            # Sürekli Uzayda Ortalama
            X_continuous = (X1 + X2 + X3) / 3
            # Sigmoid Transferi ile Binary Uzaya Geçiş (bGWO kuralı)
            prob = sigmoid_transfer(X_continuous)
            X[i] = (np.random.rand(total_features) < prob).astype(int)
            
            if np.sum(X[i]) == 0: X[i, np.random.randint(total_features)] = 1
                
    return alpha_metrics, convergence

# =====================================================================================
# YARIŞTIRMA VE GÖRSELLEŞTİRME (BENCHMARKING)
# =====================================================================================
if __name__ == "__main__":
    print("Veri Seti Yükleniyor ve Hazırlanıyor...")
    file_path = 'energydata_complete.csv'
    df = pd.read_csv(file_path)
    X = df.drop(columns=['date', 'Appliances'])
    y = df['Appliances']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pop_size = 10
    max_iter = 15
    
    print("\n--- 1. bPSO ÇALIŞTIRILIYOR ---")
    start_time = time.time()
    bpso_metrics, bpso_conv = run_bpso(X_train, X_test, y_train, y_test, pop_size, max_iter)
    bpso_time = time.time() - start_time
    print(f"bPSO Bitti. Süre: {bpso_time:.1f}s | RMSE: {bpso_metrics['rmse']:.4f}")

    print("\n--- 2. bGWO ÇALIŞTIRILIYOR ---")
    start_time = time.time()
    bgwo_metrics, bgwo_conv = run_bgwo(X_train, X_test, y_train, y_test, pop_size, max_iter)
    bgwo_time = time.time() - start_time
    print(f"bGWO Bitti. Süre: {bgwo_time:.1f}s | RMSE: {bgwo_metrics['rmse']:.4f}")

    print("\n--- 3. SENİN ALGORİTMAN: bMPSOGOA ÇALIŞTIRILIYOR ---")
    start_time = time.time()
    # Senin kodundan gelen fonksiyon çağrılır
    bmpsogoa_metrics, bmpsogoa_conv = run_bMPSOGOA(X_train, X_test, y_train, y_test, 'sigmoid', pop_size, max_iter)
    bmpsogoa_time = time.time() - start_time
    print(f"bMPSOGOA Bitti. Süre: {bmpsogoa_time:.1f}s | RMSE: {bmpsogoa_metrics['rmse']:.4f}")

    # --- GRAFİK ÇİZİMİ ---
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    plt.plot(bpso_conv, marker='^', linestyle='--', color='orange', linewidth=2, label='Binary PSO (bPSO)')
    plt.plot(bgwo_conv, marker='s', linestyle='--', color='green', linewidth=2, label='Binary GWO (bGWO)')
    plt.plot(bmpsogoa_conv, marker='o', linestyle='-', color='blue', linewidth=3, label='Proposed: bMPSOGOA')
    
    plt.title('Convergence Comparison of Binary Metaheuristic Algorithms for Feature Selection', fontsize=14, fontweight='bold')
    plt.xlabel('Iteration Session', fontsize=12)
    plt.ylabel('Validation RMSE Score', fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("binary_benchmark_comparison.png")
    plt.show()
    
    # --- SONUÇ TABLOSU ---
    results_table = [
        ['bPSO', bpso_metrics['count'], bpso_metrics['rmse'], bpso_metrics['mae'], bpso_metrics['r2']],
        ['bGWO', bgwo_metrics['count'], bgwo_metrics['rmse'], bgwo_metrics['mae'], bgwo_metrics['r2']],
        ['bMPSOGOA (Ours)', bmpsogoa_metrics['count'], bmpsogoa_metrics['rmse'], bmpsogoa_metrics['mae'], bmpsogoa_metrics['r2']]
    ]
    df_results = pd.DataFrame(results_table, columns=['Algorithm', 'Selected Features', 'RMSE', 'MAE', 'R2'])
    
    print("\n" + "="*70)
    print("FINAL FEATURE SELECTION BENCHMARK RESULTS")
    print("="*70)
    print(df_results.to_string(index=False))