import numpy as np
import time
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor, as_completed

# -------------------------------------------------------------
# 1. Worker function for a single HPO trial
#    This function is executed in a separate process.
# -------------------------------------------------------------
def train_and_evaluate(C, X_train, y_train, X_val, y_val, trial_id):
    # Initialize the model with L1 penalty and C
    clf = LogisticRegression(
        C=C,
        penalty="l1",
        solver="saga",      # Supports L1 on large, high-dim problems
        max_iter=3000,
        n_jobs=1,           # Use 1 job per process to avoid oversubscription
        random_state=42
    )

    # Train the model
    clf.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    
    # Return all results needed for tracking
    return C, acc, trial_id


if __name__ == "__main__":
    
    # -----------------------------
    # 2. Big, sparse-signal problem
    # -----------------------------
    # Key: HUGE dimensionality, tiny number of informative features,
    X, y = make_classification(
        n_samples=100000,      # big-ish, still manageable
        n_features=200,        # many features -> overfitting risk
        n_informative=10,      # tiny true signal
        n_redundant=10,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        class_sep=1.2,
        flip_y=0.08,
        weights=[0.5, 0.5],
        random_state=42,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Scale features for better convergence and to make C more meaningful
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # -----------------------------
    # 3. Parallel HPO setup
    # -----------------------------
    rng = np.random.default_rng(0)
    n_trials = 32
    results = []
    
    # Generate the C values once
    C_values = [10 ** rng.uniform(-5, -1) for _ in range(n_trials)]
    
    # Use ProcessPoolExecutor for true parallelism
    MAX_WORKERS = 4 # Adjust based on your CPU core count
    
    all_futures = {}
    start = time.time()
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks to the executor
        for i, C in enumerate(C_values):
            # Submit the task, passing the data and parameter C
            future = executor.submit(
                train_and_evaluate, C, X_train, y_train, X_val, y_val, i
            )
            all_futures[future] = i
        
        # Process results as they complete
        for i, future in enumerate(as_completed(all_futures)):
            try:
                C, acc, trial_id= future.result()
                
                print(
                    f"{trial_id:02d}: C={C:.3e}, acc={acc:.4f}"
                )
                results.append((C, acc))
            except Exception as e:
                print(f"Trial failed with error: {e}")
                
    # -----------------------------
    # 4. Best C
    # -----------------------------
    if results:
        best_C, best_acc = max(results, key=lambda t: t[1])
        print("\n--- Best result ---")
        print(f"C={best_C:.3e}, acc={best_acc:.4f}")
    else:
        print("\nNo results found.")
        
    print(f"{time.time() - start}s")
