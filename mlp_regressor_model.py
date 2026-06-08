"""
MLP (Multi-Layer Perceptron) Regressor 训练与评估
数据集：California Housing（GitHub - ageron/handson-ml2）
任务：GridSearchCV 超参数优化 + 测试集评估 + 损失曲线
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os, time, warnings
warnings.filterwarnings("ignore")

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ==================== 1. 加载数据 ====================
print("=" * 60)
print("MLP Regressor 训练与评估")
print("=" * 60)

data = np.load(os.path.join(OUT, "data_split.npz"))
X_train_full, X_test = data["X_train"], data["X_test"]
y_train_full_raw, y_test_raw = data["y_train"], data["y_test"]
print(f"训练集: {X_train_full.shape[0]}, 测试集: {X_test.shape[0]}, 特征维度: {X_train_full.shape[1]}")

y_scaler = StandardScaler()
y_train_full = y_scaler.fit_transform(y_train_full_raw.reshape(-1, 1)).ravel()
y_test = y_scaler.transform(y_test_raw.reshape(-1, 1)).ravel()

np.random.seed(42)
n_sample = 5000
idx = np.random.choice(len(X_train_full), n_sample, replace=False)
X_train_sample, y_train_sample = X_train_full[idx], y_train_full[idx]
print(f"GridSearch 采样: {n_sample}")

# ==================== 2. GridSearchCV ====================
print("\n" + "=" * 60)
print("2. GridSearchCV 超参数优化")
print("=" * 60)

param_grid = {
    "hidden_layer_sizes": [(64,), (64, 32), (128, 64)],
    "activation": ["relu", "tanh"],
    "alpha": [0.0001, 0.001],
    "learning_rate_init": [0.001, 0.01],
}
n_combos = 1
for v in param_grid.values():
    n_combos *= len(v)
print(f"参数组合: {n_combos} x 5 折 = {n_combos * 5} fits")

start = time.time()
mlp = MLPRegressor(max_iter=2000, early_stopping=True, validation_fraction=0.1,
                   random_state=42, n_iter_no_change=20)
grid = GridSearchCV(mlp, param_grid, cv=5, scoring="neg_mean_squared_error",
                    n_jobs=-1, verbose=1, return_train_score=True)
grid.fit(X_train_sample, y_train_sample)
grid_time = time.time() - start

best_params = grid.best_params_
if isinstance(best_params["hidden_layer_sizes"], str):
    import ast
    best_params["hidden_layer_sizes"] = ast.literal_eval(best_params["hidden_layer_sizes"])

print(f"\n最佳参数: {best_params}")
print(f"最佳 CV neg_MSE: {grid.best_score_:.4f}")
print(f"网格搜索耗时: {grid_time:.2f}s")

cv_results = pd.DataFrame(grid.cv_results_)
top5 = cv_results.nlargest(5, "mean_test_score")[
    ["param_hidden_layer_sizes", "param_activation", "param_alpha",
     "param_learning_rate_init", "mean_test_score", "std_test_score"]
]
top5["RMSE_scaled"] = np.sqrt(-top5["mean_test_score"])
print("\nTop-5 组合:")
for _, row in top5.iterrows():
    print(f"  hidden={row['param_hidden_layer_sizes']}, act={row['param_activation']}, "
          f"alpha={row['param_alpha']}, lr={row['param_learning_rate_init']}: "
          f"RMSE={row['RMSE_scaled']:.4f}")

# ==================== 3. 在完整训练集上重训练 ====================
print("\n" + "=" * 60)
print("3. 在完整训练集上重训练最佳模型")
print("=" * 60)

start = time.time()
best_mlp = MLPRegressor(**best_params, max_iter=2000, early_stopping=True,
                        validation_fraction=0.1, random_state=42,
                        n_iter_no_change=20, verbose=False)
best_mlp.fit(X_train_full, y_train_full)
train_time = time.time() - start

print(f"实际迭代次数: {best_mlp.n_iter_}")
print(f"最终训练损失: {best_mlp.loss_:.6f}")
print(f"总参数量: {sum(w.size for w in best_mlp.coefs_) + sum(b.size for b in best_mlp.intercepts_)}")

# ==================== 4. 测试集评估 ====================
print("\n" + "=" * 60)
print("4. 测试集评估")
print("=" * 60)

start = time.time()
y_pred_scaled = best_mlp.predict(X_test)
pred_time = time.time() - start

y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_test_orig = y_test_raw

mse = mean_squared_error(y_test_orig, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_orig, y_pred)
r2 = r2_score(y_test_orig, y_pred)

print(f"MSE:  {mse:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"MAE:  {mae:,.2f}")
print(f"R2:   {r2:.4f}")
print(f"训练耗时: {train_time:.2f}s, 预测耗时: {pred_time*1000:.2f}ms")

# ==================== 5. 图表 ====================
print("\n" + "=" * 60)
print("5. 生成图表")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

ax = axes[0]
ax.plot(best_mlp.loss_curve_, "b-", linewidth=1.5, label="Train Loss")
if best_mlp.validation_scores_:
    ax.plot(best_mlp.validation_scores_, "r-", linewidth=1.5, alpha=0.7, label="Val Score")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title(f"MLP Loss Curve (loss={best_mlp.loss_:.4f}, epochs={best_mlp.n_iter_})")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
ax.scatter(y_test_orig / 1000, y_pred / 1000, alpha=0.4, s=8, c="#ff7f0e", edgecolors="none")
lim_min = min(y_test_orig.min(), y_pred.min()) / 1000
lim_max = max(y_test_orig.max(), y_pred.max()) / 1000
ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=2, label="y=x")
ax.set_xlabel("实际房价 (千美元)")
ax.set_ylabel("预测房价 (千美元)")
ax.set_title(f"MLP: 预测 vs 实际  R2={r2:.4f}, RMSE={rmse/1000:.2f}k")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[2]
residuals = (y_test_orig - y_pred) / 1000
ax.hist(residuals, bins=60, color="#ff7f0e", edgecolor="white", alpha=0.85, density=True)
from scipy.stats import norm
mu, std = residuals.mean(), residuals.std()
xn = np.linspace(mu - 3.5 * std, mu + 3.5 * std, 200)
ax.plot(xn, norm.pdf(xn, mu, std), "r-", linewidth=2, label=f"N({mu:.1f}k, {std:.1f}k)")
ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("残差 (千美元)")
ax.set_ylabel("概率密度")
ax.set_title("MLP 残差分布")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "mlp_pred_vs_actual.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("已保存 mlp_pred_vs_actual.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(best_mlp.loss_curve_, "b-", linewidth=2, label="Train Loss")
if best_mlp.validation_scores_:
    ax.plot(best_mlp.validation_scores_, "r--", linewidth=2, alpha=0.8, label="Val Score")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title(f"MLP Regression Loss Curve  hidden={best_params['hidden_layer_sizes']}, act={best_params['activation']}, alpha={best_params['alpha']}")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "mlp_loss_curve.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("已保存 mlp_loss_curve.png")

# ==================== 6. 保存结果 ====================
print("\n" + "=" * 60)
print("6. 保存结果")
print("=" * 60)

np.savez(os.path.join(OUT, "mlp_results.npz"),
         model="MLP Regressor",
         best_params=str(best_params),
         cv_rmse_scaled=float(np.sqrt(-grid.best_score_)),
         mse=float(mse), rmse=float(rmse), mae=float(mae), r2=float(r2),
         train_time=float(train_time),
         pred_time_ms=float(pred_time * 1000),
         grid_time=float(grid_time),
         n_iter=int(best_mlp.n_iter_),
         final_loss=float(best_mlp.loss_))
print("已保存 mlp_results.npz")

print("\n" + "=" * 60)
print("MLP Regressor 训练完成！")
print("=" * 60)
