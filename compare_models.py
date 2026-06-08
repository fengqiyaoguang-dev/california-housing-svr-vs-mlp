"""
模型性能对比：SVR vs MLP Regressor
数据集：California Housing（GitHub - ageron/handson-ml2）
生成四项评估指标柱状图 + 训练/预测时间对比 + 残差对比
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import os, warnings, ast
warnings.filterwarnings("ignore")

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ==================== 1. 加载已有结果 ====================
print("=" * 60)
print("模型性能对比：SVR vs MLP Regressor")
print("=" * 60)

svr = dict(np.load(os.path.join(OUT, "svr_results.npz"), allow_pickle=True))
mlp = dict(np.load(os.path.join(OUT, "mlp_results.npz"), allow_pickle=True))

data = np.load(os.path.join(OUT, "data_split.npz"))
X_train, X_test, y_train_raw, y_test_raw = \
    data["X_train"], data["X_test"], data["y_train"], data["y_test"]

y_scaler = StandardScaler()
y_scaler.fit(y_train_raw.reshape(-1, 1))

# 重新推理生成对比散点图
def parse_params(raw):
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    return eval(str(raw)) if isinstance(raw, str) else raw

try:
    svr_params = parse_params(svr["best_params"])
    np.random.seed(42)
    idx = np.random.choice(len(X_train), 8000, replace=False)
    svr_model = SVR(**svr_params, max_iter=20000)
    svr_model.fit(X_train[idx], y_scaler.transform(y_train_raw[idx].reshape(-1, 1)).ravel())
    y_pred_svr = y_scaler.inverse_transform(svr_model.predict(X_test).reshape(-1, 1)).ravel()
except Exception as e:
    print(f"SVR 重推理跳过: {e}")
    y_pred_svr = np.zeros_like(y_test_raw)

try:
    mlp_params = parse_params(mlp["best_params"])
    mlp_model = MLPRegressor(**mlp_params, max_iter=2000, early_stopping=True,
                             validation_fraction=0.1, random_state=42, n_iter_no_change=20)
    mlp_model.fit(X_train, y_scaler.transform(y_train_raw.reshape(-1, 1)).ravel())
    y_pred_mlp = y_scaler.inverse_transform(mlp_model.predict(X_test).reshape(-1, 1)).ravel()
except Exception as e:
    print(f"MLP 重推理跳过: {e}")
    y_pred_mlp = np.zeros_like(y_test_raw)

# ==================== 2. 指标表 ====================
print("\n" + "=" * 60)
print("2. 测试集指标对比")
print("=" * 60)

svr_mse, svr_rmse, svr_mae, svr_r2 = \
    float(svr["mse"]), float(svr["rmse"]), float(svr["mae"]), float(svr["r2"])
mlp_mse, mlp_rmse, mlp_mae, mlp_r2 = \
    float(mlp["mse"]), float(mlp["rmse"]), float(mlp["mae"]), float(mlp["r2"])

rows = [
    ("MSE", svr_mse, mlp_mse),
    ("RMSE", svr_rmse, mlp_rmse),
    ("MAE", svr_mae, mlp_mae),
    ("R2", svr_r2, mlp_r2),
    ("Train (s)", float(svr["train_time"]), float(mlp["train_time"])),
    ("Pred (ms)", float(svr["pred_time_ms"]), float(mlp["pred_time_ms"])),
    ("GridSearch (s)", float(svr["grid_time"]), float(mlp["grid_time"])),
]
print(f"{'Metric':<18} {'SVR':>15} {'MLP':>15} {'Better':>8}")
print("-" * 60)
for name, s, m in rows:
    better = "SVR" if name == "R2" else ("SVR" if s < m else "MLP")
    print(f"{name:<18} {s:>15,.2f} {m:>15,.2f} {better:>8}")

print(f"\nR2 提升 (MLP vs SVR): {mlp_r2 - svr_r2:+.4f}")
print(f"RMSE 降低: {(svr_rmse - mlp_rmse) / svr_rmse * 100:.1f}%")
print(f"MAE 降低: {(svr_mae - mlp_mae) / svr_mae * 100:.1f}%")

# ==================== 3. 图表 ====================
print("\n" + "=" * 60)
print("3. 生成对比图表")
print("=" * 60)

# --- 指标柱状图 ---
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
metric_list = [
    ("MSE", svr_mse, mlp_mse, "越小越好"),
    ("RMSE", svr_rmse, mlp_rmse, "越小越好"),
    ("MAE", svr_mae, mlp_mae, "越小越好"),
    ("R2", svr_r2, mlp_r2, "越大越好"),
]
colors = ["#1f77b4", "#ff7f0e"]
for i, (name, s, m, note) in enumerate(metric_list):
    ax = axes[i]
    bars = ax.bar(["SVR", "MLP"], [s, m], color=colors, edgecolor="black", alpha=0.85)
    for bar, val in zip(bars, [s, m]):
        if name == "R2":
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", fontweight="bold", fontsize=9)
        else:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(s,m)*0.01,
                    f"{val:,.0f}", ha="center", fontweight="bold", fontsize=8, rotation=30)
    ax.set_title(f"{name} ({note})")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.suptitle("SVR vs MLP: 性能指标对比", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "metrics_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("已保存 metrics_comparison.png")

# --- 时间对比 ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

ax = axes[0]
time_data = {
    "GridSearch": [float(svr["grid_time"]), float(mlp["grid_time"])],
    "Train": [float(svr["train_time"]), float(mlp["train_time"])],
}
positions = np.arange(len(time_data))
width = 0.35
for i, (label, vals) in enumerate(time_data.items()):
    ax.bar(positions + i * width - width/2 + width/2, vals, width, label=label,
           edgecolor="black", alpha=0.85)
ax.set_xticks(positions)
ax.set_xticklabels(list(time_data.keys()))
ax.set_ylabel("耗时 (s)")
ax.set_title("GridSearch vs 训练耗时")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)

ax = axes[1]
pred_times = [float(svr["pred_time_ms"]), float(mlp["pred_time_ms"])]
bars = ax.bar(["SVR", "MLP"], pred_times, color=colors, edgecolor="black", alpha=0.85)
for bar, val in zip(bars, pred_times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(pred_times)*0.02,
            f"{val:.2f}ms", ha="center", fontweight="bold", fontsize=10)
ax.set_ylabel("耗时 (ms)")
ax.set_title("预测耗时对比")
ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.suptitle("SVR vs MLP: 时间效率对比", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(OUT, "time_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("已保存 time_comparison.png")

# --- 预测 vs 实际并排散点 ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, (y_pred, name, color, r2_val, rmse_val) in zip(
    axes,
    [(y_pred_svr, "SVR", "#1f77b4", svr_r2, svr_rmse),
     (y_pred_mlp, "MLP", "#ff7f0e", mlp_r2, mlp_rmse)]
):
    ax.scatter(y_test_raw / 1000, y_pred / 1000, alpha=0.4, s=8, c=color, edgecolors="none")
    lm = min(y_test_raw.min(), y_pred.min()) / 1000
    lm2 = max(y_test_raw.max(), y_pred.max()) / 1000
    ax.plot([lm, lm2], [lm, lm2], "r--", linewidth=2, label="y=x")
    ax.set_xlabel("实际房价 (千美元)")
    ax.set_ylabel("预测房价 (千美元)")
    ax.set_title(f"{name}: 预测 vs 实际  R2={r2_val:.4f}, RMSE={rmse_val/1000:.1f}k")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 550); ax.set_ylim(0, 550)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "pred_side_by_side.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("已保存 pred_side_by_side.png")

# --- 残差箱线图 ---
fig, ax = plt.subplots(figsize=(8, 5))
res_svr = (y_test_raw - y_pred_svr) / 1000
res_mlp = (y_test_raw - y_pred_mlp) / 1000
bp = ax.boxplot([res_svr, res_mlp], labels=["SVR", "MLP"],
                patch_artist=True, widths=0.5,
                medianprops=dict(color="red", linewidth=2))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
ax.set_ylabel("残差 = 实际 - 预测 (千美元)")
ax.set_title(f"SVR vs MLP: 残差分布  SVR mean={res_svr.mean():.1f}k std={res_svr.std():.1f}k | "
             f"MLP mean={res_mlp.mean():.1f}k std={res_mlp.std():.1f}k")
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "residuals_boxplot.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("已保存 residuals_boxplot.png")

print("\n" + "=" * 60)
print("模型对比完成！")
print("=" * 60)
