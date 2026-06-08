"""
数据探索与预处理
数据集：California Housing（加州房价预测）
来源：GitHub - ageron/handson-ml2 (Aurelien Geron, Hands-On Machine Learning)
任务：探索性数据分析 + 缺失值处理 + 特征工程 + 训练/测试集划分
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings("ignore")

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ==================== 1. 加载数据 ====================
print("=" * 60)
print("1. 加载 California Housing 数据集")
print("   来源: GitHub - ageron/handson-ml2")
print("=" * 60)

DATA_PATH = os.path.join(os.path.expanduser("~"), "scikit_learn_data", "cal_housing.csv")
df = pd.read_csv(DATA_PATH)
print(f"样本数: {df.shape[0]}, 特征数: {df.shape[1] - 1} (含1个分类特征)")

TARGET = "median_house_value"
CAT_FEATURES = ["ocean_proximity"]
NUM_FEATURES = [c for c in df.columns if c not in CAT_FEATURES + [TARGET]]
print(f"数值特征 ({len(NUM_FEATURES)}): {NUM_FEATURES}")
print(f"分类特征 ({len(CAT_FEATURES)}): {CAT_FEATURES}")
print(f"目标变量: {TARGET}")

feature_names_cn = {
    "longitude": "经度",
    "latitude": "纬度",
    "housing_median_age": "房屋中位年龄",
    "total_rooms": "总房间数",
    "total_bedrooms": "总卧室数",
    "population": "人口",
    "households": "家庭数",
    "median_income": "收入中位数",
    "ocean_proximity": "近海程度",
    "median_house_value": "房价中位数($)",
}

# ==================== 2. 数据质量检查 ====================
print("\n" + "=" * 60)
print("2. 数据质量检查")
print("=" * 60)

missing = df.isnull().sum()
missing = missing[missing > 0]
print("缺失值统计:")
for col, cnt in missing.items():
    print(f"  {feature_names_cn.get(col, col)}: {cnt} 条 ({cnt/len(df)*100:.2f}%)")

print(f"\nocean_proximity 类别分布:")
print(df["ocean_proximity"].value_counts().to_string())

desc = df[NUM_FEATURES + [TARGET]].describe().T[["mean", "std", "min", "max", "50%"]]
desc.columns = ["均值", "标准差", "最小值", "最大值", "中位数"]
desc.index = [feature_names_cn.get(i, i) for i in desc.index]
print(f"\n数值特征描述性统计:\n{desc.round(2).to_string()}")

# ==================== 3. 目标变量分布 ====================
print("\n" + "=" * 60)
print("3. 绘制目标变量分布图")
print("=" * 60)
y_raw = df[TARGET].values

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
ax.hist(y_raw / 1000, bins=80, color="#2ca02c", edgecolor="white", alpha=0.85)
ax.axvline(y_raw.mean() / 1000, color="red", linestyle="--", linewidth=2,
           label=f"均值 = {y_raw.mean()/1000:.1f}k")
ax.axvline(np.median(y_raw) / 1000, color="blue", linestyle="--", linewidth=2,
           label=f"中位数 = {np.median(y_raw)/1000:.1f}k")
ax.set_xlabel("房价中位数 (千美元)")
ax.set_ylabel("频数")
ax.set_title("加州房价中位数分布直方图")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)

ax = axes[1]
bp = ax.boxplot(y_raw / 1000, vert=True, patch_artist=True, widths=0.5,
                boxprops=dict(facecolor="#1f77b4", alpha=0.7),
                medianprops=dict(color="red", linewidth=2))
ax.set_ylabel("房价中位数 (千美元)")
ax.set_title("房价中位数箱线图")
ax.set_xticklabels([""])
ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "target_distribution.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("已保存 target_distribution.png")

# ==================== 4. 地理空间分布 ====================
print("\n" + "=" * 60)
print("4. 绘制地理空间分布图")
print("=" * 60)
fig, ax = plt.subplots(figsize=(10, 7))
sc = ax.scatter(df["longitude"], df["latitude"], c=y_raw / 1000,
                cmap="RdYlGn", alpha=0.5, s=3, edgecolors="none")
ax.set_xlabel("经度")
ax.set_ylabel("纬度")
ax.set_title("加州房价地理空间分布（颜色=房价中位数，千美元）")
cbar = plt.colorbar(sc, ax=ax, shrink=0.75)
cbar.set_label("房价中位数 (千美元)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "geo_scatter.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("已保存 geo_scatter.png")

# ==================== 5. 特征与目标散点图 ====================
print("\n" + "=" * 60)
print("5. 绘制特征与目标变量散点图")
print("=" * 60)
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, col in enumerate(NUM_FEATURES):
    ax = axes[i]
    vals = df[col].dropna()
    target_vals = df.loc[vals.index, TARGET]
    ax.scatter(vals, target_vals / 1000, alpha=0.3, s=5, c="#1f77b4", edgecolors="none")
    if len(vals) > 1:
        z = np.polyfit(vals, target_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(vals.min(), vals.max(), 100)
        ax.plot(x_line, p(x_line) / 1000, "r--", linewidth=1.5, alpha=0.7)
    ax.set_xlabel(feature_names_cn.get(col, col), fontsize=9)
    ax.set_ylabel("房价中位数 (千美元)", fontsize=9)
    ax.set_title(f"{feature_names_cn.get(col, col)} vs 房价", fontsize=10)
    ax.grid(alpha=0.3)

ax = axes[8]
oc_data = [df[df["ocean_proximity"] == cat][TARGET].values / 1000
           for cat in df["ocean_proximity"].unique()]
bp = ax.boxplot(oc_data, labels=df["ocean_proximity"].unique(),
                patch_artist=True, vert=True)
for patch in bp["boxes"]:
    patch.set_facecolor("#ff7f0e")
    patch.set_alpha(0.7)
ax.set_xlabel("近海程度", fontsize=9)
ax.set_ylabel("房价中位数 (千美元)", fontsize=9)
ax.set_title("近海程度 vs 房价", fontsize=10)
ax.tick_params(axis="x", labelsize=7, rotation=30)
ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.suptitle("各特征与房价中位数的散点图（红线=线性趋势）", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(OUT, "feature_scatter.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("已保存 feature_scatter.png")

# ==================== 6. 相关性热力图 ====================
print("\n" + "=" * 60)
print("6. 绘制相关性热力图")
print("=" * 60)
df_num = df[NUM_FEATURES + [TARGET]].copy()
df_num.columns = [feature_names_cn.get(c, c) for c in df_num.columns]
corr_matrix = df_num.corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, square=True, annot=True, fmt=".3f",
            cbar_kws={"shrink": 0.7}, ax=ax, annot_kws={"fontsize": 8})
ax.tick_params(axis="both", labelsize=9)
ax.set_title("数值特征与目标变量相关性矩阵", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "correlation_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print("\n各特征与房价中位数的相关系数（按绝对值排序）:")
corr_target = corr_matrix["房价中位数($)"].drop("房价中位数($)").sort_values(ascending=False)
for feat, val in corr_target.items():
    bar = "█" * int(abs(val) * 50)
    print(f"  {feat:12s}: {val:+.4f}  {bar}")
print("已保存 correlation_heatmap.png")

# ==================== 7. 特征工程与标准化 ====================
print("\n" + "=" * 60)
print("7. 特征工程与数据标准化")
print("=" * 60)

X = df.drop(columns=[TARGET])
y = y_raw

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
preprocessor = ColumnTransformer([
    ("num", num_pipeline, NUM_FEATURES),
    ("cat", OneHotEncoder(drop="first", sparse_output=False), CAT_FEATURES),
])
X_processed = preprocessor.fit_transform(X)

cat_encoder = preprocessor.named_transformers_["cat"]
cat_names = cat_encoder.get_feature_names_out(CAT_FEATURES)
all_feature_names = list(NUM_FEATURES) + list(cat_names)
print(f"预处理后特征维度: {X_processed.shape[1]} ({len(NUM_FEATURES)} 数值 + {len(cat_names)} 独热编码)")

# ==================== 8. 训练/测试集划分 ====================
print("\n" + "=" * 60)
print("8. 训练/测试集划分 (80% / 20%)")
print("=" * 60)
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)
print(f"训练集: {X_train.shape[0]} 条  ({X_train.shape[0]/len(y)*100:.1f}%)")
print(f"测试集: {X_test.shape[0]} 条  ({X_test.shape[0]/len(y)*100:.1f}%)")
print(f"训练集目标: 均值={y_train.mean():.2f}, 标准差={y_train.std():.2f}")
print(f"测试集目标: 均值={y_test.mean():.2f}, 标准差={y_test.std():.2f}")

# ==================== 9. 保存 ====================
print("\n" + "=" * 60)
print("9. 保存预处理数据")
print("=" * 60)
np.savez(os.path.join(OUT, "data_split.npz"),
         X_train=X_train, X_test=X_test,
         y_train=y_train, y_test=y_test,
         feature_names=np.array(all_feature_names),
         num_features=np.array(NUM_FEATURES),
         cat_feature_names=np.array(list(cat_names)))
print("已保存 data_split.npz")

df.to_csv(os.path.join(OUT, "housing_raw.csv"), index=False, encoding="utf-8-sig")

print("\n" + "=" * 60)
print("数据探索与预处理完成！")
print("=" * 60)
