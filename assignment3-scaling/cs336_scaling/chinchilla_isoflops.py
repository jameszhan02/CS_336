# Source - https://stackoverflow.com/a/60689778
# Posted by drorhun, modified by community. See post 'Timeline' for change history
# Retrieved 2026-06-09, License - CC BY-SA 4.0

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

df = pd.read_json('data/isoflops_curves.json')
# print(df)

best_idx = df.groupby("compute_budget")["final_loss"].idxmin()
optimal_points = df.loc[best_idx]
optimal_points["tokens"] = (
    optimal_points["compute_budget"]
    / (6 * optimal_points["parameters"])
)
# print(optimal_points)

C = optimal_points["compute_budget"].values
N = optimal_points["parameters"].values
D = optimal_points["tokens"].values

print(C)
print(N)
print(D)

logC = np.log(C)
logN = np.log(N)
logD = np.log(D)

print(logC)
print(logN)
print(logD)

def power_law(C, a, alpha):
    return a * (C ** alpha)


# !!!!!!!!!! IMPLEMENT WITHOUT LOG SCALE !!!!!!!!!!!
# params_N, _ = curve_fit(power_law, C, N)
# a, alpha = params_N

# params_D, _ = curve_fit(power_law, C, D)
# b, beta = params_D

# print("a =", a)
# print("alpha =", alpha)
# print("b =", b)
# print("beta =", beta)

# C_test = np.array([1e23, 1e24])

# N_pred = power_law(C_test, a, alpha)
# D_pred = power_law(C_test, b, beta)

# print("N(1e23) =", N_pred[0])
# print("N(1e24) =", N_pred[1])

# print("D(1e23) =", D_pred[0])
# print("D(1e24) =", D_pred[1])

# C_sorted = np.sort(C)
# N_pred = power_law(C_sorted, a, alpha)

# plt.figure(figsize=(8,5))

# # 原始点
# plt.scatter(C, N, label="data points")

# # 拟合曲线
# plt.plot(C_sorted, N_pred, color="red", label="curve_fit")

# plt.xlabel("Compute budget C")
# plt.ylabel("Optimal parameters N*")
# plt.title("Power Law Fit: N = a C^alpha")

# plt.legend()
# plt.show()