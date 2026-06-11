# Source - https://stackoverflow.com/a/60689778
# Posted by drorhun, modified by community. See post 'Timeline' for change history
# Retrieved 2026-06-09, License - CC BY-SA 4.0

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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

logC = np.log10(C)
logN = np.log10(N)
logD = np.log10(D)

print(logC)
print(logN)
print(logD)

logC = logC.reshape(-1, 1)
model_N = LinearRegression()
model_N.fit(logC, logN)


model_D = LinearRegression()
model_D.fit(logC, logD)





# !!!!!!!!!! IMPLEMENT WITHOUT LOG SCALE !!!!!!!!!!!
# params_N, _ = curve_fit(power_law, C, N)
# a, alpha = params_N

# params_D, _ = curve_fit(power_law, C, D)
# b, beta = params_D

# print("a =", a)
# print("alpha =", alpha)
# print("b =", b)
# print("beta =", beta)

C_test = np.array([np.log10(1e23), np.log10(1e24)])
C_test = C_test.reshape(-1, 1)
N_pred = model_N.predict(C_test)
D_pred = model_D.predict(C_test)

print("N(1e23) =", 10**N_pred[0])
print("N(1e24) =", 10**N_pred[1])

print("D(1e23) =", 10**D_pred[0])
print("D(1e24) =", 10**D_pred[1])
