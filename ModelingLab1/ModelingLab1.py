import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

def chi2_test(data, lamb, bins=50, alpha=0.05):
    n = len(data)
    observed, edges = np.histogram(data, bins=bins, density=False)

    expected_probs = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        prob = lamb * (b - a) - (lamb ** 2 / 4) * (b ** 2 - a ** 2)
        expected_probs.append(prob)
    expected = n * np.array(expected_probs)

    chi2_stat = np.sum((observed - expected) ** 2 / expected)
    df = len(observed) - 1  
    p_value = 1 - chi2.cdf(chi2_stat, df)
    critical = chi2.ppf(1 - alpha, df)

    if chi2_stat > critical:
        msg = f"Гипотеза отвергается (χ² = {chi2_stat:.2f} > {critical:.2f}, p = {p_value:.3f})"
    else:
        msg = f"Гипотеза не отвергается (χ² = {chi2_stat:.2f} ≤ {critical:.2f}, p = {p_value:.3f})"
    return msg

lamb = 5.0
n = 100000
bins_hist = 50
bins_test = 100 

u = np.random.rand(n)
x = (2.0 / lamb) * (1 - np.sqrt(1 - u))

plt.figure(figsize=(10, 6))
plt.hist(x, bins=bins_hist, density=True, alpha=0.6, edgecolor='black',
         label='Экспериментальная плотность')
plt.xlabel('x')
plt.ylabel('Плотность')
plt.title(f'Распределение f(x) = λ·(1 - λx/2) с λ = {lamb}\n{chi2_test(x, lamb, bins_test)}')
plt.legend()
plt.grid(alpha=0.3)
plt.show()