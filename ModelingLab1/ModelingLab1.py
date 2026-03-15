import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

def chi2_test(data, lamb, bins, alpha=0.05):
    n = len(data)
    observed, edges = np.histogram(data, bins=bins, range=(0, 2/lamb), density=False)

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
        result = "отвергается"
    else:
        result = "не отвергается"
    return chi2_stat, p_value, critical, result

def calc_stats(data):
    mean = np.mean(data)
    std = np.std(data, ddof=1) 
    return mean, std

def generate_sample(lamb, n):
    u = np.random.rand(n)
    x = (2.0 / lamb) * (1 - np.sqrt(1 - u))
    return x

def main():
    print(" Исследование датчика случайных чисел ")
    print("Закон распределения: f(x) = λ·(1 - λx/2) на [0, 2/λ]")
    
    try:
        lamb = float(input("Введите λ (по умолчанию 5.0): ") or 5.0)
        n_single = int(input("Введите объём выборки для гистограммы (по умолчанию 100000): ") or 100000)
        bins_hist = int(input("Число карманов гистограммы (по умолчанию 50): ") or 50)
        bins_test = int(input("Число интервалов для критерия χ² (по умолчанию 100): ") or 100)
    except:
        print("Ошибка ввода, будут использованы значения по умолчанию")
        lamb = 5.0
        n_single = 100000
        bins_hist = 50
        bins_test = 100

    sample = generate_sample(lamb, n_single)
    mean_val, std_val = calc_stats(sample)
    
    chi2_val, p_val, crit_val, res = chi2_test(sample, lamb, bins_test)
    
    print("\n Результаты для одной выборки ")
    print(f"Объём выборки n = {n_single}")
    print(f"Среднее выборочное: {mean_val:.4f} (теоретическое: {1/lamb:.4f})")
    print(f"СКО выборочное: {std_val:.4f} (теоретическое: {1/(lamb*np.sqrt(3)):.4f})")
    print(f"Статистика χ² = {chi2_val:.2f}, критическое значение (α=0.05, df={bins_test-1}) = {crit_val:.2f}")
    print(f"p-value = {p_val:.4f}")
    print(f"Гипотеза о соответствии {res}.")
    
    plt.figure(figsize=(10, 6))
    plt.hist(sample, bins=bins_hist, density=True, alpha=0.6, edgecolor='black',
             label='Экспериментальная плотность')
    x_plot = np.linspace(0, 2/lamb, 200)
    y_plot = lamb * (1 - lamb * x_plot / 2)
    plt.plot(x_plot, y_plot, 'r-', linewidth=2, label='Теоретическая плотность')
    plt.xlabel('x')
    plt.ylabel('Плотность')
    plt.title(f'Распределение с λ = {lamb}, n = {n_single}\nχ² = {chi2_val:.2f}, p = {p_val:.3f}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
    print("\n Исследование влияния параметров ")
    print("Для разных объёмов выборки и числа интервалов критерия:")
    n_values = [100, 1000, 10000, 100000]
    bins_values = [10, 20, 50, 100]
    
    print("\n{n:>10} {bins:>10} {chi2:>10} {p:>10} {result:>15}".format(
        n="n", bins="интервалы", chi2="χ²", p="p-value", result="решение"))
    for n in n_values:
        for b in bins_values:
            samp = generate_sample(lamb, n)
            chi2v, pv, _, res_v = chi2_test(samp, lamb, b)
            print("{:10d} {:10d} {:10.2f} {:10.4f} {:>15}".format(n, b, chi2v, pv, res_v))
            plt.figure(figsize=(10, 6))
            plt.hist(samp, bins=bins_hist, density=True, alpha=0.6, edgecolor='black',
                     label='Экспериментальная плотность')
            x_plot = np.linspace(0, 2/lamb, 200)
            y_plot = lamb * (1 - lamb * x_plot / 2)
            plt.plot(x_plot, y_plot, 'r-', linewidth=2, label='Теоретическая плотность')
            plt.xlabel('x')
            plt.ylabel('Плотность')
            plt.title(f'Распределение с λ = {lamb}, n = {n}\nχ² = {chi2v:.2f}, p = {pv:.3f}')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.show()
    
    print("\nВывод: с ростом объёма выборки p-value может меняться, но в среднем гипотеза не отвергается.")
    print("Число интервалов также влияет: при малом числе интервалов критерий может быть менее чувствителен.")

if __name__ == "__main__":
    main()