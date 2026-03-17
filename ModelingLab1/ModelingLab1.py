import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2, kstest, norm

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
    result = "отвергается" if chi2_stat > critical else "не отвергается"
    return chi2_stat, p_value, critical, result

def ks_test(data, lamb, alpha=0.05):
    def cdf(x):
        x = np.asarray(x)
        res = np.zeros_like(x)
        mask = (x >= 0) & (x <= 2/lamb)
        res[mask] = lamb * x[mask] - (lamb**2 / 4) * x[mask]**2
        res[x > 2/lamb] = 1.0
        return res
    D, p_value = kstest(data, cdf)
    n = len(data)
    critical = 1.36 / np.sqrt(n) 
    result = "отвергается" if D > critical else "не отвергается"
    return D, p_value, critical, result

def calc_stats(data):
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    return mean, std

def generate_sample(lamb, n):
    u = np.random.rand(n)
    x = (2.0 / lamb) * (1 - np.sqrt(1 - u))
    return x

def generate_sum_sample(lamb, n, m):
    """Генерирует n сумм по m независимых величин из исходного распределения."""
    samples = generate_sample(lamb, n * m).reshape(n, m)
    return np.sum(samples, axis=1)

def main():
    print(" Исследование датчика случайных чисел ")
    print("Закон распределения: f(x) = λ·(1 - λx/2) на [0, 2/λ]")
    print("Теоретические моменты: E[X] = 2/(3λ), D[X] = 2/(9λ²), σ = √2/(3λ)")
    
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
    
    # Теоретические значения
    theor_mean = 2 / (3 * lamb)
    theor_std = np.sqrt(2) / (3 * lamb)
    
    chi2_val, p_val_chi2, crit_chi2, res_chi2 = chi2_test(sample, lamb, bins_test)
    ks_val, p_val_ks, crit_ks, res_ks = ks_test(sample, lamb)
    
    print("\n Результаты для одной выборки ")
    print(f"Объём выборки n = {n_single}")
    print(f"Среднее выборочное: {mean_val:.4f} (теоретическое: {theor_mean:.4f})")
    print(f"СКО выборочное: {std_val:.4f} (теоретическое: {theor_std:.4f})")
    print(f"Критерий Пирсона: χ² = {chi2_val:.2f}, критическое = {crit_chi2:.2f}, p-value = {p_val_chi2:.4f}, гипотеза {res_chi2}.")
    print(f"Критерий Колмогорова: D = {ks_val:.4f}, критическое = {crit_ks:.4f}, p-value = {p_val_ks:.4f}, гипотеза {res_ks}.")
    
    plt.figure(figsize=(10, 6))
    plt.hist(sample, bins=bins_hist, density=True, alpha=0.6, edgecolor='black',
             label='Экспериментальная плотность')
    x_plot = np.linspace(0, 2/lamb, 200)
    y_plot = lamb * (1 - lamb * x_plot / 2)
    plt.plot(x_plot, y_plot, 'r-', linewidth=2, label='Теоретическая плотность')
    plt.xlabel('x')
    plt.ylabel('Плотность')
    plt.title(f'λ = {lamb}, n = {n_single}\n'
              f'χ² = {chi2_val:.2f} (крит. {crit_chi2:.2f}) p = {p_val_chi2:.3f} → {res_chi2}\n'
              f'KS D = {ks_val:.4f} (крит. {crit_ks:.4f}) p = {p_val_ks:.3f} → {res_ks}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
    print("\n Исследование влияния параметров ")
    print("Для разных объёмов выборки и числа интервалов критерия:")
    n_values = [100, 1000, 10000, 100000]
    bins_values = [10, 20, 50, 100]
    
    print("\n" + "="*100)
    print("{:>6} {:>5} {:>8} {:>8} {:>8} {:>15} {:>8} {:>8} {:>15}".format(
        "n", "bins", "χ²", "χ²_crit", "p_χ²", "реш_χ²", "D", "D_crit", "реш_KS"))
    print("="*100)

    for n in n_values:
        for b in bins_values:
            samp = generate_sample(lamb, n)
            chi2v, pv_chi2, crit_chi2v, res_chi2v = chi2_test(samp, lamb, b)
            ksv, pv_ks, crit_ksv, res_ksv = ks_test(samp, lamb)
            print("{:6d} {:5d} {:8.2f} {:8.2f} {:8.4f} {:>15} {:8.4f} {:8.4f} {:>15}".format(
                n, b, chi2v, crit_chi2v, pv_chi2, res_chi2v, ksv, crit_ksv, res_ksv))
    
    print("\nВывод: с ростом объёма выборки p-value может меняться, но в среднем гипотеза не отвергается.")
    print("Число интервалов также влияет: при малом числе интервалов критерий может быть менее чувствителен.")
    
    print("\n" + "="*60)
    print("Исследование ЦПТ: сумма независимых величин")
    print("="*60)
    try:
        m = int(input("Введите число слагаемых (по умолчанию 30): ") or 30)
        n_cpt = int(input("Введите объём выборки сумм (по умолчанию 10000): ") or 10000)
    except:
        print("Ошибка ввода, будут использованы значения по умолчанию")
        m = 30
        n_cpt = 10000
    
    sums = generate_sum_sample(lamb, n_cpt, m)
    mean_sum = np.mean(sums)
    std_sum = np.std(sums, ddof=1)
    
    theor_mean_sum = m * (2 / (3 * lamb))
    theor_var_sum = m * (2 / (9 * lamb**2))
    theor_std_sum = np.sqrt(theor_var_sum)
    
    print(f"\nПараметры суммы {m} слагаемых (n = {n_cpt}):")
    print(f"Среднее выборочное: {mean_sum:.4f} (теоретическое: {theor_mean_sum:.4f})")
    print(f"СКО выборочное: {std_sum:.4f} (теоретическое: {theor_std_sum:.4f})")
    
    plt.figure(figsize=(10, 6))
    plt.hist(sums, bins=bins_hist, density=True, alpha=0.6, edgecolor='black',
             label='Гистограмма сумм')
    x_norm = np.linspace(theor_mean_sum - 4*theor_std_sum, theor_mean_sum + 4*theor_std_sum, 200)
    y_norm = norm.pdf(x_norm, theor_mean_sum, theor_std_sum)
    plt.plot(x_norm, y_norm, 'r-', linewidth=2, label='Нормальная аппроксимация (ЦПТ)')
    plt.xlabel('Сумма')
    plt.ylabel('Плотность')
    plt.title(f'ЦПТ: сумма {m} слагаемых, λ = {lamb}\n'
              f'Среднее = {mean_sum:.3f} (теор. {theor_mean_sum:.3f}), σ = {std_sum:.3f} (теор. {theor_std_sum:.3f})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()