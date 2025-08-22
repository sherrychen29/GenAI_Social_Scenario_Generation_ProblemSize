
import math
from math import comb, sqrt
import pandas as pd
import os
# This script performs a binomial power analysis for classification accuracy
# It calculates the power of detecting a difference in classification accuracy
# between a baseline (p0) and various scenarios (p1) using a binomial test.
# It also computes the critical number of successes needed to reject the null hypothesis
# and provides confidence intervals for the observed success rate.
def cohen_h(p1, p0):
    p1 = min(max(p1, 1e-12), 1 - 1e-12)
    p0 = min(max(p0, 1e-12), 1 - 1e-12)
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p0))

def binom_pmf(n, k, p):
    return comb(n, k) * (p**k) * ((1-p)**(n-k))

def binom_sf(n, k, p):
    return sum(binom_pmf(n, i, p) for i in range(k, n+1))

def binomial_test_pvalue(k, n, p0):
    return binom_sf(n, k, p0)

def find_critical_k(n, p0, alpha):
    for k in range(0, n+1):
        if binom_sf(n, k, p0) <= alpha:
            return k
    return n+1

def power_binomial(p1, p0, n, alpha=0.05):
    kcrit = find_critical_k(n, p0, alpha)
    if kcrit > n:
        return 0.0, kcrit
    power = binom_sf(n, kcrit, p1)
    return power, kcrit

def wilson_ci(k, n, alpha=0.05):
    z = 1.96
    phat = k / n
    denom = 1 + z*z/n
    centre = phat + z*z/(2*n)
    margin = z * math.sqrt((phat*(1-phat) + z*z/(4*n)) / n)
    lower = (centre - margin) / denom
    upper = (centre + margin) / denom
    return max(0.0, lower), min(1.0, upper)

# parameters from classification accuracy in sequence
p0 = 1/3
p1 = [0.96,0.92,0.86,0.64,0.84,0.64,0.91,0.88]
n = 300
alpha = 0.05

# Calculations
results = []
for i in range(len(p1)):

    h = f"{cohen_h(p1[i], p0):.2f}"

    k_obs = int(round(p1[i] * n))
    pval = f"{binomial_test_pvalue(k_obs, n, p0):.2e}"
    power, kcrit = power_binomial(p1[i], p0, n, alpha)
    ci_lower, ci_upper = wilson_ci(k_obs, n, alpha=alpha)
    results.append([i,n,p1[i],h,pval,alpha,power,kcrit,f"{ci_lower:.2f}",f"{ci_upper:.2f}" ])

df_results = pd.DataFrame(results, columns=['Scenario', 'n', 'p1', 'Cohen_h', 'p-value', 'Alpha', 'Power', 'Critical_k', 'CI_Lower', 'CI_Upper'])
df_results.to_csv(os.path.join(os.getcwd(),"StatsResults","binomial_analysis_results.csv"), index=False)
print(df_results)