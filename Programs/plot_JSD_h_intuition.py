#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: antoniosquicciarini
KAI - ZEN
"""
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
from fun_IM_generation import jensen_shannon_divergence
from fun_KDE_transformations import add_epsilon
import argparse
from scipy.integrate import quad, simps, trapz
from sklearn.neighbors import KernelDensity


def kde(X, h):
    kde = KernelDensity(bandwidth=h, kernel='gaussian')
    kde.fit(X.reshape(-1, 1))

    def pdf(x_values):
        return np.exp(kde.score_samples(x_values.reshape(-1, 1)))
    
    return pdf

def standardise(X):
    return (X-X.mean())/X.std()


parser = argparse.ArgumentParser(description="Your script description here")
x_discr = 400000
parser.add_argument('--x-discr', nargs='+', type=int, default=x_discr, help='x discretization points')
args = parser.parse_args()

X_1 = np.array([0.0, 0.8, 1.0])
X_2 = np.array([0.5, 1.1, 1.5])
X_3 = np.array([0.1, 0.3, 2.1])
X_means = [X_1.mean(), X_2.mean(), X_3.mean()]


X_1 = standardise(X_1)
X_2 = standardise(X_2)
X_3 = standardise(X_3)

h_list = [0.001, 0.1, 0.5, 1.0]

#h_list = [0.00001, 0.0001, 0.0005]
#h_list = [100., 1000.]


eps_pdf = 1E-12#Little value of probability over all the domain

x_min = min(min(X_1), min(X_2))-4*max(h_list)
x_max =  max(max(X_1), max(X_2))+4*max(h_list)
JDS_max = np.log(3)

fig, axes = plt.subplots(1, len(h_list), figsize=(4*len(h_list),  2.5)) 
n_discr = 10000

for ii, h in enumerate(h_list):
    print(f"{ii} -> h:{h}")
    
    pdf_1 = add_epsilon(kde(X_1, h) , eps_pdf)
    pdf_2 = add_epsilon(kde(X_2, h) , eps_pdf)
    pdf_3 = add_epsilon(kde(X_3, h) , eps_pdf)

    # pdf_1 = add_epsilon(gaussian_kde(X_1, bw_method=h) , eps_pdf)
    # pdf_2 = add_epsilon(gaussian_kde(X_2, bw_method=h), eps_pdf)
    # pdf_3 = add_epsilon(gaussian_kde(X_3, bw_method=h), eps_pdf)



    pdf_list = [pdf_1, pdf_2, pdf_3]
    support = np.array([[min(X_1)-4*h, max(X_1)+4*h], 
                        [min(X_2)-4*h, max(X_2)+4*h], 
                        [min(X_3)-4*h, max(X_3)+4*h]])

    JDS = jensen_shannon_divergence(pdf_list, support, args)

    x_values_1 = np.linspace(support[0,0], support[0,1], n_discr)
    x_values_2 = np.linspace(support[1,0], support[1,1], n_discr)
    x_values_3 = np.linspace(support[2,0], support[2,1], n_discr)

    axes[ii].plot(x_values_1, pdf_1(x_values_1), color='red', label='PDF 1')
    axes[ii].plot(x_values_2, pdf_2(x_values_2), color='blue', label='PDF 2')
    axes[ii].plot(x_values_3, pdf_3(x_values_3), color='green', label='PDF 3')

    axes[ii].grid()
    axes[ii].set_title(f"h:{h} - JDS:{JDS:.3f} ({JDS/JDS_max:.2%})", fontsize=12)
    axes[ii].set_xlim(x_min+min(X_means), x_max+max(X_means))
plt.legend()
plt.subplots_adjust(wspace=0.2, hspace=0.2, left=0.05, right=0.95)

plt.show()









