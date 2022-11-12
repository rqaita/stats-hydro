#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""@author: rq.aita

Selecionar duas bacias hidrográficas. Para cada bacia hidrográfica 
selecionar uma série pluviométrica e uma série fluviométrica utilizando 
a base de dados da Agencia Nacional de Águas (ANA). Baixar as séries 
temporais médias em escala diária, mensal e anual para cada estação.
As séries selecionadas devem ter no mínimo 30 anos de extensão sem falhas.

Desenvolver os seguintes itens:
• Testes de hipótese de estacionariedade, independência e homogeneidade
para cada série em escala anual. 
• Intervalos de confiança sobre a média de cada série (No caso das 
séries em escala mensal, gerar intervalos de confiança para cada mês). 
• Definição de regressão linear simples considerando como variável 
dependente a vazão média anual observada numa das estações selecionadas.
No total são duas regressões.
• Definir uma regressão linear múltipla (mais de duas variáveis 
independente) que tenha como variável dependente a vazão máxima anual 
numa das estações selecionadas. 
• Análise de frequência para as séries de vazão de máximos anuais para 
períodos de retorno de 10,50 e 100 anos 
(Considerar o melhor ajuste de distribuição)

Apresentar o trabalho em formato artigo com os seguintes elementos: 
Resumo, Introdução, Métodos, Resultados, Conclusões.

"""

# %% Definições

import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
# import seaborn as sns
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = ['Arial']
salvar = True


def reg_linear(x, y):
    """ Given 2 series of N values, X and Y, finds the better fit for
    a linear function of the form Y = aX + b.
    
    Parameters
    ----------
    x : np.array of len (N,)
        Independent variable.
    y : np.array of len (N,)
        Dependent variable.
    
    Returns
    -------
    a : float
        Angular coefficient.
    b : float
        Linear intercept.
    r2 : float
        Coefficient of determination.
        
    """
    # Reshaping variable to be applied in the function
    x = x.reshape((-1, 1))
    
    # Fitting the regression
    reg = linear_model.LinearRegression()
    reg.fit(x, y)
    
    # Coefficients: Y = aX + b
    a = reg.coef_[0]
    b = reg.intercept_
    
    # Coefficient of determination
    r2 = reg.score(x, y)
    
    print("R2 =", round(r2, 3))
    print("Ajuste obtido: Y =", round(a, 3), "X + ", round(b, 3))
    
    return a, b, r2


def resample_data(df, tipo, escala):  
    """Recebe a série em escala diária e a retorna em uma escala 
    diferente.

    Entrada
    =======
    df : pd.Series
        Index deve ser em datetime
    tipo : str
        Reamostragem que será feita - 'sum', 'max', 'med'
    escala : str
        Nova escala temporal - 'mes', 'ano'

    Retorno
    =======
    df_resampled : pd.Series

    """
    freq = {'mes' : '1MS', 'ano' : '1YS'}
    freq = freq[escala]

    # Série acumulada
    if tipo == 'sum':
        df_resampled = df.resample(freq).sum()
    # Série de máximos
    elif tipo == 'max':
        df_resampled = df.resample(freq).max()
    # Série média
    elif tipo == 'med':
        df_resampled = df.resample(freq).mean()

    # Retira valor caso df original possuísse falha na medição
    df_resampled = df_resampled[df_resampled.index.isin(df.index)]

    return df_resampled


# %% Leitura de dados

# Estação Tiririca - Camaçari/BA
df_tiririca = pd.read_excel(
    'org_geral.xlsx', sheet_name='CAM', usecols='A:C',
    header=0, index_col=0, names=['data', 'fluv', 'pluv'],
)
# Estação Fazendinha - São José dos Pinhais/PR
df_fazendinha = pd.read_excel(
    'org_geral.xlsx', sheet_name='SJP', usecols='A:C',
    header=0, index_col=0, names=['data', 'fluv', 'pluv'],
)


# %% Análise de frequências - série de máximos anuais

df = resample_data(df_tiririca['pluv'], tipo='max', escala='ano')
df_sort = df.sort_values()

## Ajuste das distribuições
# Funções
dist_func = [
    stats.gumbel_r,
    stats.expon,
    stats.norm,
    stats.lognorm,
    stats.pearson3,
    stats.pearson3,
    stats.genextreme,
]
# Nome da distribuição
dist_nome = [
    "Gumbel",
    "Exponencial",
    "Normal",
    "Log normal",
    "Person 3",
    "Log Person 3",
    "GEV",
]
# Parâmetros por MMV
dist_par = [func.fit(df) for func in dist_func]
# Empirical distribution - Weibull pp
emp_cdf = stats.mstats.plotting_positions(df_sort, alpha=0, beta=0)

## Gráfico - Q x Freq
fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 4))
# Empirical distribution - Weibull pp
ax.scatter(df_sort, emp_cdf, label="Empírica", marker='.')
# Fitted distributions
for nome, func, par in zip(dist_nome, dist_func, dist_par):
    ax.plot(df_sort, func.cdf(df_sort, *par), label=nome)
# General settings
ax.legend()
ax.grid()
ax.set_xlabel("Vazão (m$^3$/s)")
ax.set_ylabel("Frequência Acumulada")
if salvar:
    fig.savefig(
        'figures/QxFreq_' + 'CAM' + '.png',
        format='png', dpi=300,
    )

## Gráfico - Q x TR (log)
fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 4))
# Empirical distribution - Weibull pp
ax.scatter(1/(1-emp_cdf), df_sort, label="Empírica", marker='.')
# Fitted distributions
for nome, func, par in zip(dist_nome, dist_func, dist_par):
    ax.plot(1/(1 - func.cdf(df_sort, *par)), df_sort, label=nome)
# General settings
ax.legend()
ax.grid(axis='y')
ax.set_ylabel("Vazão (m$^3$/s)")
ax.set_xlabel("Frequência Acumulada")
ax.set_xscale('log')
ax.set_xlim([1, 100])
if salvar:
    fig.savefig(
        'figures/QxTR_' + 'CAM' + '.png',
        format='png', dpi=300,
    )

## Teste de hipótese - KS
for nome, func, par in zip(dist_nome, dist_func, dist_par):
    p = stats.kstest(
        emp_cdf, func.cdf(df_sort, *par), alternative='two-sided'
    ).pvalue
    print(nome)
    if p <= 0.05:
        print("Rejeitado, com p-valor", p)
    elif 0.05 < p <=1:
        print("Hipótese nula não rejeitada, com p-valor", p)


# %% Regressão linear simples - vazão média anual

df_pluv = resample_data(df_tiririca['pluv'], tipo='sum', escala='ano')
df_fluv = resample_data(df_tiririca['fluv'], tipo='med', escala='ano')


## Método dos mínimos quadrados

a, b, R2 = reg_linear(x=df_pluv.to_numpy(), y=df_fluv.to_numpy())
x = np.arange(df_pluv.min(), df_pluv.max())
y = a * x + b

fig, ax = plt.subplots(constrained_layout=True)
# Medições
ax.scatter(df_pluv, df_fluv, label="Medições")
# Ajuste
ax.plot(
    x, y, 
    label="Ajuste (R²=" + str(R2.round(3)) + ')', 
    color="gray", linestyle="--",
)
# General settings
ax.legend()
ax.set_xlabel("Precipitação (mm)")
ax.set_ylabel("Vazão (m$^3$/s)")
if salvar:
    fig.savefig(
        'figures/RLS_' + 'CAM' + '.png',
        format='png', dpi=300,
    )

# %% Regressão linear múltipla

# Definir uma regressão linear múltipla (mais de duas variáveis 
# independente) que tenha como variável dependente a vazão máxima
# anual numa das estações selecionadas. 



