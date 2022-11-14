#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""@author: rq.aita

Selecionar duas bacias hidrográficas. Para cada bacia hidrográfica 
selecionar uma série pluviométrica e uma série fluviométrica utilizando 
a base de dados da Agencia Nacional de Águas (ANA). Baixar as séries 
temporais médias em escala diária, mensal e anual para cada estação.
As séries selecionadas devem ter no mínimo 30 anos de extensão sem falhas.

Desenvolver os seguintes itens:
OK Testes de hipótese de estacionariedade, independência e homogeneidade
para cada série em escala anual. 
• Intervalos de confiança sobre a média de cada série (No caso das 
séries em escala mensal, gerar intervalos de confiança para cada mês). 
OK Definição de regressão linear simples considerando como variável 
dependente a vazão média anual observada numa das estações selecionadas.
No total são duas regressões.
OK Definir uma regressão linear múltipla (mais de duas variáveis 
independente) que tenha como variável dependente a vazão máxima anual 
numa das estações selecionadas. 
OK Análise de frequência para as séries de vazão de máximos anuais para 
períodos de retorno de 10, 50 e 100 anos 
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
salvar = False


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


def IC_media(df):
    """Calcula o intervalo de confiança da média populacional, para
    o caso onde não se tem a variância populacional.

    Entrada
    =======
    df
    
    Saída
    =====
    int_inf
    int_sup
       
    """

    # Supondo população normal, conforme não rejeitado por KS, tem-se:
    # - Se a variância populacional não é conhecida:
    std_amostral = df.std()
    media_amostral = df.mean()
    N = df.size
    nu = N - 1
    t = abs(stats.t.ppf(0.025, nu))

    # Função pivô
    int_inf = media_amostral - t * std_amostral / np.sqrt(N)
    int_sup = media_amostral + t * std_amostral / np.sqrt(N)

    print(int_inf.round(2), '< média <', int_sup.round(2))

    return (int_inf, int_sup)


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

df = resample_data(df_fazendinha['fluv'], tipo='max', escala='ano')
df_sort = df.sort_values()
salvar = True

## Ajuste das distribuições
# Funções
dist_func = [
    stats.gumbel_r,
    # stats.expon,
    stats.norm,
    stats.lognorm,
    stats.pearson3,
    stats.genextreme,
]
# Nome da distribuição
dist_nome = [
    "Gumbel",
    # "Exponencial",
    "Normal",
    "Log normal",
    "Person 3",
    "GEV",
]
# Parâmetros por MMV
dist_par = [func.fit(df) for func in dist_func]
# Empirical distribution - Weibull pp
emp_cdf = stats.mstats.plotting_positions(df_sort, alpha=0, beta=0)
# Auxiliar milenar
tr = np.linspace(0.1, 101, 1000)

## Gráfico - Q x Freq
fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 4))
# Empirical distribution - Weibull pp
ax.scatter(df_sort, emp_cdf, label="Empírica", marker='.', color='purple')

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
        'figures/QxFreq_' + 'SJP' + '.png',
        format='png', dpi=300,
    )

## Gráfico - Q x TR (log)
fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 4))
# Empirical distribution - Weibull pp
ax.scatter(1/(1-emp_cdf), df_sort, label="Empírica", marker='.', color='purple')
# Fitted distributions
for nome, func, par in zip(dist_nome, dist_func, dist_par):
    ax.plot(tr, func.ppf(1-1/tr, *par), label=nome)
# General settings
ax.legend()
ax.grid(axis='y')
ax.set_ylabel("Vazão (m$^3$/s)")
ax.set_xlabel("Tempo de Retorno (anos)")
ax.set_xscale('log')
ax.set_xlim([1, 100])
ax.set_ylim([1, 30])
if salvar:
    fig.savefig(
        'figures/QxTR_' + 'SJP' + '.png',
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

df_pluv = [
    resample_data(df_fazendinha['pluv'], tipo='sum', escala='ano'),
    resample_data(df_tiririca['pluv'], tipo='sum', escala='ano'),
]
df_fluv = [
    resample_data(df_fazendinha['fluv'], tipo='med', escala='ano'),
    resample_data(df_tiririca['fluv'], tipo='med', escala='ano'),
]
salvar = True

fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(8, 4))

for i in range(2):

## Método dos mínimos quadrados
    a, b, R2 = reg_linear(x=df_pluv[i].to_numpy(), y=df_fluv[i].to_numpy())
    x = np.arange(df_pluv[i].min(), df_pluv[i].max())
    y = a * x + b

    # Medições
    ax[i].scatter(df_pluv[i], df_fluv[i], label="Medições", marker='.')
    # Ajuste
    ax[i].plot(
        x, y, 
        label="Ajuste (R²=" + str(R2.round(3)) + ')', 
        color="gray", linestyle="--",
    )
    # General settings
    ax[i].legend()
    ax[i].set_xlabel("Precipitação (mm)")
    ax[i].set_ylabel("Vazão (m$^3$/s)")

ax[0].set_title('Estação SJP')
ax[1].set_title('Estação CAM')
if salvar:
    fig.savefig(
        'figures/RLS.png',
        format='png', dpi=300,
    )


# %% Regressão linear múltipla

# Variável dependente: Q máxima CAM
dfy = resample_data(df_tiririca['fluv'], tipo='max', escala='ano')
# Variável independente 1: P acumulada CAM
dfx1 = resample_data(df_tiririca['pluv'], tipo='sum', escala='ano')
# Variável independente 2: Q máxima SJP
dfx2 = resample_data(df_fazendinha['fluv'], tipo='max', escala='ano')
# Variável independente 3: P acumulada SJP
dfx3 = resample_data(df_fazendinha['pluv'], tipo='sum', escala='ano')

# Pegando período com dados em comum
df = pd.DataFrame(dfy).rename(columns={'fluv':'CAM_Q'})
df['CAM_P'] = dfx1
df['SJP_Q'] = dfx2
df['SJP_P'] = dfx3
df = df.dropna()

# Organizando dados
x = df.iloc[:, 1:].to_numpy()
y = df['CAM_Q'].to_numpy()

# Fitting the regression
reg = linear_model.LinearRegression()
reg.fit(x, y)
coef = reg.coef_
intercept = reg.intercept_
r2 = reg.score(x, y)


# %% Testes de hipótese - feito no R
# Testes de hipótese de estacionariedade, independência e homogeneidade
# Preparação dos dados

df = resample_data(df_fazendinha['pluv'], tipo='sum', escala='ano')
df.index = df.index.year
df.to_csv('SJP_pluv.csv', sep='\t')

df = resample_data(df_fazendinha['fluv'], tipo='max', escala='ano')
df.index = df.index.year
df.to_csv('SJP_fluv.csv', sep='\t')

df = resample_data(df_tiririca['pluv'], tipo='sum', escala='ano')
df.index = df.index.year
df.to_csv('CAM_pluv.csv', sep='\t')

df = resample_data(df_tiririca['fluv'], tipo='max', escala='ano')
df.index = df.index.year
df.to_csv('CAM_fluv.csv', sep='\t')


# %% Intervalo de confiança - média populacional

# df = resample_data(df_fazendinha['fluv'], tipo='med', escala='ano')
df = df_fazendinha['fluv']
IC = {}
IC[''] = IC_media(df)
pd.DataFrame(IC).to_clipboard()

df = resample_data(df_tiririca['fluv'], tipo='med', escala='ano')
# df = df_tiririca['fluv']
IC = {}
IC[''] = IC_media(df)
pd.DataFrame(IC).to_clipboard()

df = resample_data(df_fazendinha['fluv'], tipo='med', escala='mes')
df_mes = {}
for m in range(1, 13):
    x = df[df.index.month == m]
    df_mes[m] = x.values
df = pd.DataFrame(df_mes)
IC = {}
for m in range(1, 13):
    IC[m] = IC_media(df[m])
pd.DataFrame(IC).to_clipboard()

df = resample_data(df_tiririca['fluv'], tipo='med', escala='mes')
df_mes = {}
for m in range(1, 13):
    x = df[df.index.month == m]
    df_mes[m] = x.values
df = pd.DataFrame(df_mes)
IC = {}
for m in range(1, 13):
    IC[m] = IC_media(df[m])
pd.DataFrame(IC).to_clipboard()
