#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 22:42:00 2022

@author: rq.aita
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = ['Arial']
salvar = False

def descriptive_stats(data, print_results=False):
    """Calcula as estatísticas descritivas da série fornecida.

    Entrada
    =======
    data : pd.DataFrame
    print_results : boolean

    Retorno
    =======
    desc_stats : pd.DataFrame 

    """
    N    = data.count()  # size of the series
    mean = round(data.mean(), 2)
    std  = round(data.std(), 2)
    var  = round(data.var(), 2)
    q1   = round(data.quantile(q=0.25), 2)
    q2   = round(data.quantile(q=0.5), 2)
    q3   = round(data.quantile(q=0.75), 2)
    AIQ  = round(q3 - q1, 2)
    xmax = round(data.max(), 2)
    xmin = round(data.min(), 2)
    A    = round(xmax - xmin, 2)
    cvar = round(100 * std / mean, 0)
    skew = round(data.skew(), 3)
    kurt = round(data.kurtosis(), 3)

    desc_stats = {}
    desc_stats['Tamanho']                 = [N   , '-']
    desc_stats['Média']                   = [mean, 'm3/s']
    desc_stats['Desvio padrão']           = [std , 'm3/s']
    desc_stats['Variância']               = [var , 'm6/s2']
    desc_stats['Primeiro quartil']        = [q1  , 'm3/s']
    desc_stats['Mediana']                 = [q2  , 'm3/s']
    desc_stats['Terceiro quartil']        = [q3  , 'm3/s']
    desc_stats['Amplitude inter-quartil'] = [AIQ , 'm3/s']
    desc_stats['Máximo']                  = [xmax, 'm3/s']
    desc_stats['Mínimo']                  = [xmin, 'm3/s']
    desc_stats['Amplitude']               = [A   , 'm3/s']
    desc_stats['Coeficiente de variação'] = [cvar, '%']
    desc_stats['Assimetria']              = [skew, '-']
    desc_stats['Curtose']                 = [kurt, '-']

    desc_stats = pd.DataFrame(desc_stats).transpose()
    desc_stats.columns = ['Valor', 'Unidade']
    
    if print_results:
        print(desc_stats)
    return desc_stats


def activ_2_stats(df, tipo, print_results=False):  
    """Obtém as estatísticas descritivas da série em questão.

    Entrada
    =======
    df : pd.DataFrame
    tipo : str
    print_results : boolean

    Retorno
    =======
    desc_stats_ano : pd.DataFrame 
    desc_stats_dia : pd.DataFrame
    desc_stats_mes : tuple
    df_dia : pd.DataFrame
    df_mes : pd.DataFrame
    df_ano : pd.DataFrame

    """
    ## Escala diária
    # Organização
    df_dia = df[tipo]
    # Cálculo das estatísticas
    if print_results:
        print("\n")
        print("Diário")
    desc_stats_dia = descriptive_stats(df_dia, print_results)

    ## Escala mensal
    # Organização
    if tipo == 'fluv':
        df_mes = df_dia.resample('1MS').mean()
    elif tipo == 'pluv':
        df_mes = df_dia.resample('1MS').sum()
    df_mes = df_mes[df_mes.index.isin(df_dia.index)]
    # Cálculo das estatísticas
    desc_stats_mes = []
    for m in range(1, 13):
        if print_results:
            print("\n")
            print("Mês:", m)
        desc_stats_mes.append(descriptive_stats(
            df_mes[df_mes.index.month == m], print_results,
        ))

    ## Escala anual
    # Organização
    if tipo == 'fluv':
        df_ano = df_dia.resample('1YS').mean()
    elif tipo == 'pluv':
        df_ano = df_dia.resample('1YS').sum()
    df_ano = df_ano[df_ano.index.isin(df_dia.index)]
    # Cálculo das estatísticas
    if print_results:
        print("\n")
        print("Anual")
    desc_stats_ano = descriptive_stats(df_ano, print_results)

    return (desc_stats_ano, desc_stats_dia, desc_stats_mes,
        df_dia, df_mes, df_ano)


## Leitura de dados

df_tiririca = pd.read_excel(
    'org_geral.xlsx', sheet_name='CAM', usecols='A:C',
    header=0, index_col=0, names=['data', 'fluv', 'pluv'],
)

df_fazendinha = pd.read_excel(
    'org_geral.xlsx', sheet_name='SJP', usecols='A:C',
    header=0, index_col=0, names=['data', 'fluv', 'pluv'],
)
print_results = False

tiririca = [
    activ_2_stats(df_tiririca, 'pluv', print_results=print_results),
    activ_2_stats(df_tiririca, 'fluv', print_results=print_results),
]
fazendinha = [
    activ_2_stats(df_fazendinha, 'pluv', print_results=print_results),
    activ_2_stats(df_fazendinha, 'fluv', print_results=print_results),
]


## Gráfico de dispersão (vazão vs precipitação)

estacao = ['CAM', 'SJP']
esc_num = [-3, -2, -1]
esc_tit = ['DIA', 'MES', 'ANO']

for est in estacao:
    for num, tit in zip(esc_num, esc_tit):
        if est == 'SJP':
            df_pluv = fazendinha[0][num]
            df_fluv = fazendinha[1][num]
        elif est == 'CAM':
            df_pluv = tiririca[0][num]
            df_fluv = tiririca[1][num]

        fig, ax = plt.subplots(constrained_layout=True)
        ax.scatter(df_pluv, df_fluv, marker='.')
        ax.set_xlabel('Precipitação (mm)')
        ax.set_ylabel('Vazão (m³/s)')
        if salvar:
            fig.savefig(
                'figures/DISP_' + est + '_' + tit + '.png',
                format='png', dpi=300,
            )
        fig.show()


## Histograma

estacao = ['CAM', 'SJP']
tipo = ['PLUV', 'FLUV']
esc_num = [-1]
esc_tit = ['ANO']

for est in estacao:
    for tip in tipo:
        for num, tit in zip(esc_num, esc_tit):
            if est == 'SJP' and tip == 'PLUV':
                df = fazendinha[0][num]
            elif est == 'SJP' and tip == 'FLUV':
                df = fazendinha[1][num]
            elif est == 'CAM' and tip == 'PLUV':
                df = tiririca[0][num]
            elif est == 'CAM' and tip == 'FLUV':
                df = tiririca[1][num]
            else:
                continue

            # Histogram
            N = df.shape[0]
            NC = round(1 + 3.3 * np.log10(N))  # Sturges
            counts, bins_pol = np.histogram(df, bins=NC)

            # Frequency polygon
            bins_pol = np.concatenate([
                np.array([bins_pol[0]]),
                (bins_pol[:-1] + bins_pol[1:]) / 2,
                np.array([bins_pol[-1]])
            ])
            q1, q3 = np.percentile(df, [25, 75])
            whisker_low = q1 - (q3 - q1) * 1.5
            whisker_high = q3 + (q3 - q1) * 1.5
            outliers = df[(df > whisker_high) | (df < whisker_low)]
            counts = np.concatenate([np.zeros(1), counts, np.zeros(1)])

            # Figura
            fig, ax = plt.subplots(
                2, 1, constrained_layout=True, gridspec_kw={"height_ratios":[2, 1]}
            )
            n, bins, _ = ax[0].hist(
                    x=df, bins='sturges', color='#36759f', edgecolor='#3d3d3d',
            )
            ax[0].set_ylabel("Freq. Absoluta")
            ax[0].set_xticks(bins)
            ax[0].set_xticklabels(bins.round())

            ax_right = ax[0].twinx()
            ax_right.set_ylabel("Freq. Relativa")
            ax_right.hist(
                    x=df, bins=len(n), color='#36759f', edgecolor='#3d3d3d', 
                    weights=np.ones_like(df.to_numpy())/len(df),
            )
            ax_right.plot(bins_pol, counts/len(df), '--', color='#3d3d3d')

            sns.violinplot(x=df, cut=0, ax=ax[1])
            ax[1].set_xticks(bins)
            ax[1].set_xticklabels(bins.round())
            ax[1].set_yticks([])
            if tip == 'FLUV':
                ax[1].set_xlabel('Vazão (m$^3$/s)')
            elif tip == 'PLUV':
                ax[1].set_xlabel('Precipitação (mm)')
            
            ax[1].set_ylabel("Violin Plot")
            sns.scatterplot(
                x=outliers, y=0, marker='D', edgecolor='#3d3d3d', color='#3d3d3d',
                ax=ax[1]
            )

            if salvar:
                fig.savefig(
                    'figures/HIST_' + est + '_' + tit + '_' + tip + '.png',
                    format='png', dpi=300,
                )
            fig.show()


## Box-plot-whiskers

estacao = ['CAM', 'SJP']
tipo = ['PLUV', 'FLUV']
tit = 'MES'

for est in estacao:
    for tip in tipo:
        if est == 'SJP' and tip == 'PLUV':
            df = fazendinha[0][-2]
        elif est == 'SJP' and tip == 'FLUV':
            df = fazendinha[1][-2]
        elif est == 'CAM' and tip == 'PLUV':
            df = tiririca[0][-2]
        elif est == 'CAM' and tip == 'FLUV':
            df = tiririca[1][-2]
        else:
            continue
        
        df_mes = {}
        for m in range(1, 13):
            x = df[df.index.month == m]
            df_mes[m] = x.values
        df = pd.DataFrame(df_mes)

        # Figura
        fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 4))

        sns.violinplot(data=df, cut=0, ax=ax)

        for m in range(1, 13):
            q1, q3 = np.percentile(df[m], [25, 75])
            whisker_low = q1 - (q3 - q1) * 1.5
            whisker_high = q3 + (q3 - q1) * 1.5
            outliers = df[m][(df[m] > whisker_high) | (df[m] < whisker_low)]
            sns.scatterplot(x)

        ax.plot(range(12), df.mean(), "--", color='red', alpha=0.75, label="Média")
            
        ax.set_xlabel('Mês')
        if tip == 'FLUV':
            ax.set_ylabel('Vazão (m$^3$/s)')
        elif tip == 'PLUV':
            ax.set_ylabel('Precipitação (mm)')
        ax.legend()
        if salvar:
            fig.savefig(
                'figures/BPW_' + est + '_' + tit + '_' + tip + '.png',
                format='png', dpi=300,
            )
        fig.show()


## Curva de permanência

estacao = ['CAM', 'SJP']
tip = 'FLUV'
tit = 'ANO'

for est in estacao:
    if est == 'SJP' and tip == 'PLUV':
        df = fazendinha[0][-2]
    elif est == 'SJP' and tip == 'FLUV':
        df = fazendinha[1][-2]
    elif est == 'CAM' and tip == 'PLUV':
        df = tiririca[0][-2]
    elif est == 'CAM' and tip == 'FLUV':
        df = tiririca[1][-2]
    else:
        continue

    # Flow duration curve
    vazao_decr = df.sort_values(ascending=False)
    N = vazao_decr.shape[0]
    n = np.arange(N) + 1
    n_N = n / N

    fig, ax = plt.subplots()
    ax.plot(n_N * 100, vazao_decr, "--")

    ax.fill_between(n_N * 100, vazao_decr, alpha=0.2)

    ax.set_xlabel('Permanência (%)')
    ax.set_xlim([n_N.min() * 100, 100])
    if tip == 'FLUV':
        ax.set_ylabel('Vazão (m$^3$/s)')
    elif tip == 'PLUV':
        ax.set_ylabel('Precipitação (mm)')
    ax.set_ylim([0, vazao_decr.max()])

    if salvar:
        fig.savefig(
            'figures/PERM_' + est + '_' + tit + '_' + tip + '.png',
            format='png', dpi=300,
        )
    fig.show()