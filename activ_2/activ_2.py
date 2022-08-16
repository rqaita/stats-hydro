#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 22:42:00 2022

@author: rq.aita
"""

import pandas as pd
from datetime import date, timedelta

def read_data(file):
    # Leitura do dado
    df = pd.read_excel(file, header=None)
    i = df.iloc[0, 0].date()
    f = df.iloc[-1, 0].date()
    
    # Início e fim da série
    print("Início da série:", i)
    print("Fim da série:", f)
    return df, i, f


def daily_scale(df, i, f):
    # Criação das listas de dia, mês e ano
    dia = []
    mes = []
    ano = []
    
    while i <= f:  # enquanto há dia na série
        dia.append(i.day)
        mes.append(i.month)
        ano.append(i.year)
        i += timedelta(days=1)
        
    return pd.DataFrame({'dia':dia, 'mes':mes, 'ano':ano, 'data':df.loc[:, 1]})


def month_scale(df, i, f):
    # Criação das listas de dia, mês e ano
    mes = []
    ano = []
    media = []

    for a in range(i.year, f.year + 1):  # para todos os anos
        df_year = df[df['ano'] == a]
        
        if i.year == f.year:
            i_mes = i.month
            f_mes = f.month
        elif a == i.year:
            i_mes = i.month
            f_mes = 12
        elif a == f.year:
            i_mes = 1
            f_mes = f.month 
        else:
            i_mes = 1
            f_mes = 12

        for m in range(i_mes, f_mes + 1):  # para todos os meses do ano
            df_month = df_year[df_year['mes'] == m]
            mes.append(m)
            media.append(df_month["data"].mean())
            ano.append(a)

    return pd.DataFrame({'mes':mes, 'ano':ano, 'data':media})


def year_scale(df, i, f):
    # Criação das listas de dia, mês e ano
    ano = []
    media = []

    for a in range(i.year, f.year + 1):  # para todos os anos
        df_year = df[df['ano'] == a]
        media.append(df_year["data"].mean())
        ano.append(a)

    return pd.DataFrame({'ano':ano, 'data':media})


def multiple_scales(df, i, f):
    df_dia = daily_scale(df, i, f)
    df_mes = month_scale(df_dia, i, f)
    df_ano = year_scale(df_mes, i, f)
    
    return df_dia, df_mes, df_ano


def descriptive_stats(data):
    
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
    
    print(desc_stats)
    return desc_stats


def activ_2_stats(file):
    # Leitura dos dados
    df, i, f = read_data(file)
    
    # Múltiplas escalas
    df_dia, df_mes, df_ano = multiple_scales(df, i, f)
    
    # Anual
    data = df_ano["data"]
    print("Anual")
    desc_stats_ano = descriptive_stats(data)
    print("\n")
    
    # Diário
    data = df_dia["data"]
    print("Diário")
    desc_stats_dia = descriptive_stats(data)
    print("\n")
    
    # Mensal
    desc_stats_mes = []
    for m in range(1, 13):
        data = df_mes[df_mes["mes"] == m]["data"]
        print("Mês:", m)
        desc_stats_mes.append(descriptive_stats(data))
        print("\n")
    
    return (desc_stats_ano, desc_stats_dia, desc_stats_mes, i, f,
        df_dia, df_mes, df_ano)