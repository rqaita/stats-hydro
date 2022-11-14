# Preamble
library(trend)
library(MEDITS)

# Defining path to file
setwd('~/Documents/GitHub/stats-hydro/activ_3')

# Loading file serie_anual.txt
serie_anual <- read.csv('SJP_fluv.csv', sep='\t', header=TRUE)
vazao <- serie_anual$fluv

## WALD AND WOLFOWITZ TEST FOR INDEPENDENCE AND STATIONARY
ww.test(vazao)

## MANN-WHITNEY TEST FOR HOMOGENEITY
M = length(vazao)
N = floor(M/2)
x = vazao[1:N]
y = vazao[(N+1):M]
wilcox.test(x, y)

## SPEARMAN TEST FOR STATIONARITY
spear(vazao)
