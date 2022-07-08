# Preamble
library(fitdistrplus)

# Defining path to file
setwd('/Users/rq.aita/Códigos/stats-hydro/files')
# setwd('/home/ufpr/Documentos/GitHub/stats-hydro/files')

# Loading file serie_anual.txt
serie_anual <- read.csv('serie_anual.txt', sep='\t', header=FALSE)

rnormal <- rnorm(5000, 10, 2)
hist(rnormal)

library(reticulate)
# Verify what version of python is installed
repl_python()

pnorm(-1)
qnorm(0.98) * 5000 + 10000
