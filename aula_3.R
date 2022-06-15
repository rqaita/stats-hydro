# Preamble
library(ggplot2)
library(Hmisc)

# Defining path to file
setwd('/Users/rq.aita/Códigos/stats-hydro/files')
# setwd('/home/ufpr/Documentos/GitHub/stats-hydro/files')

# Loading file serie_anual.txt
serie_anual <- read.csv('serie_anual.txt', sep='\t', header=FALSE)
anos  <- serie_anual$V1
vazao <- serie_anual$V2

## DESCRIPTIVE STATISTICS
describe(serie_anual)

# Mean
mean = mean(vazao)

# Standart deviation
sd = sd(vazao)

# Variance
var = var(vazao)

# Median
median = median(vazao)

# Mode
# Doesn't have a built-in function

# Range
max = max(vazao)
min = min(vazao)
range = max - min

# Quantiles
quantile = quantile(vazao)

## GRAPHICS
# Box-plot diagram
ggplot(serie_anual) +
  geom_boxplot(aes(y=V2)) +
  labs(y=expression('Vazão (m'^3*'/s)')) +
  theme_linedraw() +
  theme(panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.x = element_blank(),
        )

# Violin plot
ggplot(serie_anual) +
  geom_violin(aes(x=V1, y=V2)) +
  labs(x='', y=expression('Vazão (m'^3*'/s)')) +
  theme_linedraw() +
  theme(panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.x = element_blank(),
  )
