# Preamble
library(ggplot2)

# Assigning values
x <- 4
y <- 10
z <- x + y

# Vector - beginning in i = 1 (includes the last value of the range)
a <- c(1, 2, 6, 10, 20, 30, 25, 100, 38, 120)
b <- a[2:4]

# Plotting graphics
plot(a, col='red', type='b')

# Defining path to file
setwd('/Users/rq.aita/Códigos/stats-hydro/files')

# Loading file serie_anual.txt
serie_anual <- read.csv('serie_anual.txt', sep='\t', header=FALSE)

# Extracting values from data frame
vazao <- serie_anual$V2
plot(vazao, type='l')

# Plotting time series
serie_temporal <- ts(vazao, start=1938, end=1999)
plot(serie_temporal, type='l')

anos <- serie_anual$V1
plot(anos, vazao, type='l', col='blue')

# TODO
# Plot the following graphics: bar, line, flow duration curve and histogram

# Line
ggplot(serie_anual) + 
  geom_line(aes(x=V1, y=V2)) + 
  labs(x='Ano', y=expression('Vazão (m'^3*'/s)')) +
  theme_linedraw()

# Bar
ggplot(serie_anual) + 
  geom_col(aes(x=V1, y=V2)) + 
  labs(x='Ano', y=expression('Vazão (m'^3*'/s)')) +
  theme_linedraw()

# Histogram
N  <- dim(serie_anual)[1]
NC <- round(1 + 3.3 * log(N))  # Sturges
ggplot(serie_anual) + 
  geom_histogram(aes(V2), bins=NC) + 
  labs(x=expression('Vazão (m'^3*'/s)'), y='Frequência Absoluta') +
  theme_linedraw()

# Frequency polygon
ggplot(serie_anual) +
  geom_freqpoly(aes(V2), bins=NC) + 
  labs(x=expression('Vazão (m'^3*'/s)'), y='Frequência Absoluta') +
  theme_linedraw()

# Flow duration curve
vazao_decr <- sort(vazao, decreasing=TRUE)
N <- length(vazao_decr)
n <- c(1:N)
n_N <- n / N
ggplot(serie_anual) +
  geom_line(aes(x=n_N*100, y=vazao_decr)) + 
  labs(x='Permanência (%)', y=expression('Vazão (m'^3*'/s)')) +
  theme_linedraw()