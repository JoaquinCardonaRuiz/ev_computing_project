install.packages("ggpubr")
library(ggpubr)
install.packages("stargazer")
library(stargazer)

#reading in data
df <- read.csv("/Users/frankle/Downloads/results_maxfit_exp.csv")
df$Enemy <-as.character(df$Enemy)
print(df)


#anova interaction
anova1 <- aov(df$max_fitness ~ df$Enemy * df$fitness_sharing)
summary(anova1)


#f$Enemy:df$fitness_sharing  p-value: 0.1849 <-- so we do not reject H0 --> there is no dependancy between enemy and fit_sharing

#anova "seperate": 
anova2 <- aov(df$max_fitness ~ df$Enemy + df$fitness_sharing)
summary(anova2)

