install.packages("ggpubr")
library(ggpubr)

#reading in data
df <- read.csv("/Users/frankle/Downloads/Gains.csv")
df$enemy <-as.character(df$enemy)
print(df)

#anova interaction
anova1 <- aov(df$Gain ~ df$enemy * df$fit)
summary(anova1)

#f$Enemy:df$fitness_sharing  p-value: 0.1849 <-- so we do not reject H0 --> there is no dependancy between enemy and fit_sharing

#anova "seperate": 
anova2 <- aov(df$Gain ~ df$enemy + df$fit)
summary(anova2)
coef(anova2)
