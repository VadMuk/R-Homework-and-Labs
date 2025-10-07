# Homework 5: Handling Missing Data in R

# Software Tools

R language (version ≥ 4.0)

Packages: mice, VIM, titanic, dplyr, ggplot2, cowplot, missForest

# Task Description

Perform the following steps in R:

# 1. Load and Examine Data

Use the built-in airquality dataset.

data <- airquality
head(data)
summary(data)

# 2. Introduce Artificial Missing Values
data[4:10, 3] <- NA
data[1:5, 4] <- NA

# 3. Analyze Missing Data

Define a function to calculate the percentage of missing values:

pMiss <- function(x) sum(is.na(x)) / length(x) * 100
apply(data, 2, pMiss)
apply(data, 1, pMiss)

# 4. Visualize Missing Patterns

Use the mice and VIM packages

# 5. Impute Missing Data

Apply the MICE algorithm with Predictive Mean Matching

# 6. Compare Distributions

Plot the observed vs. imputed data

# 7. Fit a Linear Model

Fit a model and pool results

# 8. Additional Task (Titanic Dataset)

Use the titanic_train dataset to impute missing Age values using different methods.

# (a) Simple Value Imputation
library(titanic)
library(dplyr)

# (b) MICE Imputation
titanic_numeric <- titanic_train %>% select(Survived, Pclass, SibSp, Parch, Age)

# (c) Visualization

plot_grid(h1, h2, h3, h4, nrow = 2)
