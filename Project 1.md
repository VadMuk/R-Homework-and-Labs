# Detailed Project Plan

# 1. Choosing a Topic and Dataset
Goal: select a dataset that is personally interesting to you or related to another discipline/research project.
Examples of suitable datasets:
- Economics: data on income and education, for example, gapminder www.gapminder.org/data/
- Medicine: data on the effect of physical activity on health for example, NHANES www.cdc.gov/nchs/nhanes/).
- Sports: statistics of NBA/NHL/soccer players.
- Ecology: data on air pollution and its relationship to health or temperature.
- Education: student performance and number of study hours.
  
# 2. Formulating Research Questions
The focus should be on a few meaningful questions, not on processing all variables.
Examples:
1. Is there a relationship between education level and income?
2. Does the number of workouts affect weight/health?
3. How does GDP change with an increase in life expectancy?
3. Importing and Preparing the Data
   
# 3. Use tidyverse for:
- Importing (readr or readxl).
- Cleaning (dplyr for filtering, transforming).
- Converting categorical variables (mutate, factor).
- Reshaping into tidy format (tidyr).
Example:
library(tidyverse)

data <- read_csv("dataset.csv") %>%
  clean_names() %>%
  filter(!is.na(variable1))
  
# 4. Descriptive Analysis
Build basic visualizations:
- Histograms (geom_histogram).
- Scatterplots (geom_point).
- Boxplots for categorical variables (geom_boxplot).

Calculate basic statistics: mean, median, sd, min/max.

# 5. Linear Regression
Fit one or more linear regression models.
Format:
model <- lm(income ~ education_years + age, data = data)
summary(model)
Interpret coefficients:
- How the dependent variable changes when a predictor increases by one unit.
- Test predictor significance.

# 6. Visualization of Regression Results
Scatterplot with regression line:
ggplot(data, aes(x = education_years, y = income)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE)
Visualize model residuals (augment from broom package).
Plot predicted values.

# 7. Interpretation and Conclusions
Answer the initial research questions.
Explain the practical meaning of the coefficients.
Draw conclusions: whether hypotheses are confirmed, any unexpected findings.

# 8. Final Report Structure
1. Introduction – what the data is, why it is analyzed, research questions.
2. Methods – which packages, which model.
3. Results – visualizations, regression output tables.
4. Interpretation – meaning of each result.
5. Conclusions and limitations – main findings, data limitations, possible extensions.
