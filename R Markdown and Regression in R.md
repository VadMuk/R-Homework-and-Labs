# Laboratory Work: R Markdown and Regression in R

# 1.Data Preparation

  - Download the dataset like readwrite2.csv.
  - Import the dataset into R and inspect its structure using head() and str().
  - Briefly describe the variables (reading, writing, etc.).

# 2.Building the Regression Model

  - Fit a simple linear regression model predicting writing scores from reading scores.
  
  - Display the summary of the model and interpret:
  
    Intercept
    Regression coefficient
    R² and F-statistic

# 3.Visualization

  - Create a scatterplot of reading vs. writing scores.

  - Add both a linear regression line and a LOESS smoothing line.

  - Comment on the linearity of the relationship.

# 4.Model Assumptions

  - Test the assumption of homoscedasticity using the Breusch-Pagan test (ncvTest()).

  - Check residual normality using the Shapiro-Wilk test and Q-Q plot.

  - Plot diagnostic graphs (plot(model)) and describe the findings.

# 5. Outlier and Influence Analysis

  - Calculate Cook’s distance and identify potential influential observations.

  - Use olsrr functions (e.g., ols_plot_cooksd_chart, ols_plot_dfbetas) to visualize influential cases.

  - Create a new dataset excluding observations with standardized residuals > 2.

# 6.Improved Model

  - Re-run the regression on the cleaned dataset.

  - Compare the new model with the original in terms of:

  - Regression coefficient (beta)

  - Residual standard error

    - R² and F-statistic

    - Discuss how removing outliers influenced model performance.

# 7.Standardized Model

  - Build a standardized regression model using scale().

  - Interpret the standardized coefficient and compare it with the unstandardized one.

