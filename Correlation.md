# Lab 2: Correlation Analysis Using R

# 1. Load the dataset

  Use the file data_ITAproject.csv provided.
  Display the first few rows and summary statistics for all numeric variables.

# 2. Compute Covariance

  Calculate the covariance between HolisticScore and Composite.
  Interpret the result: does it indicate a positive or negative relationship?

# 3. Compute Correlations

  Calculate Pearson, Spearman, and Kendall correlations between HolisticScore and Composite.
  Compare the values and explain in which situations each method is most appropriate.

# 4. Generate a Correlation Matrix

  Compute the Pearson correlation matrix for all score-related variables (HolisticScore, Composite, Pronunciation, LexicalGrammar, RhetoricalOrganization, TopicDevelopment).
  Interpret the two highest and two lowest correlation values.

# 5. Visualize the Correlation Matrix

  Use the ggcorrplot package to create a heatmap of the correlation matrix.
  Describe how the color gradient reflects the strength and direction of correlations.

# 6. Scatterplot Matrix

  Using the psych package, generate a scatterplot matrix of the same variables.
  Comment on any visible linear or non-linear trends.

# 7. Exploratory Research Task

  Load the FlegeYeniKomshianLiu.sav dataset.
  Examine the relationship between Age of Acquisition (AOA) and English Pronunciation (PronEng):
      Check normality using histograms, Q-Q plots, and the Shapiro-Wilk test.
      Decide which correlation test is appropriate (Pearson or Spearman) and justify your choice.
      Compute the correlation and interpret its value and direction.

# 8. Confidence Interval

  Use the RVAideMemoire package to compute a 95% confidence interval for the correlation coefficient.
  Interpret the meaning of this interval.
