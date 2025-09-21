# Laboratory Work: Tidyverse in R

# 1. Data Loading

- Install and load the library:

install.packages("tidyverse")  
library(tidyverse)

- Load any tabular file into R (for example, data.csv).

data <- read.csv("data.csv", sep = ";") # if csv  
View(data)

- If the data is in Excel – use the readxl package.

# 2. Exploring the Data

- Check the structure of the data (str(), glimpse()).
- Identify which variables are numeric and which are text.
- Check if there are any missing values (NA).

# 3. Selecting and Filtering Data

- Select 1–2 columns and create a new dataframe.
- Select all columns whose names contain a certain text fragment (e.g., "id" or "score").
- Select rows by condition: numeric value > specified threshold, or text column contains a certain word (use str_detect()).

# 4. Data Transformations

- Add a new column that is the result of calculations based on existing ones (e.g., a normalized value).
- Create a dummy variable (0/1) that shows whether a certain condition is met.
- Add one row with fictional data.

# 5. Sorting and Subsetting

- Sort the data by a chosen numeric variable (ascending / descending).
- Select the top 10 rows by this variable.

# 6. Grouping and Aggregation

- Group the data by any categorical variable.
-For each group, calculate:
  - number of observations
  - mean
  - median
  - standard deviation of a chosen numeric variable
- Add a new variable – the coefficient of variation (sd / mean).

# 7. Exporting Results

- Save the grouped results into a CSV file.
- Use the stargazer package to generate a report in an HTML table.

# Report

The report should include:
- description of the loaded file (column names, number of rows)
- R code examples for each task
- screenshots of intermediate results
- conclusions about the capabilities of tidyverse
