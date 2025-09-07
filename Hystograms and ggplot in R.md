
# Laboratory Work: Data Visualization in R with ggplot2

## Histograms

1. Load the built-in dataset `mtcars`.  
   - Create a histogram of the variable `mpg` with different binwidths.  

   ```r
   hist(mtcars$mpg, breaks=10, col="lightblue")

   ggplot(mtcars, aes(x=mpg)) + 
     geom_histogram(binwidth=2, fill="skyblue", color="black")
   ```

---

## Scatterplots

2. Build scatterplots between `qsec` and `mpg`.  
   - Plot only points.  
   - Plot only lines.  
   - Combine both points and lines.  

   ```r
   ggplot(mtcars, aes(x=qsec, y=mpg)) + geom_point()
   ggplot(mtcars, aes(x=qsec, y=mpg)) + geom_line()
   ggplot(mtcars, aes(x=qsec, y=mpg)) + geom_point() + geom_line()
   ```

---

## Boxplots

3. Compare `mpg` across cylinder categories (`cyl`).  

   ```r
   ggplot(mtcars, aes(x=factor(cyl), y=mpg)) + 
     geom_boxplot(fill="orange")
   ```

---

## Violin Plots

4. Load the dataset `diamonds` from ggplot2.  
   - Create a violin plot comparing `price` by `color`.  
   - Apply logarithmic scaling to the y-axis.  

   ```r
   ggplot(diamonds, aes(x=color, y=price, fill=color)) +
     geom_violin(scale="width", trim=FALSE) +
     scale_y_log10()
   ```

---

## Correlation Matrix

5. Calculate and visualize the correlation matrix of `mtcars`.  

   ```r
   cor_matrix <- cor(mtcars)
   corrplot(cor_matrix, method="ellipse")
   ```

---

## Advanced Plot

6. Create a scatterplot of `carat` vs `price` from `diamonds`.  
   - Use log scales for both axes.  
   - Color points by `color`.  
   - Add transparency to reduce overplotting.  

   ```r
   ggplot(diamonds, aes(x=carat, y=price, col=color)) +
     geom_point(alpha=0.4) +
     scale_x_log10() + 
     scale_y_log10()
   ```

---

## Control Questions

- What is the difference between histograms and barplots?  
- How does a violin plot improve upon a boxplot?  
- Why is it useful to apply log scales when plotting diamond prices?  
- How can you reduce overplotting in scatterplots?  
- What information does a correlation matrix provide?  

---

## Report Requirements

Your report should include:  

1. Title page  
2. Aim of the work  
3. Short theoretical background  
4. Screenshots or saved graphs for each task  
5. Brief conclusions 
