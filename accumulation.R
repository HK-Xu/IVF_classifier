library(vegan)
library(readr)
library(dplyr)

windowsFonts(Times=windowsFont("Times New Roman"))

quant_data_path <- ""

# input data and pre-process
quant_data <- read_delim(quant_data_path, delim = "\t", show_col_types = FALSE) %>% 
  tibble::column_to_rownames(var = "category") %>% t()


sp_accum <- specaccum(quant_data, method="random")
# predict(sp_accum)

# Plot
pdf(file = "accumulate_curve.pdf", width = 6, height  = 6)
plot(sp_accum, ci.type="poly", col="blue", lwd=2, ci.lty=0, ci.col="lightblue", 
     main=paste(stringr::str_to_title(tax_level)," accumulate curve", sep=""), 
     xlab="Sample Size", ylab=paste(stringr::str_to_title(tax_level)," count", sep=""), 
     cex.lab=1.4, family="Times")
boxplot(sp_accum, col="yellow", add=TRUE, pch="+", family = "Times")
dev.off()
