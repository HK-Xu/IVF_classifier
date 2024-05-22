suppressMessages(library(readr))
suppressMessages(library(readxl))
suppressMessages(library(dplyr))
# suppressMessages(library(pcaMethods))
suppressMessages(library(ggplot2))
suppressMessages(library(cluster))
# suppressMessages(library(ggpubr))
library(ade4)
library(cluster)
library(clusterSim)

windowsFonts(Times=windowsFont("Times New Roman"))
quant_data_path <- ""

##########################
# input data and pre-process
genus_data <- read_delim(quant_data_path, delim = "\t", show_col_types = FALSE) %>% 
  tibble::column_to_rownames(var = "category") %>% t() %>% as.matrix()

## Functions for JSD
dist.JSD <- function(inMatrix, pseudocount=0.000001, ...){
  KLD <- function(x,y){sum(x *log(x/y))}
  JSD <- function(x,y){sqrt(0.5 * KLD(x, (x+y)/2) + 0.5 * KLD(y, (x+y)/2))}
  matrixColSize <- length(colnames(inMatrix))  
  matrixRowSize <- length(rownames(inMatrix))  
  colnames <- colnames(inMatrix)  
  resultsMatrix <- matrix(0, matrixColSize, matrixColSize)  
  inMatrix <-  apply(inMatrix,1:2,function(x) ifelse (x==0,pseudocount,x))
  for(i in 1:matrixColSize){
    for(j in 1:matrixColSize){
      resultsMatrix[i,j]=JSD(as.vector(inMatrix[,i]), as.vector(inMatrix[,j]))
    }
  }
  colnames -> colnames(resultsMatrix) -> rownames(resultsMatrix)
  as.dist(resultsMatrix)->resultsMatrix  
  attr(resultsMatrix, "method") <- "dist"  
  return(resultsMatrix)
}

# The K value with the largest CH index is selected as the best clustering number
data.dist = dist.JSD(t(genus_data))# 

nclusters = NULL
for(k in 1:20){
  if(k==1){
    nclusters[k] = NA
  }else{
    data.cluster_temp = pam(data.dist, k=k, diss = TRUE)
    nclusters[k] = index.G1(genus_data, as.vector(data.cluster_temp$clustering), d = data.dist, centrotypes = "medoids")}} 

plot(nclusters, type="h", xlab="k clusters", ylab="CH index", main="Optimal number of clusters") 

# PAM clusters according to JSD distance (divided into K groups)
k_best <- 3

data.cluster <- pam(data.dist, k=k_best, diss = TRUE)

# ## plot 1
obs.pcoa=dudi.pco(data.dist, scannf=F, nf=15)


fit <- vegan::envfit(obs.pcoa$li, genus_data, perm = 2000)
p <-  data.frame(fit$vectors$pvals)
p$label <-  rownames(p)
p <-  p[order(p$fit.vectors.pvals, decreasing=F),]

env = data.frame(genus_data[,colnames(genus_data) %in% p$label[1:20]])

selected_genus <- c() # Genus to be displayed in the plot

env = data.frame(genus_data[,colnames(genus_data) %in% selected_genus])
fit = vegan::envfit(obs.pcoa$li, env, perm = 2000)
data.frame(fit$vectors$pvals)
fit$vectors$arrows
head(fit$vectors)$arrows
sample = obs.pcoa$li
for(i in 1:nrow(obs.pcoa$li)){
  if(obs.pcoa$li[i,2] < 0) sample[i,2] = obs.pcoa$li[i,2] - 0.01
  else if(obs.pcoa$li[i,2] > 0) sample[i,2] = obs.pcoa$li[i,2] + 0.01 
  else if(obs.pcoa$li[i,2] == 0) sample[i,2] = obs.pcoa$li[i,2] + 0.01
}

genus = data.frame(head(fit$vectors)$arrows)

pdf("Enterotype_PCoA.pdf", width=7, height=7)
par(family = "Times")
s.class(obs.pcoa$li, fac=as.factor(as.vector(data.cluster$clustering)), grid=F, sub="Principal coordiante analysis (JSD)", csub=1, pch=21, cpoint=2, clabel=2, cstar=1, col=c("#005A32", "#3366CC", "#990000"))
text(x = genus[,1]/4, y = genus[,2]/10, labels = rownames(genus), cex=1.5, col="black", font=1)
dev.off()

