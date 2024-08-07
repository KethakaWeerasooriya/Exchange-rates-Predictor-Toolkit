library(cluster)
library(dplyr)
library(factoextra)
library(fpc)
library(ggplot2)
library(readxl)
library(tidyverse)
library(NbClust)
options(warn=-1)

#Set path to directory
setwd("C:/Users/hirun/Desktop/MLCW")



wine_data <- read_excel("Whitewine_v6.xlsx")
wine_data
str(wine_data)


# Preprocessing dataset
W_data <- wine_data[1:11]


# Boxplot to visualize outliers
boxplot(W_data, outline = TRUE)


# the quartiles of every variable
q1 <- apply(W_data, 2, quantile, probs = 0.25, na.rm = TRUE)
q3 <- apply(W_data, 2, quantile, probs = 0.75, na.rm = TRUE)

# interquartile range
iqr <- q3 - q1


# Out of range outliers
outliers <- apply(W_data, 1, function(x) any(x < q1 - 1.5*iqr | x > q3 + 1.5*iqr))

#'wine_data_filtered' object is created by eliminating the outliers.
wine_data_filtered <- W_data[!outliers, ]




# Min-max normalization
min_max_normalize <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

wine_data_normalized <- as.data.frame(lapply(wine_data_filtered[1:11], min_max_normalize))

# Boxplot of normalized data
boxplot(wine_data_normalized, main = "Boxplot of Min-Max Normalized Data", cex.lab = 1.2, cex.axis 
        = 1.2, cex.main = 1.8)


# Copying  the class names
class_names <- wine_data_filtered$Class
wine_data_filtered <- wine_data_filtered[1:11]
boxplot(wine_data_filtered)






# Scaling the data
scaled_wine_data <- scale(center = TRUE, wine_data_filtered) 
scaled_wine_data
summary(scaled_wine_data)
str(scaled_wine_data)
df <- scaled_wine_data
pca_W_data <- prcomp(df, center = TRUE, scale = TRUE) 
summary(pca_W_data)
# Eigenvalue extraction
scaled_eigen <- eigen(cov(df))
print(scaled_eigen)
data_pc <- as.matrix(df) %*% scaled_eigen$vectors
head(data_pc)
pc <- prcomp(x = wine_data_filtered, center = TRUE, scale. = TRUE)
head(pc$x)
summary(pc)
# Scree plot
screeplot(pca_W_data, type = "line", main = "Scree plot", cex.lab = 1.2, cex.axis = 1.2, cex.main = 1.8)
mean_dev <- scale(wine_data_filtered, center = TRUE, scale = FALSE)
head(mean_dev)

transformed_wine_data <- as.data.frame(-pc$x[,1:2]) 
head(transformed_wine_data)




#  Elbow method as1st
fviz_nbclust(transformed_wine_data, kmeans, method = 'wss')
# Average silhouette method as2nd 
fviz_nbclust(transformed_wine_data, kmeans, method = 'silhouette')
# Gap Statistic Algorithm as 
fviz_nbclust(transformed_wine_data, kmeans, method = 'gap_stat')
# NbClust method
nb <- NbClust(transformed_wine_data, distance = "euclidean", min.nc = 2, max.nc = 10, method = 
                "kmeans")
nb
# Handling and presenting several optimal options in relation to the quantity of clusters
if (length(nb$Best.nc) > 1) {
  cat("There are several optimal solutions for the number of clusters.:\n")
  print(nb$Best.nc)
} else {
  fviz_nbclust(nb)
}


#4

# Optimal number of clusters (k) based on transformed_wine_data is 2
k <- 2
kmeans_wine <- kmeans(transformed_wine_data, centers = k, nstart = 10)
kmeans_wine
fviz_cluster(kmeans_wine, data = transformed_wine_data)
wine_cluster <- data.frame(wine_data_filtered, cluster = as.factor(kmeans_wine$cluster))
head(wine_cluster)
#calculate internal evaluation metrices
BSS <- sum(kmeans_wine$betweenss)
TSS <- sum(kmeans_wine$totss)
WSS <- sum(kmeans_wine$withinss)
cat("BSS:" , BSS , "\n")
cat("WSS:" , WSS , "\n")
cat("Ratio:" , BSS/TSS , "\n")
cat("Ratio:" , BSS/WSS , "\n")
# Silhouette coefficient
sil_coeff <- silhouette(kmeans_wine$cluster, dist(transformed_wine_data))
fviz_silhouette(sil_coeff)


# Scaling the data
wine_data_scaled <- scale(wine_data_filtered)

# Plotting boxplots
par(mfrow=c(3, 4))
for (i in 1:11) {
  boxplot(wine_data_filtered[[i]], main = colnames(wine_data_filtered)[i])
}


#taskE
# Performing PCA
pca_analysis <- prcomp(wine_data_scaled, scale. = TRUE)  # scale. = TRUE can be omitted since data is already scaled
print(pca_analysis)
summary(pca_analysis)

#show the eigen Values
print(pca_analysis$sdev^2)
#show the eigen Vectors
print(pca_analysis$rotation)
#Cumulative score per principal component
cumulative_variance <- cumsum(pca_analysis$sdev^2 / sum(pca_analysis$sdev^2))*100
cumulative_variance
#Find the quantity of fundamental elements that get a cumulative score of at least 85%.
num_PCs <- which(cumulative_variance > 85)[1]
num_PCs
#Make a new, modified dataset using the selected PCs.
pca_analysis_transform <- as.data.frame(pca_analysis$x[, 1:num_PCs])
pca_analysis_transform
boxplot(pca_analysis_transform)
# Method 2:To find the elbow point, plot the PCA. 
plot(pca_analysis, type="line", main = "Screeplot")
# Find out the quantity of newly created cluster centers.
# Method 1: NbClust 
# Calculate distance using Euclidean

nb_pca <- NbClust(
  data = pca_analysis_transform,  # Verify that the pca_analysis_transform is ready.
  distance = "euclidean",
  min.nc = 2,
  max.nc = 10,
  method = "kmeans",
  index = "all"
)

nb_pca
# Handling and showing various optimal solutions based on the number of clusters
if (length(nb_pca$Best.nc) > 1) {
  cat("Multiple best solutions found for the number of clusters:\n")
  print(nb_pca$Best.nc)
} else {
  fviz_nbclust(nb_pca)
}
# Method 2: Gap statistics
gap_stat <- clusGap(pca_analysis_transform, FUN = kmeans, nstart = 2, K.max = 10, B = 100)

plot(gap_stat, main = "Gap Statistic", xlab = "Number of Clusters")
# Method 3: Elbow plot
elbow_plot <- fviz_nbclust(pca_analysis_transform, kmeans, method = "wss")
plot(elbow_plot)
# WSS plot
fviz_nbclust(pca_analysis_transform, kmeans, method = "wss", k.max = 10) + theme_minimal() + 
  ggtitle("the Elbow Method")


# Method 4: Silhouette method
sil_width <- c(NA)
fviz_nbclust(pca_analysis_transform, kmeans, method = 'silhouette')



#Identify the most preferred k (in this case, k = 2 is used as an example).
fav_k2 = 2
pca_kmeans_wine2 <- kmeans(pca_analysis_transform, centers = fav_k2, nstart = 25)
pca_kmeans_wine2

boxplot(pca_kmeans_wine2)
fviz_cluster(pca_kmeans_wine2, data = pca_analysis_transform, geom = c("point"), main = "Cluster 
plot for k = 2")
#Calculating the  metrix

wss_pca2 = pca_kmeans_wine2$tot.withinss
bss_pca2 = pca_kmeans_wine2$betweenss
tss_pca2 = pca_kmeans_wine2$totss
print(paste("Total within-cluster sum of square is", wss_pca2))
print(paste("Between Sum of Squares is", bss_pca2))
print(paste("Ratio of bss to tss", bss_pca2/tss_pca2))
print(paste("Ratio of bss to wss", bss_pca2/wss_pca2))
# Generating silhouette plot
silhouette_plot_pca2 <- silhouette(pca_kmeans_wine2$cluster, dist(pca_analysis_transform))



# Take out k-means clustering.
#k3 cluster.
#Subtask g: Take out k-means analysis utilizing the automatic techniques' most preferred k.
#Identify the most preferred k (in this case, k = 2 is used as an example).





fav_k = 3
pca_kmeans_wine <- kmeans(pca_analysis_transform, centers = fav_k, nstart = 25)
pca_kmeans_wine


#print in same box
boxplot(pca_kmeans_wine)
fviz_cluster(pca_kmeans_wine, data = pca_analysis_transform, geom = c("point"), main = "Cluster 
plot for k = 2")



#Calculating the  metrix
wss_pca = pca_kmeans_wine$tot.withinss
bss_pca = pca_kmeans_wine$betweenss
tss_pca = pca_kmeans_wine$totss
print(paste("Total within-cluster sum of square is", wss_pca))
print(paste("Between Sum of Squares is", bss_pca))
print(paste("Ratio of bss to tss", bss_pca/tss_pca))
print(paste("Ratio of bss to wss", bss_pca/wss_pca))
# Generating silhouette plot
silhouette_plot_pca <- silhouette(pca_kmeans_wine$cluster, dist(pca_analysis_transform))
# Average silhouette width score
silhouette_width_pca <- mean(silhouette_plot_pca[,3])
print(paste0("Average silhouette width: ", silhouette_width_pca))
plot(silhouette_width_pca)
silhouette_width_pca
sil <- silhouette(pca_kmeans_wine$cluster, dist(pca_analysis_transform))
fviz_silhouette(sil)
#Subtask i: Use the Calinski-Harabasz Index and provide examples
pca_ch_idx <- cluster.stats(dist(pca_analysis_transform), pca_kmeans_wine$cluster)$ch

print(paste("Calinski-Harabasz Index:", pca_ch_idx))
sil2 <- silhouette(pca_kmeans_wine2$cluster, dist(pca_analysis_transform))
fviz_silhouette(sil2)
#Subtask i: For k = 2, implement and show the Calinski-Harabasz Index.
pca_ch_idx <- cluster.stats(dist(pca_analysis_transform), pca_kmeans_wine2$cluster)$ch
print(paste("Calinski-Harabasz Index:", pca_ch_idx))