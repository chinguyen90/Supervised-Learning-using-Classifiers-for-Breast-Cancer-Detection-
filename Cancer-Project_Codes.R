#EXPLORING & PREPARING THE DATA
#step 1: download the data
bc <- read.csv("wisc_bc_data.csv")

#step 2: exploring and preparing the data
# examine the structure of the wbcd data frame
str(bc)

# drop the id feature
bc <- bc[-1]

# table of diagnosis
table(bc$diagnosis)

# recode diagnosis as a factor
bc$diagnosis <- factor(bc$diagnosis, levels = c("B", "M"),
                         labels = c("Benign", "Malignant"))

# table or proportions with more informative labels
round(prop.table(table(bc$diagnosis)) * 100, digits = 1)

#pie chart
pie(table(bc$diagnosis), main = "Diagnosis", col = c("Blue", "Red"))
box()
names(table(bc$diagnosis))

lbls <- paste(names(table(bc$diagnosis)), round(prop.table(table(bc$diagnosis)) * 100, digits = 1), "%")

pie(table(bc$diagnosis), labels = lbls, main = "Diagnosis", col = c("Blue", "Red"))
box()

#creat training and test data sets
bc_train <- bc[1:469, -1]
bc_test <- bc[470:569, -1]

#create labels for training and test data
bc_train_labels <- bc[1:469, 1]
bc_test_labels <- bc[470:569, 1]

##########################################

# METHOD 1: K-NN NEAREST NEIGHBOR

# normalization for numeric features
# create normalization function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
# normalize the bc data
bc_n <- as.data.frame(lapply(bc[2:31], normalize))
# confirm that normalization worked
summary(bc_n$area_mean)

# create normalized training and test data
bc_n_train <- bc_n[1:469, ]
bc_n_test <- bc_n[470:569, ]

# Step 3: training a k-NN model to the data
# load the "class" library
library(class)

bc_test_pred <- knn(train = bc_n_train, test = bc_n_test,
                      cl = bc_train_labels, k = 21)

#Step 4: Evaluating the model performance
# load the "gmodels" library
library(gmodels)

# Create the cross tabulation of predicted vs. actual
CrossTable(x = bc_test_labels, y = bc_test_pred,
           prop.chisq = FALSE)

#Accuracy of k-NN model:
accuracy_knn = (61+37)/(61+37+2+0)
accuracy_knn

# Step 5: Improving the model performance
# use the scale() function to z-score standardize a data frame
bc_z <- as.data.frame(scale(bc[-1]))

# confirm that the transformation was applied correctly
summary(bc_z$area_mean)

# create training and test datasets
bc_train <- bc_z[1:469, ]
bc_test <- bc_z[470:569, ]

# re-classify test cases
bc_test_pred <- knn(train = bc_train, test = bc_test,
                      cl = bc_train_labels, k = 21)

# Create the cross tabulation of predicted vs. actual
CrossTable(x = bc_test_labels, y = bc_test_pred,
           prop.chisq = FALSE)
#Calculate accuracy of z-score method
accuracy_z = (61+34)/(61+34+5+0)
accuracy_z

# try several different values of k
bc_n_train <- bc_n[1:469, ]
bc_n_test <- bc_n[470:569, ]

#start time
strt<-Sys.time()

bc_test_pred <- knn(train = bc_n_train, test = bc_n_test, cl = bc_train_labels, k = 1)
CrossTable(x = bc_test_labels, y = bc_test_pred, prop.chisq=FALSE)

bc_test_pred <- knn(train = bc_n_train, test = bc_n_test, cl = bc_train_labels, k = 5)
CrossTable(x = bc_test_labels, y = bc_test_pred, prop.chisq=FALSE)

bc_test_pred <- knn(train = bc_n_train, test = bc_n_test, cl = bc_train_labels, k = 11)
CrossTable(x = bc_test_labels, y = bc_test_pred, prop.chisq=FALSE)

bc_test_pred <- knn(train = bc_n_train, test = bc_n_test, cl = bc_train_labels, k = 15)
CrossTable(x = bc_test_labels, y = bc_test_pred, prop.chisq=FALSE)

bc_test_pred <- knn(train = bc_n_train, test = bc_n_test, cl = bc_train_labels, k = 21)
CrossTable(x = bc_test_labels, y = bc_test_pred, prop.chisq=FALSE)

bc_test_pred <- knn(train = bc_n_train, test = bc_n_test, cl = bc_train_labels, k = 27)
CrossTable(x = bc_test_labels, y = bc_test_pred, prop.chisq=FALSE)

#end time
print(Sys.time()-strt)

# Accuracy of the predictions using different values of k
# k=1
accuracy_1 = (58+38)/(58+38+3+1)
accuracy_1

# k=5
accuracy_5 = (61+37)/(61+37+2+0)
accuracy_5

# k=11
accuracy_11 = (61+36)/(61+36+3+0)
accuracy_11

# k=15
accuracy_15 = (61+36)/(61+36+3+0)
accuracy_15

#k=21
accuracy_21 = (61+37)/(61+37+2+0)
accuracy_21

#k=27
accuracy_27 = (61+35)/(61+35+4+0)
accuracy_27

###########################################

#METHOD 2: NAIVE BAYES
#step 3: training a model to the data
library(e1071)
bc_classifier <- naiveBayes(bc_train, bc_train_labels)


#step 4: evaluating the model
bc_eval_pred <- predict(bc_classifier,bc_test)
head(bc_eval_pred)

library(gmodels)
#create the cross tablulation of predicted vs. actual
CrossTable(bc_eval_pred, bc_test_labels, prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE, dnn = c('predicted', 'actual'))

#Knn was 98% accurate
#NaiveBayes is 95.8% accurate

#step 5: improving the model performance
bc_classifier2 <- naiveBayes(bc_train, bc_train_labels, laplace = 1)
bc_eval_pred2 <- predict(bc_classifier2, bc_test)
CrossTable(bc_eval_pred2, bc_test_labels, prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE, dnn = c('predicted', 'actual'))

#Accuracy is exactly the same (95.8%)

################################################
# METHOD 3: DECISION TREES
#Step 3: Training a model on the data

# build the simplest decision tree
library(C50)
bc_tree <- C5.0(bc_train, bc_train_labels)

# display simple facts about the tree
bc_tree

# display detailed information about the tree
summary(bc_tree)

#Step 4: Evaluating Tree model performance
# create a factor vector of predictions on test data
bc_tree_pred <- predict(bc_tree, bc_test)

# cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(bc_test_labels, bc_tree_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual class', 'predicted class'))

#Accuracy of the tree is 95%

#Step 5: Improving model performance
#Boosting the accuracy of decision trees
#boosted decision tree with 10 trials
bc_boost10 <- C5.0(bc_train, bc_train_labels,
                    trials = 10)
bc_boost10

#Get idea about the boosted tree
summary(bc_boost10)

# create a factor vector of predictions on test data
bc_tree_pred10 <- predict(bc_boost10, bc_test)

# cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(bc_test_labels, bc_tree_pred10,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual class', 'predicted class'))
# The accuracy is improved. It's 98% (higher than Naive Bayes, equal to kNN)


#TRIALS = 20
#boosted decision tree with 20 trials
bc_boost20 <- C5.0(bc_train, bc_train_labels,
                   trials = 20)
bc_boost20

#Get idea about the boosted tree
summary(bc_boost20)

# create a factor vector of predictions on test data
bc_tree_pred20 <- predict(bc_boost20, bc_test)

# cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(bc_test_labels, bc_tree_pred20,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual class', 'predicted class'))

# ACCURACY OF TRIALS 20 IS 99%

#TRIALS = 30
#boosted decision tree with 10 trials
bc_boost30 <- C5.0(bc_train, bc_train_labels,
                   trials = 30)
bc_boost30

#Get idea about the boosted tree
summary(bc_boost30)

# create a factor vector of predictions on test data
bc_tree_pred30 <- predict(bc_boost30, bc_test)

# cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(bc_test_labels, bc_tree_pred30,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual class', 'predicted class'))
#ACCURACY OF TRIALS 30 IS STILL 99%

################################################################

#METHOD 4: NEURAL NETWORKS ANN

library(MASS) # Needed to sample multivariate Gaussian distributions 
library(neuralnet) # The package for neural networks in R
library(readr)

##Step 1: Collecting and downloading data
cancer <- read_csv("wisc_bc_data.csv")
View(cancer)

#Step 2:Exploring and Preparing Data
head(cancer)

cancer <- cancer[, -c(1)]

cancer[, 1] <- as.numeric(cancer[, 1] == "M")

colnames(cancer)[1] <- "label"
colnames(cancer)[2] <- "V2"
colnames(cancer)[3] <- "V3"
colnames(cancer)[4] <- "V4"
colnames(cancer)[5] <- "V5"
colnames(cancer)[6] <- "V6"
colnames(cancer)[7] <- "V7"
colnames(cancer)[8] <- "V8"
colnames(cancer)[9] <- "V9"
colnames(cancer)[10] <- "V10"
colnames(cancer)[11] <- "V11"
colnames(cancer)[12] <- "V12"
colnames(cancer)[13] <- "V13"
colnames(cancer)[14] <- "V14"
colnames(cancer)[15] <- "V15"
colnames(cancer)[16] <- "V16"
colnames(cancer)[17] <- "V17"
colnames(cancer)[18] <- "V18"
colnames(cancer)[19] <- "V19"
colnames(cancer)[20] <- "V20"
colnames(cancer)[21] <- "V21"
colnames(cancer)[22] <- "V22"
colnames(cancer)[23] <- "V23"
colnames(cancer)[24] <- "V24"
colnames(cancer)[25] <- "V25"
colnames(cancer)[26] <- "V26"
colnames(cancer)[27] <- "V27"
colnames(cancer)[28] <- "V28"
colnames(cancer)[29] <- "V29"
colnames(cancer)[30] <- "V30"
colnames(cancer)[31] <- "V31"

cancer[1, ]

#Normalize data
# create normalization function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
# normalize the cancer data
cancer <- as.data.frame(lapply(cancer, normalize))
# confirm that normalization worked
summary(cancer$V4)

# create train and test data set
cancer_train <- cancer[1:469, ]
cancer_test <- cancer[470:569, ]

#Step 3: Training a model on the data
set.seed(12345) # to guarantee repeatable results
cancer.model <- neuralnet(formula = label ~ V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + V11 + V12 + V13 + V14 + V15 + V16 + V17 + V18 + V19 + V20 + V21 + 
                            V22 + V23 + V24 + V25 + V26 + V27 + V28 + V29 + V30 + V31,data = cancer_train)

plot(cancer.model)

# alternative plot
library(NeuralNetTools)

# plotnet
par(mar = numeric(4), family = 'serif')
plotnet(cancer.model, alpha = 0.6)

# Step 4: Evaluating model performance
# obtain model results
cancer.model.results <- compute(cancer.model, cancer_test[2:31])

# obtain predicted diagnosis values
predicted_label <- cancer.model.results$net.result

# examine the correlation between predicted and actual values
cor(predicted_label, cancer_test$label)   

#The accuracy is 0.9888

## Step 5: Improving model performance
# a more complex neural network topology with 5 hidden neurons
set.seed(12345) # to guarantee repeatable results
cancer.model2 <- neuralnet(label ~ V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + V11 + V12 + V13 + V14 + V15 + V16 + V17 + V18 + V19 + V20 + V21 + V22 + V23 + V24 + V25 + V26 + V27 + V28 + V29 + V30 + V31,data = cancer_train, hidden = 5, act.fct = "logistic")

# plot the network
plot(cancer.model2)

# plotnet
par(mar = numeric(4), family = 'serif')
plotnet(cancer.model2, alpha = 0.6)

# evaluate the results as we did before
model_results2 <- compute(cancer.model2, cancer_test[2:31])
predicted_label2 <- model_results2$net.result
cor(predicted_label2, cancer_test$label)

# The accuracy droppend to 0.975 (lower than 1 hidden node)
