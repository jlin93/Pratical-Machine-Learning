#Packages, Library, Seed -->Need to install
library(caret) #Machine learning Library
library(ggplot2) 
library(kernlab)
library(lattice)
library(randomForest) #Random forest for classigication and regression
library(rpart) #Regressive partitioning and Regression trees
library(rpart.plot) # Decision Tree Plot
library(e1071)
library(adabag) # Boosting function for the AdaBoost ensemble method
set.seed(1234) # For reproducibility

################################################################################
#Import .csv training and testing dataset | Preliminary clearning
## Data Collection for Training set: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
## Data collection for Test set: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
## After saving both data sets into my working directory, some missing values are coded as string "#DIV/0" or "" or "NA" - these will be cahnged to NA
## Both data sets contain columns with all missing values - there will be deleted

## Loading the training data set into my R seeion replacing all missing with "NA"
trainingset <- read.csv("C:/Users/Julien/Downloads/pml-training.csv", na.strings=c("NA","#DIV/0!", ""))
str(trainingset) ## to check if there is still "#DIV/0!" or "" in the trainingset
## Loading the testing data set                        
testingset <- read.csv("C:/Users/Julien/Downloads/pml-testing.csv", na.strings=c("NA","#DIV/0!",""))
str(testingset) ## to check if there is still "#DIV/0!" or "" in the testingset

## Check dimensions for number of variables and observations
dim(trainingset)
dim(testingset)

## Delete columns with all missing values
trainingset<-trainingset[,colSums(is.na(trainingset))==0]
testingset<-testingset[,colSums(is.na(testingset))==0]

##Some variables are irrelevant to our current project: user_name, raw_timestap_part_,2 cvtd_timestamp, new_window,and num_window (column 1 to 7) --> They can be delete
trainingset<-trainingset[,-c(1:7)]
testingset<-testingset[,-c(1:7)]

##Check the dimension of the new datasets
dim(trainingset) #Result: [1] 19622 (observations) 53 (variables)
dim(testingset) #Result: [1] 20 (observations) 53 (variables)
head(trainingset)
head(testingset)

################################################################################

#Partitioning the training data set to allow cross-validation
##In order to perform cross-validation, the training dataset is partitionned into 2 sets: subTraining(75%) and subTest(25%)
##The cross-validation technique used is the random subsampling.
subsamples<-createDataPartition(y=trainingset$classe,p=0.75,list=FALSE)
subTraining<-trainingset[subsamples,]
subTesting<-trainingset[-subsamples,]
dim(subTraining)
dim(subTesting)
head(subTraining)
head(subTesting)

#Look at the Data
##In Wallace Ugulino et al., 2012 study, the variable "classe" contains 5 levels: A, B, C, D, E, assessing how well they do in a particular activity
## Plotting the variable "classe" enable to see the frequency of each levels in the subTraining dataset and compare one another
plot(subTraining$classe, col="blue", main="Bar plot of levels of the variable classe within the subTraining dataset",xlab="classe levels", ylab="Frequency")

################################################################################

#First Prediction Model using the Decision Tree.
model1<-rpart(classe~., data=subTraining, method="class")
##Predicting:
prediction1<-predict(model1,subTesting, type="class")
##Plot the Decision Tree
rpart.plot(model1,main="Classification Tree", extra=102,under=TRUE,faclen=0)

## Test results on our subTesting data set using the subTraining parameter and determine the accuracy
confusionMatrix(prediction1, subTesting$classe) # Results: Accuracy=0.7394

################################################################################
#Second Prediction Model using the Random Forest.
model2<-randomForest(classe~., data=subTraining, method="class") #Longer Technique
##Predicting:
prediction2<-predict(model2, subTesting, type="class")

## Test results on our subTesting data set using the subTraining parameter and determine the accuracy
confusionMatrix(prediction2, subTesting$classe) #Results: Accuracy=0.9955!



################################################################################
#Comparaison of models
##The model2 built from the Random Forest algorithm has a higher accuracy (=0.9955 with CI:(0.993, 0.997)) compared to the Decision Tree algorithm (=0.7394 with CI:(0.727,0.752))
##Hence, the Random Forest model2 is choosen to be applied to the Testing data set
#The expected out-of-sample error is estimated at 0.5%. 
#The expected out-of-sample error is calculated as 1-accuracy for predictions made agains the cross-validation set.
#Our Test data set comprises 20 observations. With an accuracy >99% on our cross-validation data, just a very few or none of the test samples will be missclassified.

################################################################################

#Submission
##Predict outcome levels on the original testing dataset using Random Forest Algorithm
predictfinal<-predict(model2, testingset, type="class")
predictfinal

##Write files for submission
pml_write_files=function(x){
  n=length(x)
  for (i in 1:n) {
    filename=paste0("problem_id_", i, ".txt")
    write.table(x[i],filename, quote=FALSE, row.names=FALSE, col.names=FALSE)
  }
}
pml_write_files(predictfinal)
