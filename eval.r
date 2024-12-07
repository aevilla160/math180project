install.packages("randomForest")
library(randomForest)
diabetes <-  read.csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv",
                  na.strings = "?", stringsAsFactors = T)
head(diabetes)
colnames(diabetes)
#pre process
diabetes <- subset(diabetes, select = -c(Income, DiffWalk,Stroke,HeartDiseaseorAttack,AnyHealthcare,NoDocbcCost,GenHlth,MentHlth,PhysHlth,Education))
colnames(diabetes)
diabetes <- na.omit(diabetes)

str(diabetes$Diabetes_binary)
diabetes$Diabetes_binary <- as.factor(diabetes$Diabetes_binary)
str(diabetes$Diabetes_binary)
table(diabetes$Diabetes_binary)

#fit 
set.seed(123) 
train_index <- sample(1:nrow(diabetes), 0.7 * nrow(diabetes))
train_data <- diabetes[train_index, ]
test_data <- diabetes[-train_index, ]

rf_model <- randomForest(Diabetes_binary ~ .,
                         data = train_data, 
                         mtry = 3,           
                         importance = TRUE)  
print(rf_model)

predictions <- predict(rf_model, newdata = test_data)
confusionMatrix <- table(test_data$Diabetes_binary, predictions)
print(confusionMatrix)

install.packages("caret")

install.packages("nnet")
library(nnet)

model_multinom <- multinom(Diabetes_binary ~ ., data = train_data)

summary(model_multinom)

predictions <- predict(model_multinom, newdata = test_data)
library(caret)
confusion_mat <- confusionMatrix(predictions, test_data$Diabetes_binary)
print(confusion_mat)

# Now we do the heat map
install.packages("pheatmap")
library(pheatmap)
install.packages("RColorBrewer")
library(RColorBrewer)
