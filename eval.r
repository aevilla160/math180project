install.packages("randomForest")
library(randomForest)
diabetes <-  read.csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv",
                  na.strings = "?", stringsAsFactors = TRUE)
head(diabetes)
colnames(diabetes)
#pre process
diabetes <- subset(diabetes, select = -c(Income, DiffWalk,Stroke,HeartDiseaseorAttack,AnyHealthcare,NoDocbcCost,GenHlth,MentHlth,PhysHlth,Education))
colnames(diabetes)
diabetes <- na.omit(diabetes)

#convert to factor
str(diabetes$Diabetes_binary)
diabetes$Diabetes_binary <- as.factor(diabetes$Diabetes_binary)
str(diabetes$Diabetes_binary)
table(diabetes$Diabetes_binary)
#scale all predictors
numeric_data <- diabetes[sapply(diabetes, is.numeric)]
numeric_data <- scale(numeric_data)
X = model.matrix(Diabetes_binary ~ numeric_data - 1)
#use keras to build a neural network
library(keras)
model <- keras_model_sequential()
modelnn <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = 'relu', input_shape = c(10)) %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')
modelnn %>% compile(loss = "mse",
              optimizer = optimizer_rmsprop(),
              metrics = c("accuracy"))
modelnn %>% fit(X, diabetes$Diabetes_binary, epochs = 10, batch_size = 32)
summary(modelnn)   
## heat mpalibrary(ggplot2)
library(corrplot)

numeric_data <- diabetes[sapply(diabetes, is.numeric)]

cor_matrix <- cor(numeric_data, use = "complete.obs")
corrplot(cor_matrix, method = "color", addCoef.col = "black",
         tl.col = "black", tl.srt = 45, number.cex = 0.7)

#fit library(randomForest)
#fit 
set.seed(123) 
train_index <- sample(1:nrow(diabetes), 0.7 * nrow(diabetes))
train_data <- diabetes[train_index, ]
test_data <- diabetes[-train_index, ]

rf_model <- randomForest(Diabetes_binary ~ .,
                         data = train_data, 
                         mtry = 4,           
                         importance = TRUE)  
print(rf_model)

predictions <- predict(rf_model, newdata = test_data)
confusionMatrix <- table(test_data$Diabetes_binary, predictions)
print(confusionMatrix)

# Fit a Binomial Logistic Regression Model
model_binomial <- glm(Diabetes_binary ~ ., data = train_data, family = binomial())

# Model Summary
summary(model_binomial)

# Predictions
# Predict probabilities
probabilities <- predict(model_binomial, newdata = test_data, type = "response")

# Convert probabilities to class predictions (0 or 1)
#predictions <- ifelse(probabilities > 0.5, "1", "0")

# Confusion Matrix
library(caret)
confusion_mat <- confusionMatrix(as.factor(predictions), as.factor(test_data$Diabetes_binary))
print(confusion_mat)


#ROC Curve
library(pROC)
roc_curve <- roc(test_data$Diabetes_binary, probabilities)
plot(roc_curve, col = "blue", lwd = 2)
auc(roc_curve)

# Feature Importance
importance(rf_model)
varImpPlot(rf_model)


# Build neuarl network to predict diabetes
#library(neuralnet)
# Fit a Neural Network Model
#model_neural <- neuralnet(Diabetes_binary ~ ., data = train_data, hidden = 2)
#our ouput is binary so we use the logistic activation function
# Model Summary
#summary(model_neural)
