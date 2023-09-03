rm(list=ls())
library(tidyverse)
library(xgboost)
library(caret)
library(ggplot2)
library(pROC)
library(naniar)
library(class)
library("FNN")
library(tree)
library(rpart)
library(MASS)
library(rpart.plot)
library('randomForest')


## Load Data ============================================================
# Convert seven character variables to factor variables 
# Specify blank cells and recoginize them as missing values
setwd("/Users/sam/Desktop/Summer/STA380/spaceship_titanic/Final Version")
train <- read.csv("train.csv", header = TRUE, stringsAsFactors = T, na.strings = "")

## Data Engineering =====================================================
# Check NaN
dim(train) # 8693,14
colSums(is.na(train))
train$CryoSleep <- as.integer(as.logical(train$CryoSleep))
train$VIP <- as.integer(as.logical(train$VIP))
train$Transported<- as.integer(as.logical(train$Transported))

vis_miss(train)

# CryoSleep
# If NaN in CryoSleep, delete the whole row
train_clean <- train[complete.cases(train$CryoSleep), ]

# If NaN in activities while asleep, fill 0
train_clean$RoomService <- ifelse(train_clean$CryoSleep == 1 & is.na(train_clean$RoomService), 0, train_clean$RoomService)
train_clean$FoodCourt <- ifelse(train_clean$CryoSleep == 1 & is.na(train_clean$FoodCourt), 0, train_clean$FoodCourt)
train_clean$ShoppingMall <- ifelse(train_clean$CryoSleep == 1 & is.na(train_clean$ShoppingMall), 0, train_clean$ShoppingMall)
train_clean$Spa <- ifelse(train_clean$CryoSleep == 1 & is.na(train_clean$Spa), 0, train_clean$Spa)
train_clean$VRDeck<- ifelse(train_clean$CryoSleep == 1 & is.na(train_clean$VRDeck), 0, train_clean$VRDeck)

# Check no activities while sleeping
condition <- is.na(train_clean$RoomService) | is.na(train_clean$FoodCourt)| 
  is.na(train_clean$ShoppingMall) | is.na(train_clean$Spa) | is.na(train_clean$VRDeck)
num_rows <- sum(train_clean$CryoSleep=='1' & condition == 'TRUE')
num_rows

# Replace NA with the most frequently occurring non-missing value
train_clean$HomePlanet[is.na(train_clean$HomePlanet)] <- names(sort(table(train_clean$HomePlanet),decreasing = T))[1]
train_clean$CryoSleep[is.na(train_clean$CryoSleep)] <- names(sort(table(train_clean$CryoSleep),decreasing = T))[1]
train_clean$Destination[is.na(train_clean$Destination)] <- names(sort(table(train_clean$Destination),decreasing = T))[1]
train_clean$VIP[is.na(train_clean$VIP)] <- names(sort(table(train_clean$VIP),decreasing = T))[1]
train_clean$Cabin[is.na(train_clean$Cabin)] <- names(sort(table(train_clean$Cabin),decreasing = T))[1]

# Replace NA with mean value 
train_clean$Age[is.na(train_clean$Age)] <- mean(train_clean$Age, na.rm=T)
train_clean$RoomService[is.na(train_clean$RoomService)] <- mean(train_clean$RoomService, na.rm=T)
train_clean$FoodCourt[is.na(train_clean$FoodCourt)] <- mean(train_clean$FoodCourt, na.rm=T)
train_clean$ShoppingMall[is.na(train_clean$ShoppingMall)] <-mean(train_clean$ShoppingMall, na.rm=T)
train_clean$Spa[is.na(train_clean$Spa)] <- mean(train_clean$Spa, na.rm=T)
train_clean$VRDeck[is.na(train_clean$VRDeck)] <- mean(train_clean$VRDeck, na.rm=T)

# Cabin:split into 3 independent columns 
train_clean$Deck <- factor(sapply(strsplit(as.character(train_clean$Cabin), "/"), "[", 1))
train_clean$Deck_number <- factor(sapply(strsplit(as.character(train_clean$Cabin), "/"), "[", 2))
train_clean$Side <- factor(sapply(strsplit(as.character(train_clean$Cabin), "/"), "[", 3))
train_clean$CryoSleep <- as.factor(train_clean$CryoSleep)

train_clean$VIP <- as.factor(train_clean$VIP)
train_clean$Deck <- as.factor(train_clean$Deck)
train_clean$Transported <- as.factor(train_clean$Transported)

# EDA ======================================================================
# Transported ~ CryoSleep
ggplot(train_clean, aes(x=CryoSleep, fill=Transported)) + geom_bar(position = "fill") +
  labs(y="Percent") + scale_y_continuous(labels = scales::percent)

# Transported ~ VIP
ggplot(train_clean, aes(x=VIP, fill=Transported)) + geom_bar(position = "fill") +
  labs(y="Percent") + scale_y_continuous(labels = scales::percent)

# Transported ~ Age (VIP not in CryoSleep)
train_noSleep = filter(train_clean, CryoSleep == 0)
ggplot(data = train_noSleep, mapping = aes(x = Age, y = Transported, color = VIP))+
  geom_boxplot(notch=TRUE,
               outlier.colour="red", 
               outlier.shape = 1,
               outlier.size = 1)+
  coord_flip()

# Transported ~ HomePlanet
ggplot(train_clean, aes(x=HomePlanet, fill=Transported)) + geom_bar(position = "fill") +
  labs(y="Percent") + scale_y_continuous(labels = scales::percent)

# Transported ~ Destination
ggplot(train_clean, aes(x=Destination, fill=Transported)) + geom_bar(position = "fill") +
  labs(y="Percent") + scale_y_continuous(labels = scales::percent)

# Delete PassengerId','Name','Cabin' and 'Deck Number"Column
train_clean <- train_clean[,-which(names(train_clean) %in% c('PassengerId','Name','Cabin','Deck_number'))]

train_clean$CryoSleep <- as.factor(train_clean$CryoSleep)
train_clean$VIP <- as.factor(train_clean$VIP)
train_clean$Deck <- as.factor(train_clean$Deck)
train_clean$Transported <- as.factor(train_clean$Transported)

# Final Check
colSums(is.na(train_clean))
dim(train_clean)

########################### Logistic Regression ###################
# Divide Data
# Data Splitting -----------
## 80% of the sample size
smp_size <- floor(0.8 * nrow(train))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(train_clean)), size = smp_size)

train_split <- train_clean[train_ind, ]
validation_split <- train_clean[-train_ind, ]

# Logistic Regression Model --------

# Create null and full model with no interactions
null <- glm(Transported~-1, family = binomial,  data=train_split)
full <- glm(Transported~ . , family = binomial, data=train_split)

regForward <- step(null, scope=formula(full), direction='forward')
regBack <- step(full, scope = formula(null), direction = 'backward')

# Both models include all variables except for Deck Number

# Create the model and 
log_model <- glm(formula = Transported ~ CryoSleep + Spa + HomePlanet + VRDeck + RoomService + 
                   FoodCourt + Deck + Side + ShoppingMall + Destination + Age, family = binomial,
                 data = train_clean)

summary(log_model)

# VIP doesn't change the AIC so I removed in favor of a simpler model

# Validate Logistic Regression Model --------

# To validate we are going to predict based on the validation data of the model

predicts <- predict(log_model, newdata = validation_split)
prob_predictions <- exp(predicts)/(1+exp(predicts))

validation_split <- validation_split %>% 
  mutate("Trans_Pred" = ifelse(prob_predictions >= .6, 1, 0))

validation_split <- validation_split %>% 
  mutate("Correct" = ifelse(validation_split$Trans_Pred == validation_split$Transported, 1, 0))

# Create a Confustion Matrix to examine the data more
xpected_value <- factor(validation_split$Transported)
predicted_value <- factor(validation_split$Trans_Pred)
conf_matrix <- confusionMatrix(data=xpected_value, reference = predicted_value)

conf_matrix

sum(validation_split$Correct)/length(validation_split$Transported)

# The model predicts the validation data at 78.05% accuracy rate
# The error rate is 21.95%

####################################################################
############################# KNN ##################################
####################################################################

train_clean <- train_clean %>% 
  mutate('Side' = ifelse(train_clean$Side == "P", 1, 0))

train_clean <- train_clean %>% 
  mutate('Deck' = factor(paste(train_clean$Deck, train_clean$Deck_number))) 

#train_clean <- train_clean[, -which(names(train_clean) == "Deck_number")]

##Converting the categorical variables to dummy variables
cat_vars_to_convert <- c("HomePlanet","Destination")
formula <- as.formula(paste("~", paste(cat_vars_to_convert, collapse = "+"),"-1"))
dummy_cat_data <- as.data.frame(model.matrix(formula, data = train_clean))

#Combine both numerical and dummy variables data frame
full_train_data <-  cbind(train_clean[,c("CryoSleep","Transported","RoomService","FoodCourt","Spa","VRDeck")], dummy_cat_data)

#Normalize age number between 0-1
#full_train_data$Age <- (full_train_data$Age - min(full_train_data$Age)) / (max(full_train_data$Age) - min(full_train_data$Age))
full_train_data$RoomService <- (full_train_data$RoomService - min(full_train_data$RoomService)) / (max(full_train_data$RoomService) - min(full_train_data$RoomService))
full_train_data$FoodCourt <- (full_train_data$FoodCourt - min(full_train_data$FoodCourt)) / (max(full_train_data$FoodCourt) - min(full_train_data$FoodCourt))
#full_train_data$ShoppingMall <- (full_train_data$ShoppingMall - min(full_train_data$ShoppingMall)) / (max(full_train_data$ShoppingMall) - min(full_train_data$ShoppingMall))
full_train_data$Spa <- (full_train_data$Spa - min(full_train_data$Spa)) / (max(full_train_data$Spa) - min(full_train_data$Spa))
full_train_data$VRDeck <- (full_train_data$VRDeck - min(full_train_data$VRDeck)) / (max(full_train_data$VRDeck) - min(full_train_data$VRDeck))
glimpse(full_train_data)

##Dividing train and validation set
set.seed(123)  # Set a seed for reproducibility

##KNN model
results_df <- data.frame(k = numeric(), Error_Rate = numeric(),cv_val=numeric())

# Split the data into predictors (X) and target (Y)
X <- full_train_data[,-which(names(full_train_data)=="Transported")]
Y <- full_train_data$Transported

# Feature selection using RFE
#feature_sizes <- seq(2, length(names(full_train_data))-1, by = 2)  
#rfe_model <- rfe(x = X, y = Y, sizes = feature_sizes, rfeControl = rfeControl(functions = caretFuncs, method = "cv", number = 5))
# Perform k-NN with cross-validation on the reduced feature set
#Tried various k-fold cv iterations and found that the 8-fold cv is giving the optimal values
for (cv_val in c(8)) {
  k_values <- seq(5, 100, 5)
  ctrl <- trainControl(method = "cv", number = cv_val)
  knn_model <- train(Transported ~., 
                     data = full_train_data, method = "knn",
                     trControl = ctrl, 
                     metric = "Accuracy",
                     tuneGrid = data.frame(k = k_values))
  results <- knn_model$results %>% 
    mutate(
      "Error_Rate" = 1-Accuracy
    )
  results <- results[,c("k","Accuracy","Error_Rate")]
  results_full <- cbind(data.frame(cv_val),results)
  results_df <- rbind(results_df,cbind(results[which.min(results$Error_Rate),c("k","Error_Rate")],data.frame(c(cv_val))) )
  cat("Min k and error_rate for cv value as ",cv_val," is :",results[which.min(results_df$Error_Rate),]$k," ",results_df[which.min(results_df$Error_Rate),]$Error_Rate,"\n")
}

ggplot(results_full %>% filter(cv_val==8), aes(x = k, y = Error_Rate)) +
  geom_line(color = "white") +
  geom_point(color = "white") +
  xlab("k") +
  ylab("Error Rate") +
  ggtitle("KNN Error Rate for 8-fold CV") +
  theme_minimal()+
  theme_dark()+  # Set the background to dark
  theme(
    plot.background = element_rect(fill = "black"),  
    panel.background = element_rect(fill = "black"),  
    axis.line = element_line(color = "white"),
    axis.text = element_text(color = "white"),  
    axis.title = element_text(color = "white"),  
    plot.title = element_text(color = "white") 
  )

####################################################################
########################### Decision Tree ##########################
####################################################################

set.seed(1311)
#------------------------------------------------------------
#Splitting Training data into train vs validation set
val_id<-sample(1:nrow(train_clean),nrow(train_clean)/5)
val = train_clean[val_id,] #list of random indices passed as training set
train_clean= train_clean[-val_id,]
#-----------------------------------------------
#tree implementation using rpart()
#fit a big tree using rpart.control
big.tree=rpart(train_clean$Transported~HomePlanet+CryoSleep+Destination+Age+VIP+
                 RoomService+FoodCourt+ShoppingMall+Spa+VRDeck+Deck+Side,data=train_clean,
               control=rpart.control(minsplit=5,cp=0.0001))
nbig = length(unique(big.tree$where))
cat('size of big tree: ',nbig,'\n')
rpart.plot(big.tree)
#----------------------------------------------------
#calculate accuracy for the big tree on training set
predicted_values <- predict(big.tree, newdata = train_clean,type="class")
actual_values <- train_clean$Transported
accuracy_train <- sum(predicted_values == actual_values) / nrow(train_clean)
cat('Accuracy of big tree: ',accuracy_train,'(Potential Overfitting?)','\n')
#--------------------------------------------------------------
#calculate accuracy for the big tree on test set
predicted_values <- predict(big.tree, newdata = val,type="class")
actual_values <- val$Transported
accuracy_val <- sum(predicted_values == actual_values) / nrow(val)
cat('Accuracy of big tree on test set: ',accuracy_val,'\n')

####################################################################
######################### Pruned Decision Tree #####################
####################################################################

#--perform cross validation to find plot of pruning parameter vs  cross validation error and identify the best pruning parameter
plotcp(big.tree)
bestcp=big.tree$cptable[which.min(big.tree$cptable[,"xerror"]),"CP"]
print(bestcp)
#--perform pruning now--------------------------------
pruned_model<-prune(big.tree,cp=bestcp)
nbig_pruned = length(unique(pruned_model$where))
cat('size of pruned tree: ',nbig_pruned,'\n')
rpart.plot(pruned_model,main='Pruned Tree')
#----Accuracy of pruned tree on test set------------
predicted_values <- predict(pruned_model,newdata = val,type="class")
actual_values <- val$Transported
accuracy_val_prun <- sum(predicted_values == actual_values) / nrow(val)
cat('Accuracy of pruned tree on test set: ',accuracy_val_prun,'\n')

####################################################################
########################### Bagging ################################
####################################################################

#Bagging through random forest implementation
rffit=randomForest(train_clean$Transported~HomePlanet+CryoSleep+Destination+Age+VIP+
                     RoomService+FoodCourt+ShoppingMall+Spa+VRDeck+Deck+Side,data=train_clean,mtry=12,ntree=500)
varImpPlot(rffit,main="Variable importance plot for Bagging")
plot(rffit, main='Bagging')
#-----------------------------------------------------------------------------
#Bagging Accuracy in test set
predicted_values <- predict(rffit, newdata = val,type="class")
actual_values <- val$Transported
accuracy_v_bag <- sum(predicted_values == actual_values) / nrow(val)
cat('Accuracy of bagging on test set: ',accuracy_v_bag,'\n')

####################################################################
########################### Random Forest ##########################
####################################################################

#Random forest implementation
best_accuracy <-0
best_mtry<-0
var_importance_list <- list()
for (i in 1:12) {
  rf=randomForest(train_clean$Transported~HomePlanet+CryoSleep+Destination+Age+VIP+
                    RoomService+FoodCourt+ShoppingMall+Spa+VRDeck+Deck+Side,
                  data=train_clean,mtry=i,ntree=500)
  varImpPlot(rf,main= paste("Variable importance plot for mtry: ",i))
  plot(rf,main= paste("Error in Random Forest for mtry: ",i))
  predicted_values <- predict(rf, newdata = val,type="class")
  actual_values <- val$Transported
  accuracy_v_rf <- sum(predicted_values == actual_values) / nrow(val)
  if(accuracy_v_rf>best_accuracy)
  {best_accuracy<-accuracy_v_rf
  best_mtry<-i}
  #cat('Accuracy of Random Forest on test set for',i,'features',accuracy_v_rf,'\n')
}
cat('Best mtry:', best_mtry)
cat('Best Accuracy:', best_accuracy)

####################################################################
########################### XGBoosting #############################
####################################################################

# xgboost + 5-fold-cross-validation
#install.packages("xgboost")

set.seed(42)
train_indices <- createDataPartition(train_clean$Transported, p = 0.8, list = FALSE)
train_data <- train_clean[train_indices, ]
test_data <- train_clean[-train_indices, ]

# Convert character variables to factors in the training data
# Convert the target variable 'y_train' and 'y_test' to numeric
x_train <- data.matrix(train_data[, -which(names(train_data) == "Transported")])
y_train <- as.numeric(as.character(train_data$Transported))

x_test <- data.matrix(test_data[, -which(names(test_data) == "Transported")])
y_test <- as.numeric(as.character(test_data$Transported))

# Set the seed for reproducibility
set.seed(42)
# Define the number of folds for cross-validation
n_folds <- 5
# Create cross-validation folds
cv_folds <- createFolds(y_train, k = n_folds)

# Define the parameter grid for hyperparameter tuning
param_grid <- expand.grid(
  eta = c(0.01, 0.05, 0.09),
  max_depth = c(6, 7, 8),
  nrounds = c(100, 300, 500),
  subsample = c(0.6, 0.7, 0.8)
)

# Initialize a matrix to store validation accuracies for each parameter combination and fold
val_accuracy <- matrix(NA, nrow = nrow(param_grid), ncol = n_folds)

# Perform k-fold cross-validation
for (fold in 1:n_folds) {
  # Get the training and validation data for this fold
  x_train_fold <- x_train[-cv_folds[[fold]], ]
  y_train_fold <- y_train[-cv_folds[[fold]]]
  x_val_fold <- x_train[cv_folds[[fold]], ]
  y_val_fold <- y_train[cv_folds[[fold]]]
  
  # Train the xgboost model with different parameter combinations
  for (i in 1:nrow(param_grid)) {
    # Define xgboost model
    model <- xgboost(data = as.matrix(x_train_fold), label = as.numeric(y_train_fold),
                     eta = param_grid$eta[i], max_depth = param_grid$max_depth[i],
                     nrounds = param_grid$nrounds[i], subsample = param_grid$subsample[i],objective = "binary:logistic",nthread = -1)
    
    # Predict on the validation set
    yhat_val <- predict(model, as.matrix(x_val_fold))
    
    # Convert probabilities to binary predictions
    yhat_val_class <- ifelse(yhat_val > 0.5, 1, 0)
    
    # Calculate validation set accuracy
    val_accuracy[i, fold] <- mean(yhat_val_class == y_val_fold)
  }
}

# Find the index of the best parameter combination
best_param_index <- which.max(rowMeans(val_accuracy))

# Get the best parameters
best_eta <- param_grid$eta[best_param_index]
best_max_depth <- param_grid$max_depth[best_param_index]
best_nrounds <- param_grid$nrounds[best_param_index]
best_subsample <- param_grid$subsample[best_param_index]

cat("Best Parameters:\n")
cat("eta =", best_eta, "\n")
cat("max_depth =", best_max_depth, "\n")
cat("nrounds =", best_nrounds, "\n")
cat("subsample =", best_subsample, "\n")

# Use the best parameters to train the final model
final_model <- xgboost(data = as.matrix(x_train), label = y_train,
                       eta = best_eta, max_depth = best_max_depth,
                       nrounds = best_nrounds, subsample = best_subsample,objective = "binary:logistic", nthread = -1)

# Evaluate the final model's performance on the test set
yhat_test <- predict(final_model, as.matrix(x_test))
yhat_test_class <- ifelse(yhat_test > 0.5, 1, 0)
accuracy_test <- mean(yhat_test_class == y_test)
cat("Final Model accuracy on test set:", accuracy_test, "\n")

# Visualizations ==========================================

# Feature Importance Plot =================================
# Convert Feature to factor and reorder based on Gain
feature_importance <- xgb.importance(feature_names = colnames(x_train), model = final_model)
feature_importance$Feature <- factor(feature_importance$Feature, levels = feature_importance$Feature[order(-feature_importance$Gain)])

# Plot the bar chart with re-ordered levels
ggplot(data = feature_importance, aes(x = Feature, y = Gain)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Feature Importance", x = "Feature", y = "Gain") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        panel.background = element_rect(fill = "black"),
        plot.background = element_rect(fill = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "white"),
        axis.text = element_text(color = "white"),
        axis.title = element_text(color = "white"),
        plot.title = element_text(color = "white"))

# Learning Curves ==========================================
train_sizes <- seq(0.1, 1, 0.1)
train_accuracy <- c()
val_accuracy <- c()

for (size in train_sizes) {
  n_samples <- floor(size * nrow(train_data))
  train_accuracy_fold <- c()
  val_accuracy_fold <- c()
  
  for (fold in 1:n_folds) {
    indices <- sample(1:nrow(train_data), n_samples)
    x_subset <- x_train[indices, ]
    y_subset <- y_train[indices]
    
    # Get the training and validation data for this fold
    x_val_fold <- x_train[cv_folds[[fold]], ]
    y_val_fold <- y_train[cv_folds[[fold]]]
    
    # Train the model on the subset
    model <- xgboost(data = as.matrix(x_subset), label = y_subset,
                     eta = best_eta, max_depth = best_max_depth,
                     nrounds = best_nrounds, subsample = best_subsample)
    
    # Predict on the training and validation sets
    yhat_train <- predict(model, as.matrix(x_subset))
    yhat_val <- predict(model, as.matrix(x_val_fold))
    
    # Convert probabilities to binary predictions
    yhat_train_class <- ifelse(yhat_train > 0.5, 1, 0)
    yhat_val_class <- ifelse(yhat_val > 0.5, 1, 0)
    
    # Calculate training and validation set accuracies for this fold
    train_accuracy_fold <- c(train_accuracy_fold, mean(yhat_train_class == y_subset))
    val_accuracy_fold <- c(val_accuracy_fold, mean(yhat_val_class == y_val_fold))
  }
  
  # Take the average of accuracies from all folds
  train_accuracy <- c(train_accuracy, mean(train_accuracy_fold))
  val_accuracy <- c(val_accuracy, mean(val_accuracy_fold))
}

# Plot Learning Curves
ggplot(data = data.frame(Train_Size = train_sizes,
                         Training_Accuracy = train_accuracy,
                         Validation_Accuracy = val_accuracy), 
       aes(x = Train_Size)) +
  geom_line(aes(y = Training_Accuracy, color = "Training Accuracy"), size = 1.2) +
  geom_line(aes(y = Validation_Accuracy, color = "Validation Accuracy"), size = 1.2) +
  geom_point(aes(y = max(val_accuracy), color = "Final Model Accuracy"), size = 3) +
  labs(title = "Learning Curves",
       x = "Training Set Size", y = "Accuracy") +
  scale_color_manual(values = c("Training Accuracy" = "#1f78b4", 
                                "Validation Accuracy" = "#e31a1c", 
                                "Final Model Accuracy" = "#33a02c")) +
  theme_minimal() +
  theme(panel.background = element_rect(fill = "black"),
        plot.background = element_rect(fill = "black"),
        panel.grid.major = element_line(color = "white"),
        panel.grid.minor = element_blank(),
        axis.line = element_line(color = "white"),
        axis.text = element_text(color = "white"),
        axis.title = element_text(color = "white"),
        plot.title = element_text(color = "white"))

# Error Distribution Plot ===========================
error <- abs(yhat_test - y_test)

# Plot the histogram with a black background and green bars
ggplot(data = data.frame(Error = error), aes(x = Error)) +
  geom_histogram(fill = "green", color = "white", bins = 30, alpha = 0.7) +  # Adjust the alpha for transparency
  labs(title = "Error Distribution", x = "Absolute Error") +
  theme_minimal() +
  theme(panel.background = element_rect(fill = "black"),  # Set background color to black
        plot.background = element_rect(fill = "black"),
        panel.grid.major = element_blank(),  # Remove grid lines
        panel.grid.minor = element_blank(),
        axis.line = element_line(color = "white"),
        axis.text = element_text(color = "white", size = 12),  # Adjust axis text size
        axis.title = element_text(color = "white", size = 14),  # Adjust axis title size
        plot.title = element_text(color = "white", size = 18),  # Adjust plot title size
        legend.position = "none")  # Remove legend

# Confusion Matrix ================================
# Calculate the predicted class based on the probability threshold of 0.5
yhat_test_class <- ifelse(yhat_test > 0.5, 1, 0)
# Calculate the predicted class based on the probability threshold of 0.5
yhat_test_class <- ifelse(yhat_test > 0.5, 1, 0)
# Create the confusion matrix
conf_matrix <- table(Actual = y_test, Predicted = yhat_test_class)
print(conf_matrix)
# Convert confusion matrix to data frame
conf_matrix_df <- as.data.frame(conf_matrix)
# Plot confusion matrix heat map with a black background and gradient colors
ggplot(conf_matrix_df, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile(color = "black", size = 0.5) +  # Add black borders to the tiles
  geom_text(aes(label = Freq), vjust = 1, color = "white", size = 4) +  # Set text color to white and adjust font size
  scale_fill_gradient(low = "#000627", high = "#08521c") +  # Gradient color from black (#000000) to blue (#08519c)
  labs(title = "Confusion Matrix Heat Map", x = "Predicted", y = "Actual") +
  theme_minimal() +
  theme(panel.background = element_rect(fill = "black"),  # Set background color to black
        plot.background = element_rect(fill = "black"),
        panel.grid.major = element_blank(),  # Remove grid lines
        panel.grid.minor = element_blank(),
        axis.line = element_line(color = "white"),
        axis.text = element_text(color = "white", size = 12),  # Adjust axis text size
        axis.title = element_text(color = "white", size = 14),  # Adjust axis title size
        plot.title = element_text(color = "white", size = 18),  # Adjust plot title size
        legend.position = "none")  # Remove legend

