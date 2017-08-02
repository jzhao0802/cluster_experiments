library(tidyverse)
library(mlr)
library(xgboost)
setwd("F:/Daniel/clustering")
source("palab_model/palab_model.R")

# ------------------------------------------------------------------------------
#
# This script tries to make sure that we can replicate the PR curve Orla and Hui
# got during the project. They trained on 1:200 and tested on 1:853. We check
# both XGBoost and RF both with default parameters
#
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# LOAD DATA AND SPLIT INTO TEST AND TRAIN
# ------------------------------------------------------------------------------

project_folder = "F:/Daniel/clustering"
results_folder = file.path(project_folder, "results/bi")
data_folder = file.path(project_folder, "data/bi")

train = read_rds(file.path(data_folder, "train.rds"))
test = read_rds(file.path(data_folder, "test.rds"))

# given a dataframe this returns an mlr classification dataset with matching
create_dataset <- function(data, target_col, id_col, match_col, name){
  data[[target_col]] = as.factor(data[[target_col]])
  matches = as.factor(data[[match_col]])
  data[[match_col]] = NULL
  ids =  data[[id_col]]
  data[[id_col]] = NULL
  dataset <- makeClassifTask(id = name, data = data, target = target_col,
                             positive = 1, blocking = matches)
  list(ids=ids, dataset=dataset)
}

# create train dataset
td = create_dataset(train, "label", "patient_id", "matched_patient_id", 
                    "BI-train")
train_dataset = td$dataset
train_ids = td$ids

# create test dataset
td = create_dataset(test, "label", "patient_id", "matched_patient_id", 
                    "BI-test")
test_dataset = td$dataset
test_ids = td$ids

# ------------------------------------------------------------------------------
# TRAIN XGB ON TRAIN PREDICT TEST
# ------------------------------------------------------------------------------

# define xgboost model with default params
lrn_xgb <- makeLearner("classif.xgboost", predict.type="prob")
lrn_xgb$par.vals = list(
  nrounds = 100,
  verbose = F,
  objective = "binary:logistic",
  nthread = 20
)

# train and save model on train
xgb_train <- train(lrn_xgb, train_dataset)
write_rds(xgb_train, file.path(results_folder, "train_xgb.rds"))

# predict test and save pr curve
bin_num = 20
pred_train = predict(xgb_train, test_dataset)
pr_train <- binned_perf_curve(pred_train, x_metric="rec", y_metric="prec", 
                              bin_num=bin_num)
readr::write_csv(pr_train$curve, file.path(results_folder, "xgb_pr_train.csv"))

# ------------------------------------------------------------------------------
# TRAIN XGB ON TRAIN PREDICT TEST
# ------------------------------------------------------------------------------

# define xgboost model with default params
lrn_rf <- makeLearner("classif.ranger", predict.type="prob")
lrn_rf$par.vals = list(
  mtry = round(sqrt(sum(train_dataset$task.desc$n.feat))),
  num.trees = 500,
  min.node.size = 10,
  nthread = 20
)

# train and save model on train2
rf_train <- train(lrn_rf, train_dataset)
write_rds(rf_train, file.path(results_folder, "train_rf.rds"))

# predict test and save pr curve
bin_num = 20
pred_train = predict(rf_train, test_dataset)
pr_train <- binned_perf_curve(pred_train, x_metric="rec", y_metric="prec", 
                              bin_num=bin_num)
readr::write_csv(pr_train$curve, file.path(results_folder, "rf_pr_train.csv"))