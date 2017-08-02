library(tidyverse)
library(mlr)
library(xgboost)
library(mclust)
library(palabmod)
library(parallel)
library(parallelMap)
setwd("F:/Daniel/clustering")

# ------------------------------------------------------------------------------
#
# BASIC SETUP OF EXPERIMENT 
#
# ------------------------------------------------------------------------------

# number of bins to use for binning the PR curve and the point to use to eval
bin_num = 20
pr_point = 25
# define variables for the hyper param tuning
cv_rounds = 2
search_rounds = 20
# how many negative to keep for each positive - keep all that's my advice
downsample_ratio = 20

# ------------------------------------------------------------------------------
# LOAD DATA AND SPLIT INTO TEST AND TRAIN
# ------------------------------------------------------------------------------

project_folder = "F:/Daniel/clustering"
results_folder = file.path(project_folder, "results/teva_10")
data_folder = file.path(project_folder, "data/teva")
test = read_rds(file.path(data_folder, "test_cleaned.rds"))
train = read_rds(file.path(data_folder, "train_cleaned.rds"))

# create train dataset
td = utils_create_dataset(train, "label", "patient_id", 
                          "matched_patient_id", "Train")
train_dataset = td$dataset
train_ids = td$ids

# create test dataset
td = utils_create_dataset(test, "label", "patient_id", 
                          "matched_patient_id", "Test")
test_dataset = td$dataset
test_ids = td$ids

train_to_be_clustered = getTaskData(train_dataset) %>% 
  select(-label)

test_to_be_clustered = getTaskData(test_dataset) %>% 
  select(-label)

pos_data = train_to_be_clustered %>% 
  filter(label==1) %>% 

pos_data_train = train_to_be_clustered %>% 
  filter(label==1) %>% 
  sample_n(1000)

mcl = mclustBIC(pos_data, G = 3)
mcl = Mclust(pos_data_train, x=mcl)
cluster_weights = predict.Mclust(mcl, train_to_be_clustered)$z
test_cluster_weights = predict.Mclust(mcl, test_to_be_clustered)$z
num_clusters = dim(cluster_weights)[2]

# ------------------------------------------------------------------------------
# TRAIN XGB ON TRAIN PREDICT TEST
# ------------------------------------------------------------------------------

# define xgboost model with default params
lrn <- makeLearner("classif.xgboost", predict.type="prob")
lrn$par.vals = list(
  nrounds = 100,
  verbose = F,
  objective = "binary:logistic",
  nthread = 20
)

lrn_benchmark <- train(lrn, train_dataset)
write_rds(lrn_benchmark, file.path(results_folder, "benchmark_model.rds"))

# predict test with the model and save pr curve
pred_benchmark = predict(lrn_benchmark, test_dataset)
pr_benchmark <- perf_binned_perf_curve(pred_benchmark, x_metric="rec", 
                                       y_metric="prec", bin_num=bin_num)
perf_plot_pr_curve(pred_benchmark)

# ------------------------------------------------------------------------------
# TRAIN XGB ON TRAIN WITH GM WEIGHTS
# ------------------------------------------------------------------------------

clusters = 1:num_clusters
clustered_models = list()
for (c in clusters){
  lrn_untuned = train(lrn, train_dataset, weights = cluster_weights[,c])
  clustered_models[[c]] = lrn_untuned
  model_name = paste("clustered", c, "_model.rds", sep="")
  model_path = file.path(results_folder, model_name)
  write_rds(lrn_untuned, model_path)
}
  
# predict all test samples with each of the cluster specific models
clustered_preds = list()
clustered_preds_obj = list()
for (c in clusters){
  # predict with model, save predictions
  pred_clustered = predict(clustered_models[[c]], test_dataset)
  clustered_preds_obj[[c]] = pred_clustered
  clustered_preds[[toString(c)]] = pred_clustered$data$prob.1
}
# column bind prdictions of individual models
clustered_preds = as.data.frame(clustered_preds)

# calculate weighted mean
weighted_preds = rowSums(clustered_preds * test_cluster_weights)/rowSums(test_cluster_weights)
# get prediction for ensemble model
master_pred = pred_benchmark
master_pred$data$prob.1 = weighted_preds
pr_master <- perf_binned_perf_curve(master_pred, x_metric="rec", 
                                       y_metric="prec", bin_num=bin_num)
perf_plot_pr_curve(pred_benchmark)