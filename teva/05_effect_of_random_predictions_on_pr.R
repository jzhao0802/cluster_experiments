library(tidyverse)
library(mlr)
library(xgboost)
library(palabmod)
setwd("F:/Daniel/clustering")

# ------------------------------------------------------------------------------
#
# This experiment was designed to assess what happens if we take an XGBoost 
# model's 30k predictions and randomly shuffle 100, 500, ... , 5000 of them.
# How does this affect the PR curve?
#
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# LOAD DATA AND SPLIT INTO TEST AND TRAIN
# ------------------------------------------------------------------------------

project_folder = "F:/Daniel/clustering"
results_folder = file.path(project_folder, "results/teva4_random")
data_folder = file.path(project_folder, "data/teva")

train2 = read_rds(file.path(data_folder, "train2.rds"))
test = read_rds(file.path(data_folder, "test_cleaned.rds"))

# create train2 dataset
td = utils_create_dataset(train2, "label", "patient_id", 
                          "matched_patient_id", "Train2")
train2_dataset = td$dataset
train2_ids = td$ids

# create test dataset
td = utils_create_dataset(test, "label", "patient_id", 
                          "matched_patient_id", "Test")
test_dataset = td$dataset
test_ids = td$ids

# ------------------------------------------------------------------------------
# TRAIN XGB ON TRAIN2 PREDICT TEST
# ------------------------------------------------------------------------------

# define xgboost model with default params
lrn <- makeLearner("classif.xgboost", predict.type="prob")
lrn$par.vals = list(
  nrounds = 100,
  verbose = F,
  objective = "binary:logistic",
  nthread = 20
)

# train and save model on train2
lrn_benchmark <- train(lrn, train2_dataset)
write_rds(lrn_benchmark, file.path(results_folder, "benchmark_model.rds"))

# predict test with the model and save pr curve
pred_benchmark = predict(lrn_benchmark, test_dataset)

# ------------------------------------------------------------------------------
# replace 100, 500, 1000, 2500, 5000 preds randomly
# ------------------------------------------------------------------------------

get_pr_point <- function(pred, point, bin_num){
  # if we don't have two classes left in the cluster we return NaN
  pr_curve <- tryCatch({
    perf_binned_perf_curve(pred, bin_num=bin_num)
  }, error = function(e){
    warning(paste("Error in pr curve calculation for cluster"))
  })
  if(typeof(pr_curve) != "character"){
    pr_curve <- perf_binned_perf_curve(pred, bin_num=bin_num)
    pr_eval = perf_get_curve_point(point, pr_curve)
  }else{
    pr_eval = NA
  }
  pr_eval
}

test_size = dim(pred_benchmark$data)[1]

# define points on pr curve to run experiment for
points = c(.05, .1, .15, .2, .25, .3, .35, .4, .5)
for (p in points){
  print(p)
  point = p
  bin_num = 20
  
  samples = c(100, 500, 1000, 2500, 5000)
  num_exp = 50
  results = matrix(rep(0, length(samples)*num_exp), nrow=num_exp)
  
  for (i in 1:length(samples)){
    for(j in 1:num_exp){
      # take sample_n predictions and shuffle them
      sample_n = samples[i]
      sampled_ix = sample(1:test_size, sample_n)
      shuffle_ix =  sample(1:sample_n, sample_n)
      random_blend = pred_benchmark
      shuffled_vals = random_blend$data$prob.1[sampled_ix[shuffle_ix]]
      random_blend$data$prob.1[sampled_ix] = shuffled_vals
      results[j, i] = get_pr_point(random_blend, point, bin_num)
    }
  }
  
  # plot experiments as a boxplot
  benchmark_eval = get_pr_point(pred_benchmark, point, bin_num)
  df = data.frame(results)
  colnames(df) = samples
  df = stack(df)
  df$ind = factor(df$ind, levels=samples)
  ggplot(df, aes(x = ind, y = values)) +
    geom_boxplot() +
    geom_hline(aes(yintercept=benchmark_eval), colour="#990000", linetype="dashed") +
    ggtitle(point)
  ggsave(file.path(results_folder, paste("point_", point, ".pdf", sep="")))
}
