library(tidyverse)
library(mlr)
library(xgboost)
library(palabmod)
setwd("F:/Daniel/clustering")

# ------------------------------------------------------------------------------
#
# BASIC SETUP OF EXPERIMENT - this version calculates the average and weighted
# average of the cluster specific models, and then averages these average 
# models with the benchmar.
#
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# LOAD DATA AND SPLIT INTO TEST AND TRAIN
# ------------------------------------------------------------------------------

project_folder = "F:/Daniel/clustering"
results_folder = file.path(project_folder, "results/teva6_average_additive")
data_folder = file.path(project_folder, "data/teva")

train = read_rds(file.path(data_folder, "train_cleaned.rds"))
train2 = read_rds(file.path(data_folder, "train2.rds"))
test = read_rds(file.path(data_folder, "test_cleaned.rds"))

# create train dataset
td = utils_create_dataset(train, "label", "patient_id", 
                          "matched_patient_id", "Train")
train_dataset = td$dataset
train_ids = td$ids

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
bin_num = 20
pr_benchmark <- perf_binned_perf_curve(pred_benchmark, x_metric="rec", 
                                       y_metric="prec", bin_num=bin_num)
perf_plot_pr_curve(pred_benchmark)

# ------------------------------------------------------------------------------
# DEFINE EXPERIMENT GRID
# ------------------------------------------------------------------------------

ps = makeParamSet(
  # parameters specific to clustering
  makeDiscreteParam("k", values = c(2, 3, 5)),
  makeLogicalParam("standardize", default = TRUE),
  makeDiscreteParam("method", values = c("kmeans", "hclust")),
  makeDiscreteParam("dist_m", values = c("euclidean", "manhattan"),
                    requires = quote(method == "hclust")),
  makeDiscreteParam("agg_m", values = c("ward.D"),
                    requires = quote(method == "hclust")),
  # constant parameters for the clustered dataset creation
  makeDiscreteParam("min_cluster_size", values = c(0.05)),
  makeDiscreteParam("ratio", values = c(10))
)
experiments =  generateGridDesign(ps)

# ------------------------------------------------------------------------------
# DEFINE CLUSTER EXPERIMENT FUNCTION
# ------------------------------------------------------------------------------

get_number_of_positives <- function (dataset){
  positive_label = dataset$task.desc$positive
  target_col = dataset$task.desc$target
  data = getTaskData(dataset)
  sum(data[target_col] == positive_label)
}

clustering_experiment <- function(params, train_dataset, train_ids, point,
                                  test_dataset, lrn, pred_benchmark, bin_num){
  
  # create output folder and experiment name
  exp_name = paste("M_", params$method,
                   "_K_", params$k,
                   "_S_", as.integer(params$standardize),
                   sep="")
  if (params$method == "hclust"){
    exp_name = paste(exp_name,
                     "_D_", params$dist_m,
                     "_A_", params$agg_m,
                     sep="")
  }
  exp_folder = file.path(results_folder, exp_name)
  utils_create_output_folder(exp_folder)
  
  # convert k and ratio from factor to integer
  k = as.numeric(as.character(params$k))
  ratio = as.numeric(as.character(params$ratio))
  min_cluster_size = as.numeric(as.character(params$min_cluster_size))
  
  # get clustered datasets
  cd = clustering_pos_k_datasets(train_dataset, train_ids, k, ratio, 
                                 min_cluster_size, params$method, 
                                 params$dist_m, params$agg_m, 
                                 params$standardize)
  clusters = cd[[3]]$clusters
  cluster_sizes = cd[[3]]$cluster_sizes
  centroids = cd[[3]]$centroids
  clustered_datasets = cd[[1]]
  
  # train clustered models on clustered datasets and save them
  clustered_models = list()
  for (c in clusters){
    lrn_untuned <- train(lrn, clustered_datasets[[as.integer(c)]])
    clustered_models[[as.integer(c)]] = lrn_untuned
    model_name = paste("clustered", c, "_model.rds", sep="")
    model_path = file.path(exp_folder, model_name)
    write_rds(lrn_untuned, model_path)
  }
  
  # predict all test samples with each of the cluster specific models
  clustered_preds = list()
  clustered_preds_obj = list()
  for (c in clusters){
    c_int = as.integer(c)
    
    # predict with model, save predictions
    pred_clustered = predict(clustered_models[[c_int]], test_dataset)
    clustered_preds_obj[[c]] = pred_clustered
    clustered_preds[[c]] = pred_clustered$data$prob.1
  }
  # column bind prdictions of individual models
  clustered_preds = as.data.frame(clustered_preds)
  
  # ----------------------------------------------------------------------------
  # for each test sample we find the cluster they belong to, based on 
  # Euclidean distance, then we apply the cluster specific model and the
  # benchmark model to those cluster specific samples only. 
  
  # start building up one row of the experiments results table
  output = list()
  output["experiment"] = exp_name
  output["train_obs"] = train_dataset$task.desc$size
  output["test_obs"] = test_dataset$task.desc$size
  pr_benchmark = perf_binned_perf_curve(pred_benchmark, bin_num=bin_num)
  eval_benchmark_test = perf_get_curve_point(point, pr_benchmark)
  output["eval_benchmark"] = eval_benchmark_test
  
  # extract test data without label so we can calculate distance to centroids
  test_dataset_data = getTaskData(test_dataset)
  test_dataset_data$label =NULL
  if (params$standardize){
    test_dataset_data <- normalizeFeatures(test_dataset_data,
                                           method="standardize")
  }
  test_dist_to_centroid <- t(fields::rdist(centroids, test_dataset_data))
  closest_model = apply(test_dist_to_centroid, 1, which.min)
  
  # function to return a point on a pr curve in a defensive manner
  get_pr_point <- function(pred, point, bin_num){
    # if we don't have two classes left in the cluster we return NaN
    pr_curve <- tryCatch({
      perf_binned_perf_curve(pred, bin_num=bin_num)
    }, error = function(e){
      warning(paste("Error in pr curve calculation for cluster", c,
                    "in model", exp_name, sep=" "))
    })
    if(typeof(pr_curve) != "character"){
      pr_curve <- perf_binned_perf_curve(pred, bin_num=bin_num)
      pr_eval = perf_get_curve_point(point, pr_curve)
    }else{
      pr_eval = NA
    }
    pr_eval
  }
  
  # ----------------------------------------------------------------------------
  # predict test data with average and weighteed average models blended with 
  # benchmark
  
  # average model blended with benchmark
  pred_benchmark_copy = pred_benchmark
  clustered_preds_copy = clustered_preds
  clustered_preds_copy$benchmark = pred_benchmark_copy$data$prob.1
  pred_benchmark_copy$data$prob.1 = apply(clustered_preds_copy, 1, mean)
  output["ave"] = get_number_of_positives(pred_benchmark_copy, point, bin_num)
  
  # turn distances into weights
  dist2weight <- function(x){
    normalized_row = x/sum(x)
    weight = 1/normalized_row
    normalized_weight = weight*length(x)/sum(weight)
    normalized_weight
  }
  weights_for_test_samples = t(apply(test_dist_to_centroid, 1, dist2weight))
  
  # combine clustered model predictions using weights
  weighted_preds = apply(weights_for_test_samples * clustered_preds, 1, mean)
  pred_benchmark_copy = pred_benchmark
  tmp_df = data.frame(weighted_preds, pred_benchmark_copy$data$prob.1)
  pred_benchmark_copy$data$prob.1 = apply(tmp_df, 1, mean)
  output["w_ave"] = get_number_of_positives(pred_benchmark_copy, point, bin_num)
  
  # save collected pr stats and save it to csv
  results_file = file.path(results_folder, paste(exp_name, ".csv", sep=""))
  readr::write_csv(as.data.frame(output), results_file)
  output
}

# ------------------------------------------------------------------------------
# RUN EXPERRIMENT
# ------------------------------------------------------------------------------


for (e in 1:nrow(experiments)){
  print(e)
  params = experiments[e, ]
  output = clustering_experiment(params, train_dataset, train_ids, .25, 
                                 test_dataset, lrn, pred_benchmark, 20)
  
  # concatenate the results into a single dataframe
  if(is.null(outputs)){
    outputs = bind_rows(output)
  }else{
    outputs = bind_rows(outputs, output)
  }
}
output_file = file.path(results_folder, "clustering_results.csv")
# if we have more cols than rows, transpose for easier readability
if (dim(outputs)[1] < dim(outputs)[2]){
  outputs = as.data.frame(t(as.data.frame(outputs)))
  write.csv(outputs, output_file)
}else{
  write_csv(outputs, output_file)
}  
