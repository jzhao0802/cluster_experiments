library(tidyverse)
library(mlr)
library(xgboost)
library(palabmod)
setwd("F:/Daniel/clustering")

# ------------------------------------------------------------------------------
#
# BASIC SETUP OF EXPERIMENT
#
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# LOAD DATA AND SPLIT INTO TEST AND TRAIN
# ------------------------------------------------------------------------------

project_folder = "F:/Daniel/clustering"
results_folder = file.path(project_folder, "results/bi2")
data_folder = file.path(project_folder, "data/bi")

train = read_rds(file.path(data_folder, "train.rds"))
train2 = read_rds(file.path(data_folder, "train2.rds"))
test = read_rds(file.path(data_folder, "test.rds"))

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
  makeDiscreteParam("ratio", values = c(100))
)
experiments =  generateGridDesign(ps)

# ------------------------------------------------------------------------------
# DEFINE CLUSTER EXPERIMENT FUNCTION
# ------------------------------------------------------------------------------

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
  clustered_preds_obj = list()
  for (c in clusters){
    c_int = as.integer(c)
    clustered_preds_obj[[c]] = predict(clustered_models[[c_int]], test_dataset)
  }
  
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
  output["eval_benchmark"] = perf_get_curve_point(point, pr_benchmark)
  
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
  # extract predictions and pr evaluations for clusters WITHIN test data
  for (c in clusters){
    c_int = as.integer(c)
    cluster_size_train = clustered_datasets[[c_int]]$task.desc$size
    cluster_mask = closest_model == c_int
    cluster_size_test = sum(cluster_mask)
    
    # save cluster specific pr evaluation for benchmark model
    cluster_pred_benchmark = pred_benchmark
    cluster_pred_benchmark$data = cluster_pred_benchmark$data[cluster_mask,]
    cluster_eval_benchmark = get_pr_point(cluster_pred_benchmark, point, bin_num)
    
    # save cluster specific pr evaluation for cluster specific model
    pred_clustered = clustered_preds_obj[[c]]
    pred_clustered$data = pred_clustered$data[cluster_mask,]
    cluster_eval_cluster = get_pr_point(pred_clustered, point, bin_num)
    
    # save pr evaluation for blended model where only this cluster is
    # predicted by the cluster specific model and the rest is by the benchmark
    cluster_pred_blended = pred_benchmark
    cluster_pred_blended$data[cluster_mask,] = pred_clustered$data
    cluster_eval_blended = get_pr_point(cluster_pred_blended, point, bin_num)
    
    # add results to output row    
    cluster_name = paste("cluster", c_int, sep="")
    output[paste(cluster_name, "train_obs", sep="_")] = cluster_size_train
    output[paste(cluster_name, "test_obs", sep="_")] = cluster_size_test
    output[paste(cluster_name, "eval_benchmark", sep="_")] = cluster_eval_benchmark
    output[paste(cluster_name, "eval_clustered", sep="_")] = cluster_eval_cluster
    output[paste(cluster_name, "eval_blended", sep="_")] = cluster_eval_blended
  }
  
  # save collected pr stats and save it to csv
  results_file = file.path(results_folder, exp_name, ".csv", sep="")
  readr::write_csv(as.data.frame(output), results_file)
  output
}

# ------------------------------------------------------------------------------
# RUN EXPERRIMENT
# ------------------------------------------------------------------------------

outputs = NULL
for (e in 1:nrow(experiments)){
  print(e)
  params = experiments[e, ]
  output = clustering_experiment(params, train_dataset, train_ids, "auc", 
                                 test_dataset, lrn, pred_benchmark, 20)
  
  # concatenate the results into a single dataframe
  if(is.null(outputs)){
    outputs = bind_rows(output)
  }else{
    outputs = bind_rows(outputs, output)
  }
}
write_csv(outputs, "clustering_results.csv")