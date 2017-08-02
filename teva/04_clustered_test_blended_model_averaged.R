library(tidyverse)
library(mlr)
library(xgboost)
library(palabmod)
setwd("F:/Daniel/clustering")

# ------------------------------------------------------------------------------
#
# BASIC SETUP OF EXPERIMENT - this version averages the benchmark and cluster
# specific prediction in the blended model, instead of simply replacing the 
# benchmark with the cluster specific.
#
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# LOAD DATA AND SPLIT INTO TEST AND TRAIN
# ------------------------------------------------------------------------------

project_folder = "F:/Daniel/clustering"
results_folder = file.path(project_folder, "results/teva3_additive")
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
  # extract predictions and pr evaluations for clusters WITHIN test data
  cluster_better = 0
  blended_better = 0
  for (c in clusters){
    # find number of samples and positives in train
    c_int = as.integer(c)
    clustered_dataset = clustered_datasets[[c_int]]
    cluster_size_train = clustered_dataset$task.desc$size
    cluster_pos_size_train = get_number_of_positives(clustered_dataset)
    # find number of samples and positives in train
    cluster_mask = closest_model == c_int
    cluster_size_test = sum(cluster_mask)
    test_dataset_target = getTaskData(test_dataset)[getTaskTargetNames(test_dataset)]
    test_dataset_target = test_dataset_target[cluster_mask,]
    test_dataset_pos_label = test_dataset$task.desc$positive
    cluster_pos_size_test = sum(test_dataset_target == test_dataset_pos_label)
    
    # save cluster specific pr evaluation for benchmark model
    cluster_pred_benchmark = pred_benchmark
    cluster_pred_benchmark$data = cluster_pred_benchmark$data[cluster_mask,]
    cluster_eval_benchmark = get_pr_point(cluster_pred_benchmark, point, bin_num)
    
    # save cluster specific pr evaluation for cluster specific model
    pred_clustered = clustered_preds_obj[[c]]
    pred_clustered$data = pred_clustered$data[cluster_mask,]
    cluster_eval_cluster = get_pr_point(pred_clustered, point, bin_num)
    
    # save pr evaluation for blended model: we take all of the benchmark model
    # on the test and blend in the predictions of the clustered model for only 
    # the samples that belong to this cluster
    cluster_pred_blended = pred_benchmark
    cluster_pred_blended$data[cluster_mask,] = pred_clustered$data
    # add in benchmark model instead of just replacing it
    cluster_pred_blended$data$prob.1 = (cluster_pred_blended$data$prob.1 + 
                                        pred_benchmark$data$prob.1)/2
    cluster_eval_blended = get_pr_point(cluster_pred_blended, point, bin_num)
    
    # count the number of times the clustered model is better on test than bench
    if(cluster_eval_cluster > cluster_eval_benchmark)
      cluster_better = cluster_better + 1
    # count the number of times the blended model is better than benchmark
    if(cluster_eval_blended > eval_benchmark_test)
      blended_better = blended_better + 1
    
    # add results to output row    
    cluster_name = paste("cluster", c_int, sep="")
    output[paste(cluster_name, "train_obs", sep="_")] = cluster_size_train
    output[paste(cluster_name, "train_pos_obs", sep="_")] = cluster_pos_size_train
    output[paste(cluster_name, "test_obs", sep="_")] = cluster_size_test
    output[paste(cluster_name, "test_pos_obs", sep="_")] = cluster_pos_size_test
    output[paste(cluster_name, "eval_benchmark", sep="_")] = cluster_eval_benchmark
    output[paste(cluster_name, "eval_clustered", sep="_")] = cluster_eval_cluster
    output[paste(cluster_name, "eval_blended", sep="_")] = cluster_eval_blended
  }
  
  # add counts to output row
  output["cluster_better"] = cluster_better
  output["blended_better"] = blended_better
  
  # save collected pr stats and save it to csv
  results_file = file.path(results_folder, paste(exp_name, ".csv", sep=""))
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
