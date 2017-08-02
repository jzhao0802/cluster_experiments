library(tidyverse)
library(mlr)
library(xgboost)
setwd("F:/Daniel/clustering")
source("palab_model/palab_model.R")

# ------------------------------------------------------------------------------
#
# BASIC SETUP OF EXPERIMENT
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

# create test dataset
td = create_dataset(test, "label", "patient_id", "matched_patient_id", 
                    "BI-test")
test_dataset = td$dataset
test_ids = td$ids

# ------------------------------------------------------------------------------
# LOAD TRAIN2 CREATE MLR DATASET FROM IT
# ------------------------------------------------------------------------------

# train2 is downsampled to 1:100 just like the clustered datasets, but the 
# negatives are chosen randomly, not based on clustering, therefore this is the
# data we need to benchmark the clustered models against
train2 = read_rds(file.path(data_folder, "train2.rds"))
t2d = create_dataset(train2, "label", "patient_id", "matched_patient_id", 
                     "BI-train2")
train2_dataset = t2d$dataset
train2_ids = t2d$ids

# ------------------------------------------------------------------------------
# TRAIN XGB ON TRAIN2 PREDICT TEST
# ------------------------------------------------------------------------------

# define xgboost model with default params
lrn_xgb <- makeLearner("classif.xgboost", predict.type="prob")
lrn_xgb$par.vals = list(
  nrounds = 100,
  verbose = F,
  objective = "binary:logistic",
  nthread = 20
)

# train and save model on train2
xgb_train2 <- train(lrn_xgb, train2_dataset)
write_rds(xgb_train2, file.path(results_folder, "train2_xgb.rds"))

# predict test and save pr curve
bin_num = 20
pred_train2 = predict(xgb_train2, test_dataset)
pr_train2 <- binned_perf_curve(pred_train2, x_metric="rec", y_metric="prec", 
                               bin_num=bin_num)
readr::write_csv(pr_train2$curve, file.path(results_folder, "pr_train2.csv"))

# ------------------------------------------------------------------------------
# SETUP CLUSTERING EXPERIMENTS AS A GRID
# ------------------------------------------------------------------------------

ps = makeParamSet(
  makeDiscreteParam("k", values = c(2, 3, 5, 10)),
  makeLogicalParam("normalize", default = TRUE),
  makeDiscreteParam("method", values = c("kmeans", "hclust")),
  makeDiscreteParam("dist_m", values = c("euclidean", "manhattan"), 
                    requires = quote(method == "hclust")),
  makeDiscreteParam("agg_m", values = c("single", "complete", "average"), 
                    requires = quote(method == "hclust"))
)
experiments =  generateGridDesign(ps)

# ------------------------------------------------------------------------------
#
# FUNCTION TO RUN EXPERIMENT GIVEN A SET OF PARAMETERS
#
# ------------------------------------------------------------------------------

run_experiment <- function(exp_number){
  
  # load parameters of experiment, create output folder
  params = experiments[exp_number, ]
  exp_name = paste("M_", params$method, 
                   "_K_", params$k, 
                   "_N_", as.integer(params$normalize), 
                   sep="")
  if (params$method == "hclust"){
    exp_name = paste(exp_name, 
                     "_D_", params$dist_m,
                     "_A_", params$agg_m,
                     sep="")
  }
  
  exp_folder = file.path(results_folder, exp_name)
  create_output_folder(exp_folder)
  
  # convert k from factor to integer
  k = as.numeric(as.character(params$k))
  
  # ----------------------------------------------------------------------------
  # CREATE CLUSTERS FROM POSITIVE DATASET BASED ON EXPERIMENT VARIABLES
  # ----------------------------------------------------------------------------
  
  # define neg and pos data
  ignore_cols = c("patient_id", "matched_patient_id", "label")
  cols_to_keep = setdiff(colnames(train), ignore_cols)
  pos_train = train %>% filter(label==1)
  pos_train = pos_train[, cols_to_keep]
  if (params$normalize){
    pos_train <- normalizeFeatures(pos_train, method="standardize")
  }
  
  # cluster positive data using the parameters of the experiment
  if (params$method == "hclust"){
    pos_cluster_membership = do_hclust(pos_train, dist_m = params$dist_m,
                                       agg_m = params$agg_m, k = k)
    pos_centroids = get_centroids(pos_train, pos_cluster_membership, median)
  }else{
    # for some reason yakmoR freezes R on server 101 - tried it 5 times so we
    # use the built in kmeans with 10 restarts
    km = kmeans(pos_train, centers = k, iter.max = 50, nstart = 10)
    
    pos_cluster_membership <- km$cluster
    pos_centroids <- km$centers
  }
  
  # ----------------------------------------------------------------------------
  # CREATE DOWNSAMPLED DATASETS, ONE FOR EACH CLUSTER
  # ----------------------------------------------------------------------------
  
  clusters <- rownames(pos_centroids)
  
  # number of negatives we want to keep for each positive
  ratio = 100
  
  # list holding the three clustered and downsampled mlr datasets
  clustered_datasets = list()
  clustered_datasets_ids = list()
  
  # iterate through the cluster centroids and get the closest matched negatives
  for (c in clusters){
    # get centroid of cluster
    centroid <- pos_centroids[as.character(c), ]
    
    # get patient ids of positives who belong to this cluster
    cluster_pos_id = train %>% 
      filter(label==1 & pos_cluster_membership == as.integer(c)) %>% 
      select(patient_id)
    
    # get their matched negatives
    cluster_neg_data = train %>% 
      filter(label==0 & matched_patient_id %in% cluster_pos_id$patient_id)
    
    # save the matching info
    cluster_neg_data_info = cluster_neg_data %>% 
      select(patient_id, matched_patient_id)
    
    # get rid off ids for the distance calculation
    cluster_neg_data = cluster_neg_data[,cols_to_keep]
    if (params$normalize){
      cluster_neg_data <- normalizeFeatures(cluster_neg_data, 
                                            method="standardize")
    }
    
    # get distance of each negative in the cluster to the centroid
    centroid_cluster_dists <- t(fields::rdist(centroid, cluster_neg_data))
    
    # append this column to the matching info dataframe
    cluster_neg_data_info$dist_to_centroid = centroid_cluster_dists
    
    # for each positive keep the ratio closest of the matched negatives
    cluster_neg_id = cluster_neg_data_info %>% 
      group_by(matched_patient_id) %>%
      top_n(n = -ratio, wt = dist_to_centroid) %>% 
      select(patient_id, matched_patient_id)
    
    # create new mlr dataset from the positive and negative samples
    pos_neg_ids = c(cluster_neg_id$patient_id, cluster_pos_id$patient_id)
    cluster_data = train %>% filter(patient_id %in% pos_neg_ids)
    
    cd = create_dataset(cluster_data, "label", "patient_id", 
                        "matched_patient_id", paste("BI-", c, sep=""))
    clustered_datasets[[as.integer(c)]] = cd$dataset
    clustered_datasets_ids[[as.integer(c)]] = cd$ids
  }
  
  # ----------------------------------------------------------------------------
  # TRAIN XGBOOST ON CLUSTERED DATASETS AND PREDICT WITH THE MODELS
  # ----------------------------------------------------------------------------
  
  # define list that holds clustered model pr curves
  pr_curves = list()
  # add pr curve of train2 model, i.e. the model we're benchmarking against
  pr_curves[["train2"]] = pr_train2$curve$prec[1:bin_num]
  
  # train clustered models on clustered datasets and save them
  clustered_models = list()
  for (c in clusters){
    xgb_untuned <- train(lrn_xgb, clustered_datasets[[as.integer(c)]])
    clustered_models[[as.integer(c)]] = xgb_untuned
    model_name = paste("clustered", c, "_xgb.rds", sep="")
    model_path = file.path(exp_folder, model_name)
    write_rds(xgb_untuned, model_path)
  }
  
  # ----------------------------------------------------------------------------
  # CALCULATE PR CURVES FOR INDIVIDUAL AND ENSEMBE MODELS
  # ----------------------------------------------------------------------------
  
  # ----------------------------------------------------------------------------
  # get clustered models individually
  
  # in this list each column holds the prediction of one clustered model
  clustered_preds = list()
  for (c in clusters){
    c_int = as.integer(c)
    
    # predict with model, save predictions
    pred_clustered = predict(clustered_models[[c_int]], test_dataset)
    clustered_preds[[c]] = pred_clustered$data$prob.1
    
    # calculate pr curve
    pr <- binned_perf_curve(pred_clustered, x_metric="rec", y_metric="prec", 
                            bin_num=bin_num)
    
    # add precision column of pr curve to pr curve collecting dataframe
    pr_curves[[paste("clustered", c, sep="")]] = pr$curve$prec[1:bin_num]
    
    # save all columns of pr curve
    pred_file = paste("pr_clustered", c, ".csv", sep="")
    readr::write_csv(pr$curve, file.path(exp_folder, pred_file))
  }
  # column bind prdictions of individual models
  clustered_preds = as.data.frame(clustered_preds)
  
  # ----------------------------------------------------------------------------
  # predict with the ensemble as the average
  
  avg_preds = apply(clustered_preds, 1, mean)
  
  # save pr curves
  pred_clustered$data$prob.1 = avg_preds
  pr <- binned_perf_curve(pred_clustered, x_metric="rec", y_metric="prec", 
                          bin_num=bin_num)
  pr_curves[["avg"]] = pr$curve$prec[1:bin_num]
  readr::write_csv(pr$curve, file.path(exp_folder, "pr_clustered_avg.csv"))
  
  # ----------------------------------------------------------------------------
  # ensemble of xgboosts based on sample distance to centroids
  
  # extract test data without label so we can calculate distance to centroids
  test_dataset_data = getTaskData(test_dataset)
  test_dataset_data$label =NULL
  if (params$normalize){
    test_dataset_data <- normalizeFeatures(test_dataset_data, 
                                          method="standardize")
  }
  test_dist_to_centroid <- t(fields::rdist(pos_centroids, test_dataset_data))
  
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
  
  # save pr curves
  pred_clustered$data$prob.1 = weighted_preds
  pr <- binned_perf_curve(pred_clustered, x_metric="rec", y_metric="prec", 
                          bin_num=bin_num)
  pr_curves[["weighted_avg"]] = pr$curve$prec[1:bin_num]
  readr::write_csv(pr$curve, file.path(exp_folder, "pr_clustered_weighted_avg.csv"))
  
  # ----------------------------------------------------------------------------
  # predict with the model to which centroid the test sample is closest to
  
  closest_model = apply(test_dist_to_centroid, 1, which.min)
  n = dim(clustered_preds)[1]
  clustered_preds_m = as.matrix(clustered_preds)
  closest_preds = clustered_preds_m[cbind(1:n, closest_model)]
  
  # save pr curves
  pred_clustered$data$prob.1 = closest_preds
  pr <- binned_perf_curve(pred_clustered, x_metric="rec", y_metric="prec", 
                          bin_num=bin_num)
  pr_curves[["closest"]] = pr$curve$prec[1:bin_num]
  readr::write_csv(pr$curve, file.path(exp_folder, "pr_clustered_closest.csv"))
  
  # ----------------------------------------------------------------------------
  # save pr curve collecting dataframe and check which model did better than
  # the benchmark, i.e. randomly downsampled model
  
  pr_curves = as.data.frame(pr_curves)
  better_than_random <- function(curves){
    target = curves$train2
    check_col <- function(col){
      as.integer(col > target)
    }
    as.data.frame(lapply(curves[,2:ncol(curves)], check_col))
  }
  pr_curves_mask = better_than_random(pr_curves)
  
  # check how many times any of the models outperformed the benchmark (we don't
  # worry about the last row as prec is inaccurate at that point)
  num_better = sum(pr_curves_mask[1:bin_num-1,], na.rm=T)
  final_pr <- bind_cols(pr_curves, pr_curves_mask)
  
  # save collected pr curves of the experiment along with mask
  exp_name = paste(exp_name, "_B_", num_better, ".csv", sep="")
  final_pr_file = file.path(results_folder, exp_name)
  readr::write_csv(final_pr, final_pr_file)
}

# ------------------------------------------------------------------------------
#
# RUN ALL EXPERIMENTS
#
# ------------------------------------------------------------------------------

for (e in 1:nrow(experiments)){
  run_experiment(e)  
}