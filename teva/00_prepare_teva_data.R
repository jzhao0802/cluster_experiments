library(tidyverse)
library(mlr)

# ------------------------------------------------------------------------------
# LOAD AND CLEAN TRAIN AND TEST DATASET
# ------------------------------------------------------------------------------

train = read_rds(file.path("F:/Daniel/clustering/data/teva", "train.rds"))
test = read_rds(file.path("F:/Daniel/clustering/data/teva", "test.rds"))

# delete all cols that are flooded with missing values
cols_to_delete1 = colnames(train)[grep("_4_IDX", colnames(train))]
cols_to_delete2 = colnames(train)[grep("_4_APD", colnames(train))]
cols_to_delete = c(cols_to_delete1, cols_to_delete2)
cols_to_keep = setdiff(colnames(train), cols_to_delete)
train = train[, cols_to_keep]
test = test[, cols_to_keep]

# replace missing values with 0
train[is.na(train)] = 0
test[is.na(test)] = 0

# rename atient id and matching col, so we don't need to alter the clustering script
train <- rename(train, patient_id = PATIENT_ID)
train <- rename(train, matched_patient_id = PATIENT_ID_TEST)
test <- rename(test, patient_id = PATIENT_ID)
test <- rename(test, matched_patient_id = PATIENT_ID_TEST)

# save them
write_rds(train, file.path("F:/Daniel/clustering/data/teva", "train_cleaned.rds"))
write_rds(test, file.path("F:/Daniel/clustering/data/teva", "test_cleaned.rds"))

# make train2
train2_neg = train %>% 
  filter(label==0) %>% 
  group_by(matched_patient_id) %>% 
  sample_n(10)

train2_pos = train %>% 
  filter(label==1)

train2 = bind_rows(train2_pos, train2_neg)
write_rds(train2, file.path(project_folder, "data/teva/train2.rds"))