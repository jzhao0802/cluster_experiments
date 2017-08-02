library(tidyverse)
setwd("F:/Daniel/clustering")


project_folder = "F:/Daniel/clustering"
results_folder = file.path(project_folder, "results/teva")
data_folder = file.path(project_folder, "data/teva")

# load 1:20 train, and 1:10 train, and 1:20 test
train = read_rds(file.path(data_folder, "train_cleaned.rds"))
train2 = read_rds(file.path(data_folder, "train2.rds"))
test = read_rds(file.path(data_folder, "test_cleaned.rds"))

# downsample train, train2 and test to 100 positives
train_pos = train %>% filter(label == 1) %>% sample_n(100)
train_neg = train %>% filter(label == 0) %>% 
  filter(matched_patient_id %in% train_pos$patient_id)
train = bind_rows(train_pos, train_neg)

train2_pos = train2 %>% filter(label == 1) %>% sample_n(100)
train2_neg = train2 %>% filter(label == 0) %>% 
  filter(matched_patient_id %in% train2_pos$patient_id)
train2 = bind_rows(train2_pos, train2_neg)

test_pos = test %>% filter(label == 1) %>% sample_n(100)
test_neg = test %>% filter(label == 0) %>% 
  filter(matched_patient_id %in% test_pos$patient_id)
test = bind_rows(test_pos, test_neg)

# save the downsampled datasets
write_rds(train, file.path(data_folder, "palabmod_train.rds"))
write_rds(train2, file.path(data_folder, "palabmod_train2.rds"))
write_rds(test, file.path(data_folder, "palabmod_test.rds"))