library(tidyverse)
library(mlr)

# ------------------------------------------------------------------------------
# LOAD AND MERGE TRAINING DATA
# ------------------------------------------------------------------------------

# This version1 built the dataset from a 1:200 training set.

data_loc = "F:/Hui/Project_2016/BI_IPF_2016/04_Summary/004_data"
project_folder = "F:/Daniel/clustering/"

# specifify that patient_id and matched_patient_id have to be read in as float
# otherwise read_csv will set some of them to NA
bi_col_types = cols(
  patient_id = col_double(),
  matched_patient_id = col_double()
)

# load positives and negatives
pos = read_csv(file.path(data_loc, "all_features_pos.csv"), col_types = bi_col_types)
neg = read_csv(file.path(data_loc, "all_features_neg.csv"), col_types = bi_col_types)
neg_extra = read_csv(file.path(data_loc, "all_features_score.csv"), col_types = bi_col_types)

# discard flag variables
flag_cols = colnames(pos)[grep("_FLAG", colnames(pos))]
no_flag_cols = setdiff(colnames(pos), flag_cols)
pos = pos[, no_flag_cols]

# merge pos and neg (while also reordering negs and discarding flags from thme)
data = bind_rows(pos, neg[,colnames(pos)], neg_extra[,colnames(pos)])

# save all training data
write_rds(data, file.path(project_folder, "data/bi/data.rds"))

# ------------------------------------------------------------------------------
# LOAD DATA AND SPLIT INTO TEST AND TRAIN
# ------------------------------------------------------------------------------

data = read_rds(file.path(project_folder, "data/bi/data.rds"))

# we cannot handle na's so let's impute all missing values
imp <- mlr::impute(data, classes = list(numeric = imputeMedian(), 
                                        integer = imputeMedian(),
                                        factor = imputeMode()))
data = imp$data

# select 2000 positives randomly wo replacement
test_pos_id = data %>% 
  filter(label==1) %>% 
  sample_n(size = 2000, replace = F) %>% 
  select(patient_id)

test = data %>% filter(matched_patient_id %in% test_pos_id$patient_id)
train = data %>% filter(!(matched_patient_id %in% test_pos_id$patient_id))

# now downsample the train to 200 so we have the same what Orla and Hui had on
# the project
train_pos = train %>% filter(label == 1)
train_neg = train %>% 
  filter(label == 0) %>% 
  group_by(matched_patient_id) %>% 
  sample_n(200)
train = bind_rows(train_pos, train_neg)

write_rds(train, file.path(project_folder, "data/bi/train.rds"))
write_rds(test, file.path(project_folder, "data/bi/test.rds"))

# ------------------------------------------------------------------------------
# CREATE TRAIN2
# ------------------------------------------------------------------------------

# Train2 is downsampled just as the clustered datasets but instead of using the
# distance to the positive centroid to decide which negative to keep, we 
# randomly keep a fixed number of negatives for each positive.

train2_neg = train %>% 
  filter(label==0) %>% 
  group_by(matched_patient_id) %>% 
  sample_n(100)

train2_pos = train %>% 
  filter(label==1)

train2 = bind_rows(train2_pos, train2_neg)
write_rds(train2, file.path(project_folder, "data/bi/train2.rds"))