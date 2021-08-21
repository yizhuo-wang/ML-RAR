## Packages
rm(list = ls())
library(MASS)
library(corrplot)
library(glmnet)
library(tibble)
library(rsample)
library(caret)
library(gbm)
library(purrr)
library(randomForest)
library(kernlab)
library(e1071)
library(keras)
library(tensorflow)
library(dplyr)
library(tfdatasets)
library(pwr)
library(tidyverse)
library(pROC)
# install_tensorflow()
library(doParallel)
registerDoParallel(cores = 4)
library(weights)



source("Helper functions.R")



## Generate data

### Block 1 (1:100)
# obs: subject number
# b: biomarker number
# epsilon: tunable noise
# beta: biomarker coefficient
# gamma: biomarker-treatment interaction coefficient
# trt: treatment index
# alpha: trt main effect coefficient

trt_ini = as.matrix(rep(0, 100))
for (i in 1:100) {
  trt_ini[i] = GetTreatment(0.5)
}

# all predictors contribute
beta = as.matrix(c(0.01, 0.02, 0.01, 0.01, 0.02, 0.01))
gamma = as.matrix(c(0, 0))
epsilon = as.matrix(rep(-1.3, 100))
alpha = 1.5

df_ini = sim_all(
  obs = 100,
  b = 10,
  epsilon = epsilon,
  beta = beta,
  gamma = gamma,
  trt = trt_ini,
  alpha
)

trainData =
  df_ini[, 1:12]  ## p is unseen, should be removed from the modeling




### Block 2 (101:200)
### for subject 101-200, first decide their treatment
cov_sec = sim_all(
  obs = 100,
  b = 10,
  epsilon = epsilon,
  beta = beta,
  gamma = gamma,
  trt = trt_ini,
  alpha,
  covariate_only = TRUE
)

testData_trt1 =
  cbind(cov_sec, trt = 1)  ## if all 101-200 have treatment = 1
testData_trt0 =
  cbind(cov_sec, trt = 0)  ## if all 101-200 have treatment = 0


### Block 3 (201:300)
### for subject 201-300, first decide their treatment
cov_third = sim_all(
  obs = 100,
  b = 10,
  epsilon = epsilon,
  beta = beta,
  gamma = gamma,
  trt = trt_ini,
  alpha,
  covariate_only = TRUE
)

testData2_trt1 = cbind(cov_third, trt = 1)  ## if all 101-200 have treatment = 1
testData2_trt0 = cbind(cov_third, trt = 0)  ## if all 101-200 have treatment = 0


### 10 fold cross validation

train_control = trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 1,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)


############################# GLM ###############################
### 101:200
### fit glm using subject 1-100 data

mod_glm_1 = caret::train(
  trainData[, 1:11],
  trainData$response,
  method = "glm",
  metric = "ROC",
  family = binomial(link = 'logit'),
  trControl = train_control
)



### predict response rate for subject 101-200
pred_glm_1_trt1 = predict(mod_glm_1, testData_trt1, type = "prob")
pred_glm_1_trt0 = predict(mod_glm_1, testData_trt0, type = "prob")
pred_glm_1 = cbind(pred_glm_1_trt0[, 2], pred_glm_1_trt1[, 2])
pred_glm_1 =
  prop.table(pred_glm_1, margin = 1)  ## use this matrix to assign treatment



### assign treatment for subject 101-200 based on response rate
sec_treatment = rbinom(100, 1, pred_glm_1[, 2])
# table(sec_treatment)

### given these treatment, predict response using true model
df_sec = sim_sec(
  obs = 100,
  x = cov_sec,
  epsilon = epsilon,
  beta = beta,
  gamma = gamma,
  trt = sec_treatment,
  alpha = alpha
)
# table(df_sec$response) ## response rate increases

### 201:300
trainData_2 =
  rbind(trainData, df_sec[, 1:12])  ## combine 1-200 subjects as training

### fit glm using subject 1-200 data
mod_glm_2 = caret::train(
  trainData_2[, 1:11],
  trainData_2$response,
  method = "glm",
  metric = "ROC",
  family = binomial(link = 'logit'),
  trControl = train_control
)


### predict response rate for subject 201-300
pred_glm_2_trt1 = predict(mod_glm_2, testData2_trt1, type = "prob")
pred_glm_2_trt0 = predict(mod_glm_2, testData2_trt0, type = "prob")
pred_glm_2 = cbind(pred_glm_2_trt0[, 2], pred_glm_2_trt1[, 2])
pred_glm_2 = prop.table(pred_glm_2, margin = 1)  ## use this matrix to assign treatment

### assign treatment for subject 201-300 based on response rate
third_treatment = rbinom(100, 1, pred_glm_2[, 2])
# table(third_treatment)

### given these treatment, predict response using true model
df_third = sim_sec(
  obs = 100,
  x = cov_third,
  epsilon = epsilon,
  beta = beta,
  gamma = gamma,
  trt = third_treatment,
  alpha = alpha
)
# table(df_third$response) ## response rate is even higher

######## End of data generation, begin analysis #########
df_ini$response = as.numeric(df_ini$response) - 1
df_sec$response = as.numeric(df_sec$response) - 1
df_third$response = as.numeric(df_third$response) - 1

### Response rates summary
#### response rate for three phases separately
SptRspsRate_glm = t(as.matrix(c(mean(c(
  df_ini$response
)), mean(c(
  df_sec$response
)), mean(
  c(df_third$response)
))))

#### evaluate overall response rate as the evaluation criteria
OveralRspsRate_glm = mean(SptRspsRate_glm)

### Optimal treatment percentage
df_glm = rbind(df_ini, df_sec, df_third)
opt_trt_perc_glm = t(as.matrix(prop.table(table(df_glm[, 11] == df_glm[, 17]))))


### loss function
loss_glm =
  mean(ifelse(df_glm[, 11] == df_glm[, 17], 0, abs(df_glm$p_opt - df_glm$p)))

## ChiS-quared Test & Power analysis
trt_glm = table(df_glm$trt)
trt0_glm = trt_glm[1] # number of receiving trt 0
trt1_glm = trt_glm[2] # number of receiving trt 1
table_glm = table(df_glm$trt, df_glm$response)
rr_trt0_glm = table_glm[3] / trt_glm[1] # trt 0, response 1
rr_trt1_glm = table_glm[4] / trt_glm[2] # trt 1, response 1


## IPTW &ATE

ps_ini = rep(0.5, 100)
ps_sec_glm = rep(0, 100)
ps_third_glm = rep(0, 100)

for (i in (1:100)) {
  if (sec_treatment[i] == 1) {
    ps_sec_glm[i] = pred_glm_1[i, 2]
  } else {
    ps_sec_glm[i] = pred_glm_1[i, 1]
  }
  if (third_treatment[i] == 1) {
    ps_third_glm[i] = pred_glm_2[i, 2]
  } else {
    ps_third_glm[i] = pred_glm_2[i, 1]
  }
}


ps_glm = c(ps_ini, ps_sec_glm, ps_third_glm)
ps_glm[ps_glm == 1] = max(ps_glm[ps_glm != max(ps_glm)])
ps_glm[ps_glm == 0] = min(ps_glm[ps_glm != min(ps_glm)])
ps_glm[is.na(ps_glm)] = 0.5
ps_glm[is.infinite(ps_glm)] = min(ps_glm)
weight_glm = df_glm$trt / ps_glm + (1 - df_glm$trt) / (1 - ps_glm)
weight_glm = weight_glm[!is.na(weight_glm) &
                          !is.infinite(weight_glm)]

ATE_glm = sum(df_glm$trt * df_glm$response / ps_glm) / 300 - sum((1 -
                                                                    df_glm$trt) * df_glm$response / (1 - ps_glm)) / 300

## weighted ChiS-quared Test & Power analysis

tst_glm = wtd.chi.sq(
  df_glm$trt,
  df_glm$response,
  weight = weight_glm,
  na.rm = TRUE,
  drop.missing.levels = TRUE
)
tst_glm_p = as.numeric(tst_glm[3])




######################## Neural net ############################
### 101:200

mod_nn_1 = caret::train(
  trainData[, 1:11],
  trainData[, 12],
  method = "nnet",
  trControl = train_control,
  tuneGrid = expand.grid(size = c(10), decay =
                           c(0.1)),
  maxit = 200
)



### predict response rate for subject 101-200
pred_nn_1_trt1 = predict(mod_nn_1, testData_trt1, type = "prob")
pred_nn_1_trt0 = predict(mod_nn_1, testData_trt0, type = "prob")

pred_nn_1 = cbind(pred_nn_1_trt0[, 1], pred_nn_1_trt1[, 1])
pred_nn_1 = prop.table(pred_nn_1, margin = 1)

pred_nn_1[is.nan(pred_nn_1)] = 0.5

### assign treatment for subject 101-200 based on response rate
sec_treatment = rbinom(100, 1, pred_nn_1[, 2])
# table(sec_treatment)

### given these treatment, predict response using true model
df_sec = sim_sec(
  obs = 100,
  x = cov_sec,
  epsilon = epsilon,
  beta = beta,
  gamma = gamma,
  trt = sec_treatment,
  alpha = alpha
)
# table(df_sec$response) ## response rate increases

### 201:300

trainData_2 = rbind(trainData, df_sec[, 1:12])  ## combine 1-200 subjects as training

mod_nn_2 = caret::train(
  trainData_2[, 1:11],
  trainData_2[, 12],
  method = "nnet",
  trControl = train_control,
  tuneGrid = expand.grid(size = c(10), decay =
                           c(0.1)),
  maxit = 200
)



### predict response rate for subject 101-200
pred_nn_2_trt1 = predict(mod_nn_2, testData2_trt1, type = "prob")
pred_nn_2_trt0 = predict(mod_nn_2, testData2_trt0, type = "prob")

pred_nn_2 = cbind(pred_nn_2_trt0[, 1], pred_nn_2_trt1[, 1])
pred_nn_2 = prop.table(pred_nn_2, margin = 1)

pred_nn_2[is.nan(pred_nn_2)] = 0.5

### assign treatment for subject 201-300 based on response rate
third_treatment = rbinom(100, 1, pred_nn_2[, 2])
# table(third_treatment)

### given these treatment, predict response using true model
df_third = sim_sec(
  obs = 100,
  x = cov_third,
  epsilon = epsilon,
  beta = beta,
  gamma = gamma,
  trt = third_treatment,
  alpha = alpha
)
# table(df_third$response) ## response rate is even higher

######## End of data generation, begin analysis #########
df_sec$response = as.numeric(df_sec$response) - 1
df_third$response = as.numeric(df_third$response) - 1
### Response rates summary
#### response rate for three phases separately
SptRspsRate_nn[j, ] = t(as.matrix(c(mean(c(
  df_ini$response
)), mean(c(
  df_sec$response
)), mean(
  c(df_third$response)
))))

#### evaluate overall response rate as the evaluation criteria
OveralRspsRate_nn[j, ] = mean(SptRspsRate_nn[j, ])

### Optimal treatment percentage
df_nn = rbind(df_ini, df_sec, df_third)
opt_trt_perc_nn[j, ] =  t(as.matrix(prop.table(table(df_nn[, 11] == df_nn[, 17]))))
# opt_trt_perc_lasso_mc = colMeans(opt_trt_perc_lasso)

### loss function
loss_nn[j, ] = mean(ifelse(df_nn[, 11] == df_nn[, 17], 0, abs(df_nn$p_opt -
                                                                df_nn$p)))

## ChiS-quared Test & Power analysis
trt_nn = table(df_nn$trt)
trt0_nn[j, ] = trt_nn[1] # number of receiving trt 0
trt1_nn[j, ] = trt_nn[2] # number of receiving trt 1
table_nn = table(df_nn$trt, df_nn$response)
rr_trt0_nn[j, ] = table_nn[3] / trt_nn[1] # trt 0, response 1
rr_trt1_nn[j, ] = table_nn[4] / trt_nn[2] # trt 1, response 1

## IPTW &ATE

ps_ini = rep(0.5, 100)
ps_sec_nn = rep(0, 100)
ps_third_nn = rep(0, 100)

for (i in (1:100)) {
  if (sec_treatment[i] == 1) {
    ps_sec_nn[i] = pred_nn_1[i, 2]
  } else {
    ps_sec_nn[i] = pred_nn_1[i, 1]
  }
  if (third_treatment[i] == 1) {
    ps_third_nn[i] = pred_nn_2[i, 2]
  } else {
    ps_third_nn[i] = pred_nn_2[i, 1]
  }
}


ps_nn = c(ps_ini, ps_sec_nn, ps_third_nn)
ps_nn[ps_nn == 1] = 0.99
ps_nn[ps_nn == 0] = Inf
ps_nn[is.na(ps_nn)] = 0.5
ps_nn[is.infinite(ps_nn)] = min(ps_nn)
weight_nn = df_nn$trt / ps_nn + (1 - df_nn$trt) / (1 - ps_nn)
weight_nn = weight_nn[!is.na(weight_nn) &
                        !is.infinite(weight_nn)]

ATE_nn[j, ] = sum(df_nn$trt * df_nn$response / ps_nn) / 300 - sum((1 -
                                                                     df_nn$trt) * df_nn$response / (1 - ps_nn)) / 300

## weighted ChiS-quared Test & Power analysis

tst_nn = wtd.chi.sq(
  df_nn$trt,
  df_nn$response,
  weight = weight_nn,
  na.rm = TRUE,
  drop.missing.levels = TRUE
)
tst_nn_p[j, ] = as.numeric(tst_nn[3])





########################### Equal randomization (1:300) ###########################################
trt_equal = as.matrix(rep(0, 300))
for (i in 1:300) {
  trt_equal[i] = GetTreatment(0.5)
}
# table(trt_ini)

epsilon = as.matrix(rep(-1.3, 300))

df_equal = sim_all(
  obs = 300,
  b = 10,
  epsilon = epsilon,
  beta = beta,
  gamma = gamma,
  trt = trt_equal,
  alpha = alpha
)

df_equal$response = as.numeric(df_equal$response) - 1
OveralRspsRate_equal[j, ] = mean(c(df_equal$response))


### Optimal treatment percentage
opt_trt_perc_equal[j, ] =  t(as.matrix(prop.table(table(
  df_equal[, 11] == df_equal[, 17]
))))

### loss function
loss_equal[j, ] = mean(ifelse(df_equal[, 11] == df_equal[, 17], 0, abs(df_equal$p_opt -
                                                                         df_equal$p)))

## ChiS-quared Test & Power analysis
trt_equal = table(df_equal$trt)
trt0_equal[j, ] = trt_equal[1] # number of receiving trt 0
trt1_equal[j, ] = trt_equal[2] # number of receiving trt 1
table_equal = table(df_equal$trt, df_equal$response)
rr_trt0_equal[j, ] = table_equal[3] / trt_equal[1] # trt 0, response 1
rr_trt1_equal[j, ] = table_equal[4] / trt_equal[2] # trt 1, response 1

## IPTW &ATE
ps_equal = rep(0.5, 300)
weight_equal = df_equal$trt / ps_equal + (1 - df_equal$trt) / (1 - ps_equal)
weight_equal = weight_equal[!is.na(weight_equal) &
                              !is.infinite(weight_equal)]

ATE_equal[j, ] = sum(df_equal$trt * df_equal$response / ps_equal) /
  300 - sum((1 - df_equal$trt) * df_equal$response / (1 - ps_equal)) / 300

## weighted ChiS-quared Test & Power analysis

tst_equal = wtd.chi.sq(
  df_equal$trt,
  df_equal$response,
  weight = weight_equal,
  na.rm = TRUE,
  drop.missing.levels = TRUE
)
tst_equal_p[j, ] = as.numeric(tst_equal[3])

