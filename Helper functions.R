################### Assign treatments #####################

# equal randomization if dProbTrt1 = 0.5
GetTreatment <- function(dProbTrt1)
{
  nTrt <- rbinom(1, 1, dProbTrt1)
  return(nTrt)
}

trt_ini = as.matrix(rep(0, 100))
for (i in 1:100) {
  trt_ini[i] = GetTreatment(0.5)
}
# table(trt_ini)

#################### Simulate first block patients data ####################
sim_all = function(obs, b, epsilon, beta, gamma, trt, alpha,
                   covariate_only = FALSE) {
    CovMatrix = outer(1:b, 1:b, function(x, y) {
        .7 ^ sqrt(abs(x - y))
    })
    x = mvrnorm(obs, rep(0, b), CovMatrix)
    allX = paste("x", 1:ncol(x), sep = "")
    colnames(x) = c(allX)
    eta = as.matrix(
        alpha * trt + beta[1] * (x[, 1]) ^ 2 +
            beta[2] * (x[, 2]) ^ 2 +
            beta[3] * (x[, 3]) ^ 3 +
            beta[4] * (x[, 4]) ^ 3 +
            beta[5] * (x[, 5]) ^ 1 +
            beta[6] * (x[, 6]) ^ 1 +
            ifelse(x[, 7] > 0, 1, 0) +
            ifelse(x[, 8] > 0, (x[, 8]) ^ 2, 0) +
            ifelse(x[, 9] > 0, (x[, 9]) ^ 3, 0) +
            ifelse(x[, 10] > 0, 1, 0) + gamma[1] * x[, 5] * trt + gamma[2] * x[, 6] * trt + epsilon
    )
	
	trt0 = as.matrix(rep(0,obs))
	trt1 = as.matrix(rep(1,obs))
	trt_opt = rep(0,obs)
	
	eta1 = as.matrix(
        alpha * trt1 + beta[1] * (x[, 1]) ^ 2 +
            beta[2] * (x[, 2]) ^ 2 +
            beta[3] * (x[, 3]) ^ 3 +
            beta[4] * (x[, 4]) ^ 3 +
            beta[5] * (x[, 5]) ^ 1 +
            beta[6] * (x[, 6]) ^ 1 +
            ifelse(x[, 7] > 0, 1, 0) +
            ifelse(x[, 8] > 0, (x[, 8]) ^ 2, 0) +
            ifelse(x[, 9] > 0, (x[, 9]) ^ 3, 0) +
            ifelse(x[, 10] > 0, 1, 0) + gamma[1] * x[, 5] * trt1 + gamma[2] * x[, 6] * trt1 + epsilon
    )
	
	eta0 = as.matrix(
        alpha * trt0 + beta[1] * (x[, 1]) ^ 2 +
            beta[2] * (x[, 2]) ^ 2 +
            beta[3] * (x[, 3]) ^ 3 +
            beta[4] * (x[, 4]) ^ 3 +
            beta[5] * (x[, 5]) ^ 1 +
            beta[6] * (x[, 6]) ^ 1 +
            ifelse(x[, 7] > 0, 1, 0) +
            ifelse(x[, 8] > 0, (x[, 8]) ^ 2, 0) +
            ifelse(x[, 9] > 0, (x[, 9]) ^ 3, 0) +
            ifelse(x[, 10] > 0, 1, 0) + gamma[1] * x[, 5] * trt0 + gamma[2] * x[, 6] * trt0 + epsilon
    )
	
	p = 1 / (1 + exp(-eta))
	p1 = 1 / (1 + exp(-eta1))
	p0 = 1 / (1 + exp(-eta0))
	p_opt = ifelse(p1>p0,p1,p0)
	
	for (i in 1:obs){
    if (p0[i] > p1[i]) {
       trt_opt[i] = 0 
    } else {
       trt_opt[i] = 1}}
	   
    response = as.factor(rbinom(obs, 1, p))  ## get (observed) binary response
	levels(response)=c("No","Yes")
	
	
    if (covariate_only) {
       data.frame(x) ### for subject after 100, we want to generate covariate only first
    } else {
        data.frame(x, trt, response,p, p1, p0,p_opt,trt_opt)
    }
}

###################### Simulate 2nd, 3rd, 4th... block patients data ########################

sim_sec = function(obs, x, epsilon, beta, gamma, trt, alpha) {
  allX = paste("x", 1:ncol(x), sep = "")
  colnames(x) = c(allX)
  eta = as.matrix(
    alpha * trt + beta[1] * (x[, 1]) ^ 2 +
      beta[2] * (x[, 2]) ^ 2 +
      beta[3] * (x[, 3]) ^ 3 +
      beta[4] * (x[, 4]) ^ 3 +
      beta[5] * (x[, 5]) ^ 1 +
      beta[6] * (x[, 6]) ^ 1 +
      ifelse(x[, 7] > 0, 1, 0) +
      ifelse(x[, 8] > 0, (x[, 8]) ^ 2, 0) +
      ifelse(x[, 9] > 0, (x[, 9]) ^ 3, 0) +
      ifelse(x[, 10] > 0, 1, 0) + gamma[1] * x[, 5] * trt + gamma[2] * x[, 6] * trt + epsilon
  )
  
  trt0 = as.matrix(rep(0,obs))
  trt1 = as.matrix(rep(1,obs))
  trt_opt = rep(0,obs)
  
  eta1 = as.matrix(
    alpha * trt1 + beta[1] * (x[, 1]) ^ 2 +
      beta[2] * (x[, 2]) ^ 2 +
      beta[3] * (x[, 3]) ^ 3 +
      beta[4] * (x[, 4]) ^ 3 +
      beta[5] * (x[, 5]) ^ 1 +
      beta[6] * (x[, 6]) ^ 1 +
      ifelse(x[, 7] > 0, 1, 0) +
      ifelse(x[, 8] > 0, (x[, 8]) ^ 2, 0) +
      ifelse(x[, 9] > 0, (x[, 9]) ^ 3, 0) +
      ifelse(x[, 10] > 0, 1, 0) + gamma[1] * x[, 5] * trt1 + gamma[2] * x[, 6] * trt1 + epsilon
  )
  
  eta0 = as.matrix(
    alpha * trt0 + beta[1] * (x[, 1]) ^ 2 +
      beta[2] * (x[, 2]) ^ 2 +
      beta[3] * (x[, 3]) ^ 3 +
      beta[4] * (x[, 4]) ^ 3 +
      beta[5] * (x[, 5]) ^ 1 +
      beta[6] * (x[, 6]) ^ 1 +
      ifelse(x[, 7] > 0, 1, 0) +
      ifelse(x[, 8] > 0, (x[, 8]) ^ 2, 0) +
      ifelse(x[, 9] > 0, (x[, 9]) ^ 3, 0) +
      ifelse(x[, 10] > 0, 1, 0) + gamma[1] * x[, 5] * trt0 + gamma[2] * x[, 6] * trt0 + epsilon
  )
  
  p = 1 / (1 + exp(-eta))
  p1 = 1 / (1 + exp(-eta1))
  p0 = 1 / (1 + exp(-eta0))
  p_opt = ifelse(p1>p0,p1,p0)
  
  for (i in 1:obs){
    if (p0[i] > p1[i]) {
      trt_opt[i] = 0 
    } else {
      trt_opt[i] = 1}}
  
  response = as.factor(rbinom(obs, 1, p))  ## get (observed) binary response
  levels(response)=c("No","Yes")
  data.frame(x, trt, response, p, p1, p0,p_opt,trt_opt)
}



###################### Calculate statistical power ############################

calc.power = function(p,threthold){
  res=0
  for (i in 1:M) {
    if(p[i]<threthold){
      res=res+(1/M)
    }else{
      res=res
    }
  }
  return(res)
}


####################### Replace possible NAs and NaNs ###############################

is.nan.data.frame <- function(x)
  do.call(cbind, lapply(x, is.na))

####################### Normalizer (min-max method) ##################################

# Build a `normalize()` function
normalize <- function(x) {
  if(min(x, na.rm=TRUE)!=max(x, na.rm=TRUE)) {
    res <- ((x - min(x, na.rm=TRUE)) / (max(x, na.rm=TRUE) - min(x, na.rm=TRUE)))
  } else {
    res <- 0
  }
  res
}
