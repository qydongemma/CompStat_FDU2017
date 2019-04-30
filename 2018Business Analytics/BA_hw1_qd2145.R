rm(list=ls())
setwd("/Users/apple/Desktop/18 fall/Business Analytics/Homework/HW01")
### Homework 01 - Business Analytics
### Qingyuan Dong (qd2145)

## Question 1 (Linear Regression)
#treat grade as continuous variable
school <- read.csv('school_data.csv',sep=',',header=TRUE)
sch <- school[c(2,3,12)]
attach(sch)
grade.lr <- lm(perc_math_above4 ~ grade)
summary(grade.lr)

par(mfrow = c(1,1))
plot(grade, perc_math_above4)
abline(grade.lr)

#treat grade as categorical variable
library(psych)
sch = cbind(sch, dummy.code(sch$grade))
#head(sch)
sch <- sch[-c(1,4)]
#head(sch)
grade_reg = lm(perc_math_above4 ~ . - community_school, data = sch)
summary(grade_reg)

#multivariate regression
multi_reg = lm(perc_math_above4 ~ ., data = sch)
summary(multi_reg)

###### Question 2 (Linear Regression) ###############

egg <- read.csv('egg_production.csv',sep=',',header=TRUE)
summary(egg)
attach(egg)
egg_reg <- lm(eggs ~ feed)
summary(egg_reg)
egg_fullreg <- lm(eggs ~ .,data=egg)
summary(egg_fullreg)

plot(feed, temperature)
abline(h=0)
abline(h=35)

#create a new variable
tem_sec <- ifelse(temperature < 35 & temperature > 0, 1, 0)
egg <- cbind(egg,tem_sec)
egg_newreg <- lm(eggs ~ .,data=egg)
summary(egg_newreg)

# best model
egg_best <- lm(eggs ~ feed + tem_sec)
summary(egg_best)
confint(egg_best, level = 0.99)

# predict eggs based on the best model
predict(egg_best, data.frame(feed = 25, tem_sec = 0), 
        interval = "prediction",level=0.9)

###### Question 3 (Linear Model Selection) ###############

ibm <- read.csv('ibm_return.csv',sep=',',header=TRUE)
ibm$Date = as.Date(ibm$Date,"%m/%d/%Y")
summary(ibm)

# divide data
attach(ibm)
cutoffDate = as.Date("2013-4-2")
train_data = subset(ibm, Date < cutoffDate)
test_data = subset(ibm, Date >= cutoffDate)

# separate data into 4 months, and train on the following month
train_set = list()
test_set = list()
date_range = range(ibm$Date)

for (i in 1:4){
  offset = (i - 1) * 30
  train_set[[i]] = subset(ibm, ibm$Date >= (date_range[1]+offset) & ibm$Date < (date_range[1]+offset+4*30))
  test_set[[i]] = subset(ibm, ibm$Date >= (date_range[1]+offset+4*30) & ibm$Date < (date_range[1]+offset+5*30))
}

head(train_set[[1]])
head(test_set[[2]])

# try traditional train-test methods on each fold
test_mse = 0
for (i in 1:4){
  traReg = lm(Return ~ X1D, data = train_set[[i]] )
  pred = predict(traReg, test_set[[i]])
  test_mse = test_mse + mean( ( test_set[[i]]$Return - pred )^2 )
  print(test_mse)
}
test_mse = test_mse/4
test_mse

# find the best model using best subset selection
library(leaps)
library(ISLR)
# define a new function (refer to https://rpubs.com/davoodastaraky/subset)
predict.regsubsets =function (object ,newdata ,id ,...){
  form=as.formula(object$call [[2]])
  mat=model.matrix(form,newdata)
  coefi=coef(object ,id=id)
  xvars=names(coefi)
  mat[,xvars]%*%coefi
}

MSE = matrix(NA, 4, 8, dimnames=list(NULL, paste(1:8)))
for(j in 1:4){
  best.subset = regsubsets(Return ~ ., data = train_set[[j]], nvmax = 8)
  for(t in 1:8){
    pred = predict.regsubsets(best.subset, test_set[[j]],id = t)
    MSE[j,t] = mean( (test_set[[j]]$Return - pred)^2)
  }
}
mean.MSE = apply(MSE, 2, mean)
mean.MSE
par(mfrow=c(1,1))
plot(mean.MSE,type='b')

best.size = which(mean.MSE == min(mean.MSE))
reg.best = regsubsets(Return ~ .,data = train_data, nvmax=19)
coef(reg.best, best.size)

#evaluate the mse on the test data
pred = predict.regsubsets(reg.best, test_data,id = best.size)
test.MSE = mean( (test_data$Return - pred)^2)
test.MSE


# lasso regression
library(glmnet)

# Set up a grid of Lambda parameters to try
grid = c(0, 0.001, 0.01, 0.1, 1, 10, 100, 1000)

# 4-flod validation loop
lasso.mse <- numeric(4)
mean.lasso.mse <- numeric(length(grid))
nzero <- c()
lambda <- c()
for(j in 1:length(grid)){
  for(i in 1:4){
    X = model.matrix(Return ~ . - 1, data = train_set[[i]])[ ,]
    y = train_set[[i]]$Return
    lasso.mod = glmnet(X, y, alpha = 1, lambda = grid[j])
    nzero = cbind(nzero, lasso.mod$df)
    lambda = cbind(lambda, lasso.mod$lambda)
    newX = model.matrix(Return ~ .- 1, data = test_set[[i]])
    lasso.pred = predict(lasso.mod, newx = newX, s = grid[j])
    lasso.mse[i] = mean( (test_set[[i]]$Return - lasso.pred)^2)
  }
  mean.lasso.mse[j] = mean(lasso.mse)
}
mean.lasso.mse
best.lambda = grid[which(mean.lasso.mse == min(mean.lasso.mse))]
best.lambda
lasso.mod
# let's see how many variables shrunked in each loop
n.l <- rbind(nzero, lambda)
print(n.l)

X.train = model.matrix(Return ~ . - 1, data = train_data)[ ,]
y.train = train_data$Return

lasso.best = glmnet( X.train, y.train, alpha = 1, lambda = 0.1)
lasso_coef = predict( 
  glmnet(X.train, y.train, alpha = 1, lambda = 0.1), 
  type = "coefficients" )
lasso_coef

#test our final model
X.test = model.matrix(Return ~ . - 1, data = test_data)[ ,]
y.test = test_data$Return

final.pred = predict(lasso.best, newx = X.test, s = 0.1)
lasso.test.mse = mean( (test_data$Return - final.pred)^2)
lasso.test.mse

# create a trading strategy
strategy = lm(Return~X1D+X5M+X1Y, data = train_data)
pred_return = predict(strategy, test_data)

# Find investment decisions every day
portfolio = sign(pred_return)
perf = prod( 1 + (portfolio * test_data$Return/100) )
perf

# Find performance from just holding IBM
perf_ibm = prod( 1 + (test_data$Return/100) )
perf_ibm


##### Question 5 (KNN) ############
cuisine <- read.csv('cuisine2.csv',sep=',',header=TRUE)
row.names(cuisine) = cuisine[,1]
cuisine = cuisine[, -1]
my.name = "Qingyuan Dong"
my.cuisine <- cuisine[my.name, ]
my.cuisine

# divide data into train and test
attach(cuisine)
train = subset(cuisine, Section == my.cuisine$Section)
test = subset(cuisine, Section != my.cuisine$Section)

# find 5 closest students to me in my section
dists = as.matrix(dist(train, diag = TRUE) )
close.names = names(sort(dists[my.name, ])[1:7])
close.names
dists[my.name, close.names]
cuisine[close.names, ]

# remove the row "Xinyi Li" (full of NA info)
cuisine = cuisine[-which(rownames(cuisine) %in% "Xinyi Li"), ]
train = train[-which(rownames(train) %in% "Xinyi Li"), ]

# complete the missing rankings with 3NN
k = 4

for(i in 1: nrow(train)){
  closest = sort(as.matrix(dist(train, diag = TRUE))[i,-i])
  train_ordered = train[names(closest), ]
  
  for (j in 1: ncol(train)){
    if(is.na(train[i,j] == FALSE)){
      closest = closest[!is.na(closest)]
      train[i,j] <- mean( closest[1: min(length(closest), k)] )
    }
  }
}

# find the best k

for(k in 1:20){
  
  for(i in 1: nrow(train)){
    k.mse <- 0
    closest = sort(as.matrix(dist(train, diag = TRUE))[i,-i])
    train_ordered = train[names(closest), ]
    preds = c()
    for (j in 1: colnames(train_ordered)){
      closest = train_ordered[ , j]
      closest = closest[!is.na(closest)]
      preds[j] <- mean( closest[1: min(length(closest), k)] )
      non_na_vals = (!is.na(train[i, ]))
      
    }
    k.mse = k.mse + mean(
      as.numeric(train[i, non_na_vals] - preds[non_na_vals])^2) 
  }
  k.rmse[k] = sqrt(k.mse/nrow(train))
}
k.rmse








