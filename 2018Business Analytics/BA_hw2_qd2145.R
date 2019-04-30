rm(list=ls())
setwd("/Users/apple/Desktop/BA_HW02")

##############
# Question 1 #

#Load the data
dogdata= read.csv("dog.csv")


set.seed(4650)
train = sample(1:nrow(dogdata), 0.75*nrow(dogdata))
test = (1:nrow(dogdata)) [ ( (1:nrow(dogdata)) %in% train) == FALSE]
dog.train = dogdata[train, ]
dog.test = dogdata[test, ]
summary(dog.train)
attach(dog.train)

range(tree_score)

profits =  c(c(0,-1),c(0,0.5))
profitPerThreshold = vector("numeric",149)
for (s in 2:149)
{
  dogpred = tree_score > s
  # table to see error rates
  classficationTable = table(predict = dogpred,
                             truth = dog.train$dog )
  profitPerThreshold[s] = sum(classficationTable * profits)
  
}
plot(2:149,profitPerThreshold[2:149],pch = 15, xlab = "Threshold")

best_s = which.max(profitPerThreshold[2:149])+1
dogpred = tree_score > best_s
classficationTable = table(predict = dogpred,
                           truth = dog)
classficationTable
sum(classficationTable * profits)

attach(dogdata)
dogpred = tree_score > best_s
classficationTable = table(predict = dogpred,
                           truth = dog)
classficationTable
sum(classficationTable * profits)

perfect_classifier = c(c(2145,0),c(0,1148))
sum(perfect_classifier * profits)

# fit logistic regression
attach(dog.train)
lgfit = glm(dog ~., data = dog.train, family = binomial)
summary(lgfit)

lgPrediction = predict(lgfit, newdata = dog.train, type = "response")
range(lgPrediction)
threshold = seq(0.09,0.62,0.01)
lgprofit = numeric(54)
for(i in 1:54){
  lgDecision = ifelse(lgPrediction > threshold[i],1,0)
  classficationTable = table(predict = lgDecision,
                             truth = dog.train$dog)
  lgprofit[i] = sum(classficationTable * profits)
}
plot(1:54,lgprofit,pch = 15, xlab = "Threshold")

best_s = which.max(lgprofit)
threshold[best_s]
lgPrediction = predict(lgfit, newdata = dogdata, type = "response")
lgDecision = ifelse(lgPrediction > threshold[best_s],1,0)
classficationTable = table(predict = lgDecision,
                           truth = dogdata$dog)
classficationTable
sum(classficationTable * profits)

# 1j-number of pamphlets need to send in order to get 1000 purchases
1000/(0.05 * mean(lgPrediction))


# fit a decision tree
library(ISLR)
library(tree)

attach(dog.train)
tree.dog = tree(dog ~ ., data = dog.train)
"summary(tree.dog)
plot(tree.dog)
text(tree.dog,pretty=0)
tree.dog"

set.seed(123)
cv.dog = cv.tree(tree.dog)
names(cv.dog) # size is the number of leaves
              # dev is the deviance of the resulting tree
              # k is the tuning parameter
cv.dog

plot(cv.dog$k, cv.dog$size)
par(mfrow=c(1,2))
plot(cv.dog$size,cv.dog$dev,type="b")
plot(cv.dog$k,cv.dog$dev,type="b")

best_size = cv.dog$size[which(cv.dog$dev == min(cv.dog$dev))][1]
prune.dog = prune.misclass(tree.dog, best=best_size)

par(mfrow = c(1, 1))
plot(prune.dog)
text(prune.dog,pretty=0)

attach(dogdata)
lgDecision = ifelse(park_dist < 424.845,1,0)
classficationTable = table(predict = lgDecision,
                           truth = dogdata$dog)
classficationTable
sum(classficationTable * profits)


##############
# question 2 #

# load file
rm(list=ls())
xaltra = read.csv("xaltra.csv")
attach(xaltra)

x = round(998 * True.Positive.Rate..Xaltra.)
y = round(3384 * False.Positive.Rate)
pred.readmission = x+y
pred.readmission

xaltra.cost = numeric(100)
cost.matrix = c(c(6000, 8000), c(1200, 0))
for (i in 1:100){
  confusion.table = c(c(x[i], 998-x[i]), c(y[i], 3384-y[i]))
  xaltra.cost[i] = sum(confusion.table * cost.matrix)
}
best= which(xaltra.cost == min(xaltra.cost))
c(x[best],y[best])
best.pred.adm = x[best]+y[best]
best.pred.adm
min(xaltra.cost)

reduct.cost = 7984000 - min(xaltra.cost)
reduct.cost

xaltra.fee = 250000 +45000*3
xaltra.fee

Xaltra.net.benefit = reduct.cost - xaltra.fee
Xaltra.net.benefit


##############
# question 3 #

# load file
rm(list=ls())
hs = read.csv("hillside_data.csv")
attach(hs)

x = hs$X2010.ST
y = hs$X2011.ST
reg = lm(y ~ x)
summary(reg)
shrink.coef = reg$coefficients[2]
shrink.coef

new.x <- data.frame(
  x = hs$X2011.ST
)
pred.2012 = predict(reg, newdata = new.x)
rmse = sqrt(mean((pred.2012 - hs$X2012.ST)^2))
rmse


##############
# question 4 #

# load file
rm(list=ls())
pt = read.csv("protein.csv", row.names = 1)
attach(pt)

# kmeans clustering
x = cbind(RedMeat, WhiteMeat)
set.seed(123)
kmn = kmeans(x, centers = 3, nstart = 20)
plot(x, col = kmn$cluster + 1 ,pch=10, lwd=3)

# consider all variables
wss = c()
for (i in 1:20)
{
  wss[i] = kmeans(pt, centers = i, nstart = 10)$tot.withinss
}
plot(1:20, wss)

# Five clusters seems fair
km.fit = kmeans(pt, centers = 5, nstart = 20)

# PRint the states in each cluster
for (i in 1:nrow(km.fit$centers))
{
  print(paste("Cluster", i))
  print(names(km.fit$cluster)[km.fit$cluster == i] )
}

# hierarchical clustering
pt = scale(pt)
hc.complete = hclust( dist(pt), method = "complete" )
plot(hc.complete)

# Carry out PCA
pr.out = prcomp(pt, center = TRUE, scale=TRUE)
pr.var = pr.out$sdev^2
pr.var = pr.var / sum(pr.var)

biplot(pr.out)
sum(pr.var[1:2])


# load file
rm(list=ls())

# simulation
N = 10000
x = rnorm(N, mean = 50, sd = 10)
y = numeric(N)
u = runif(N)
for (i in 1:N){
  if (u[i] <= 0.4){
    y[i] = runif(1) * 30 + 20
  }
  else{
    y[i] = runif(1) * 20 + 60
  }
}
d.demand = x+y
hist(d.demand)
quantile(d.demand, probs = c(0.1, 0.9))

ave.profit = numeric(max(d.demand))
d.profit = numeric(N)

# find optimal profit
for (k in 1:max(d.demand)){
  for (i in 1:N){
    if (k > d.demand[i]){
      d.profit[i] = 4 * d.demand[i] - k
    }else{
      d.profit[i] = 3 * k
    }
  }
  ave.profit[k] = mean(d.profit)
}
best.k = which(ave.profit == max(ave.profit))
best.k
ave.profit[best.k]



