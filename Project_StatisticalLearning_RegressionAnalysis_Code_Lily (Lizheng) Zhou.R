rm(list = ls())    # delete objects
cat("\014")
graphics.off()
library(ISLR)
library(randomForest)
library(ggplot2)
library(grid)
library(gridExtra)
library(class)
library(dplyr)
library(glmnet)
library(MASS)
library(rmutil)
library(tictoc)
library(latex2exp)
library(e1071)
library(tidyverse)
library(RColorBrewer)
library(coefplot)

#################
## Preparation ##
#################
# import full datset (n=10000, p=88)
data10000	 =	  read.csv("D:/d/Courses/STA/STA 9890/Project/hedge-fund-x-financial-modeling-challenge/deepanalytics_dataset_10000.csv",header=TRUE)
y          =    data10000$c1
mean(y)
sd(y)
hist(y)

data10000  =    na.omit(data10000)
n          =    dim(data10000)[1]
p          =    dim(data10000)[2]-1

y          =    data10000$c1
mean(y)
sd(y) 
hist(y)

X        =   data.matrix(data10000[,-1])  

apply(X, 2, 'mean')
apply(X, 2, 'sd')

# Scale 
# we can use scale() as well
mu       =   as.vector(apply(X, 2, 'mean'))
sd       =   as.vector(apply(X, 2, 'sd'))
X.orig   =   X
for (i in c(1:n)){
  X[i,]  =   (X[i,] - mu)/sd
}

apply(X, 2, 'mean')
apply(X, 2, 'sd')
#X=X.orig

###########################################################
## 3. fit models 100 times with 4 methods, Ntrain = 0.8n ##
###########################################################
set.seed(1)

n.train           =     floor(0.8*n)
n.test            =     n-n.train

M                 =     100

# Define R-Squred
# lasso
Rsq.train.lasso   =     rep(0,M)
Rsq.test.lasso    =     rep(0,M)  
# en = elastic net
Rsq.train.en      =     rep(0,M)
Rsq.test.en       =     rep(0,M)  
# ridge
Rsq.train.ridge   =     rep(0,M)
Rsq.test.ridge    =     rep(0,M)  
# rf= randomForest
Rsq.train.rf      =     rep(0,M)
Rsq.test.rf       =     rep(0,M)  

for (m in c(1:M)) {
  # record time
  ptm              =     proc.time()
  
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]
  
  # fit lasso and calculate and record the train and test R squares 
  lasso.cv            =     cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10)
  lasso.fit           =     glmnet(X.train, y.train, alpha = 1, lambda = lasso.cv$lambda.min)
  y.train.hat         =     predict(lasso.fit, newx = X.train, type = "response") 
  y.test.hat          =     predict(lasso.fit, newx = X.test, type = "response") 
  # y.train.hat         =     X.train %*% lasso.fit$beta + lasso.fit$a0
  # y.test.hat          =     X.test %*% lasso.fit$beta  + lasso.fit$a0
  Rsq.train.lasso[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
  Rsq.test.lasso[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  
  # fit elastic-net and calculate and record the train and test R squares 
  a=0.5 # elastic-net
  en.cv            =     cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
  en.fit           =     glmnet(X.train, y.train, alpha = a, lambda = en.cv$lambda.min)
  y.train.hat      =     predict(en.fit, newx = X.train, type = "response") 
  y.test.hat       =     predict(en.fit, newx = X.test, type = "response") 
  # y.train.hat      =     X.train %*% en.fit$beta + en.fit$a0
  # y.test.hat       =     X.test %*% en.fit$beta  + en.fit$a0
  Rsq.train.en[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
  Rsq.test.en[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  
  # fit ridge and calculate and record the train and test R squares 
  ridge.cv            =     cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10)
  ridge.fit           =     glmnet(X.train, y.train, alpha = 0, lambda = ridge.cv$lambda.min)
  y.train.hat         =     predict(ridge.fit, newx = X.train, type = "response") 
  y.test.hat          =     predict(ridge.fit, newx = X.test, type = "response") 
  # y.train.hat         =     X.train %*% ridge.fit$beta + ridge.fit$a0
  # y.test.hat          =     X.test %*% ridge.fit$beta  + ridge.fit$a0
  Rsq.train.ridge[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
  Rsq.test.ridge[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  
  # fit RF and calculate and record the train and test R squares 
  rf.fit           =     randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)
  y.train.hat      =     predict(rf.fit, X.train)
  y.test.hat       =     predict(rf.fit, X.test)
  Rsq.train.rf[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
  Rsq.test.rf[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  
  # output time
  ptm    =     proc.time() - ptm
  time   =     ptm["elapsed"]
  
  # print log
  cat(sprintf("m=%3.f| 0.8n | time: %0.3f(sec): 
              Train: lasso=%.3f, en=%.3f, ridge=%.3f, rf=%.3f
              Test:  lasso=%.3f, en=%.3f, ridge=%.3f, rf=%.3f\n",m,time, 
              Rsq.train.lasso[m], Rsq.train.en[m], Rsq.train.ridge[m], Rsq.train.rf[m],
              Rsq.test.lasso[m], Rsq.test.en[m], Rsq.test.ridge[m], Rsq.test.rf[m]))

}

# save files
write.csv(data.frame(Rsq.train.lasso), 
          "D:/d/Courses/STA/STA 9890/Project/Results/Rsq.train.lasso.csv", row.names=FALSE)
write.csv(data.frame(Rsq.train.en), 
          "D:/d/Courses/STA/STA 9890/Project/Results/Rsq.train.en.csv", row.names=FALSE)
write.csv(data.frame(Rsq.train.ridge), 
          "D:/d/Courses/STA/STA 9890/Project/Results/Rsq.train.ridge.csv", row.names=FALSE)
write.csv(data.frame(Rsq.train.rf), 
          "D:/d/Courses/STA/STA 9890/Project/Results/Rsq.train.rf.csv", row.names=FALSE)
write.csv(data.frame(Rsq.test.lasso), 
          "D:/d/Courses/STA/STA 9890/Project/Results/Rsq.test.lasso.csv", row.names=FALSE)
write.csv(data.frame(Rsq.test.en), 
          "D:/d/Courses/STA/STA 9890/Project/Results/Rsq.test.en.csv", row.names=FALSE)
write.csv(data.frame(Rsq.test.ridge), 
          "D:/d/Courses/STA/STA 9890/Project/Results/Rsq.test.ridge.csv", row.names=FALSE)
write.csv(data.frame(Rsq.test.rf), 
          "D:/d/Courses/STA/STA 9890/Project/Results/Rsq.test.rf.csv", row.names=FALSE)

#######################################################################
## 4.(b) side-by-side boxplots of R-squared train and R-squared test ##
#######################################################################
Rsq.df = data.frame(c(rep("train", 4*M), rep("test", 4*M)), 
                    
                    c(rep("Lasso",M),rep("Elastic-net",M), 
                      rep("Ridge",M),rep("Random Forest",M), 
                      rep("Lasso",M),rep("Elastic-net",M), 
                      rep("Ridge",M),rep("Random Forest",M)), 
                    
                    c(Rsq.train.lasso, Rsq.train.en, Rsq.train.ridge, Rsq.train.rf, 
                      Rsq.test.lasso, Rsq.test.en, Rsq.test.ridge, Rsq.test.rf))

colnames(Rsq.df) =  c("type", "method", "R_Squared")
Rsq.df

write.csv(data.frame(Rsq.df), 
          "D:/d/Courses/STA/STA 9890/Project/Results/Rsq.df.csv", row.names=FALSE)

# Rsq.df = read.csv("D:/d/Courses/STA/STA 9890/Project/Results/Rsq.df.csv",header=TRUE)
# Rsq.df

# change the order of factor levels
Rsq.df$method       = factor(Rsq.df$method, 
                             levels=c("Lasso", "Elastic-net", "Ridge", "Random Forest"))
Rsq.df$type         = factor(Rsq.df$type, levels=c("train", "test"))

Rsq.df.boxplot = ggplot(Rsq.df) + 
  aes(x=method, y=R_Squared, fill=method) + 
  geom_boxplot() + 
  facet_wrap(~ type, ncol=2) + 
  ggtitle("Boxplots of R-Squared Train and Test with 4 Methods (train size = 0.8n, 100 samples)") + 
  ylim(0, 1)
Rsq.df.boxplot 

Rsq.df.boxplot2 = ggplot(Rsq.df) +
  aes(x=type, y=R_Squared, fill=type) +
  geom_boxplot() +
  facet_wrap(~ method, ncol=4) +
  ggtitle("Boxplots of R-Squared Train and Test with 4 methods (train size = 0.8n, 100 samples)") +
  ylim(0, 1)
Rsq.df.boxplot2

############################################
## 4.(c)(d)(f) For one on the 100 samples ##
############################################
# sampling: 
# For one on the 100 samples 
set.seed(1)

n.train           =     floor(0.8*n)
n.test            =     n-n.train

M                 =     100

shuffled_indexes_ =     sample(n)
train             =     shuffled_indexes_[1:n.train]
test              =     shuffled_indexes_[(1+n.train):n]
X.train           =     X[train, ]
y.train           =     y[train]
X.test            =     X[test, ]
y.test            =     y[test]

#########################################
## 4.(c)(f) For one on the 100 samples ##
#########################################
# Lasso
# record time
ptm                 =     proc.time()

lasso.cv            =     cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10)
lasso.fit           =     glmnet(X.train, y.train, alpha = 1, lambda = lasso.cv$lambda.min)

# output time
ptm          =     proc.time() - ptm
time.lasso_  =     ptm["elapsed"]

y.train.hat      =     predict(lasso.fit, newx = X.train, type = "response") 
y.test.hat       =     predict(lasso.fit, newx = X.test, type = "response") 
# y.train.hat      =     X.train %*% lasso.fit$beta + lasso.fit$a0
# y.test.hat       =     X.test %*% lasso.fit$beta  + lasso.fit$a0
Rsq.train.lasso_ =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
Rsq.test.lasso_  =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)

# Elastic-net
# record time
ptm              =     proc.time()

a=0.5 # elastic-net
en.cv            =     cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
en.fit           =     glmnet(X.train, y.train, alpha = a, lambda = en.cv$lambda.min)

# output time
ptm       =     proc.time() - ptm
time.en_  =     ptm["elapsed"]

y.train.hat      =     predict(en.fit, newx = X.train, type = "response") 
y.test.hat       =     predict(en.fit, newx = X.test, type = "response") 
# y.train.hat      =     X.train %*% en.fit$beta + en.fit$a0
# y.test.hat       =     X.test %*% en.fit$beta  + en.fit$a0
Rsq.train.en_    =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
Rsq.test.en_     =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)

# Ridge
# record time
ptm                 =     proc.time()

ridge.cv            =     cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10)
ridge.fit           =     glmnet(X.train, y.train, alpha = 0, lambda = ridge.cv$lambda.min)

# output time
ptm          =     proc.time() - ptm
time.ridge_  =     ptm["elapsed"]

y.train.hat         =     predict(ridge.fit, newx = X.train, type = "response") 
y.test.hat          =     predict(ridge.fit, newx = X.test, type = "response") 
# y.train.hat         =     X.train %*% ridge.fit$beta + ridge.fit$a0
# y.test.hat          =     X.test %*% ridge.fit$beta  + ridge.fit$a0
Rsq.train.ridge_    =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
Rsq.test.ridge_     =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)

# RF
# record time
ptm       =     proc.time()

rf.fit    =     randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)

# output time
ptm       =     proc.time() - ptm
time.rf_  =     ptm["elapsed"]

y.train.hat      =     predict(rf.fit, X.train)
y.test.hat       =     predict(rf.fit, X.test)
Rsq.train.rf_    =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
Rsq.test.rf_     =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)

####################################################################
## 4.(c) For one on the 100 samples, create 10-fold CV curves for ##
## lasso, elastic-net (alpha=0.5), ridge                          ##
####################################################################
par(mfrow=c(3,1))
plot(lasso.cv)
title("10-fold CV curve for lasso", line = 3)
plot(en.cv)
title("10-fold CV curve for elastic-net (alpha = 0.5)", line = 3)
plot(ridge.cv)
title("10-fold CV curve for ridge", line = 3)

par(mfrow=c(1,1))
plot(lasso.cv)
title("10-fold CV curve for lasso", line = 3)
plot(en.cv)
title("10-fold CV curve for elastic-net (alpha = 0.5)", line = 3)
plot(ridge.cv)
title("10-fold CV curve for ridge", line = 3)

################################
## 4.(f) Performance vs. Time ##
################################
cat(sprintf("Performance vs. Fitting Time (0.8n): 
            lasso: fitting time=%.3f, performance=%.3f
            en:    fitting time=%.3f, performance=%.3f
            ridge: fitting time=%.3f, performance=%.3f
            rf:    fitting time=%.3f, performance=%.3f\n", 
            time.lasso_, Rsq.test.lasso_, time.en_, Rsq.test.en_, 
            time.ridge_, Rsq.test.ridge_, time.rf_, Rsq.test.rf_))

# create performance vs. fitting time data.frame
pt.df = data.frame(c("Lasso", "EN", "Ridge", "RF"), 
                   c(time.lasso_, time.en_, time.ridge_, time.rf_), 
                   c(Rsq.test.lasso_, Rsq.test.en_, Rsq.test.ridge_, Rsq.test.rf_))

colnames(pt.df) = c("model", "fitting_time", "test_rsquared")
pt.df

write.csv(pt.df, 
          "D:/d/Courses/STA/STA 9890/Project/Results/performance_vs_time.csv", 
          row.names=FALSE)

# pt.df = read.csv("D:/d/Courses/STA/STA 9890/Project/Results/performance_vs_time.csv",header=TRUE)
# colnames(pt.df) = c("model", "fitting_time", "test_rsquared")
# pt.df

ggplot(pt.df, aes(x=fitting_time, y=test_rsquared, color=model)) +
  geom_point(size=5, alpha=0.8) + 
  labs(x="Fitting Time (sec)") + 
  labs(y="Performance (R-Squared Test)") + 
  ylim(0, 1) + 
  ggtitle("Performance vs. Fitting Time (0.8n)")

#################################################################
## 4.(d) the side-by-side boxplots of train and test residuals ##
#################################################################
set.seed(1)

# cross validation for lasso
cat("processing cross-validation for lasso:\n")
lasso.cv               =     cv.glmnet(X.train, y.train, alpha = 1, family = "gaussian",intercept = T, type.measure = "mae", nfolds = 10)
lasso.fit              =     glmnet(X.train, y.train, alpha = 1, family = "gaussian", intercept = T, lambda = lasso.cv$lambda.min)
y.train.hat.lasso      =     X.train %*% lasso.fit$beta + lasso.fit$a0  #same as: y.train.hat_ =    predict(lasso.fit, newx = X.train, type = "response", lasso.cv$lambda.min)
y.test.hat.lasso       =     X.test %*% lasso.fit$beta  + lasso.fit$a0  #same as: y.test.hat_  =    predict(lasso.fit, newx = X.test, type = "response", lasso.cv$lambda.min)
y.train.hat.lasso      =     as.vector(y.train.hat.lasso)
y.test.hat.lasso       =     as.vector(y.test.hat.lasso)

res.df.lasso           =     data.frame(c(rep("train", n.train),rep("test", n.test)), 
                                        c(1:n),
                                        c(y.train.hat.lasso - y.train, y.test.hat.lasso - y.test))
colnames(res.df.lasso) =     c("type", "NO.", "residual")

# res.df.lasso.barplot   =     ggplot(res.df.lasso, aes(x=type, y=residual, fill=type)) + 
#   geom_boxplot(outlier.size = 0.1) + 
#   ggtitle("LASSO") + 
#   theme(legend.position="bottom")
# res.df.lasso.barplot

# cross validation for elastic-net
cat("processing cross-validation for elastic-net (alpha = 0.5):\n")
a = 0.5
en.cv               =     cv.glmnet(X.train, y.train, alpha = a, family = "gaussian",intercept = T, type.measure = "mae", nfolds = 10)
en.fit              =     glmnet(X.train, y.train, alpha = a, family = "gaussian", intercept = T, lambda = en.cv$lambda.min)
y.train.hat.en      =     X.train %*% en.fit$beta + en.fit$a0  #same as: y.train.hat_ =    predict(en.fit, newx = X.train, type = "response", en.cv$lambda.min)
y.test.hat.en       =     X.test %*% en.fit$beta  + en.fit$a0  #same as: y.test.hat_  =    predict(en.fit, newx = X.test, type = "response", en.cv$lambda.min)
y.train.hat.en      =     as.vector(y.train.hat.en)
y.test.hat.en       =     as.vector(y.test.hat.en)

res.df.en           =     data.frame(c(rep("train", n.train),rep("test", n.test)), 
                                     c(1:n),
                                     c(y.train.hat.en - y.train, y.test.hat.en - y.test))
colnames(res.df.en) =     c("type", "NO.", "residual")

# res.df.en.barplot   =     ggplot(res.df.en, aes(x=type, y=residual, fill=type)) + 
#   geom_boxplot(outlier.size = 0.1) + 
#   ggtitle("Elastic-net (alpha = 0.5)") + 
#   theme(legend.position="bottom")
# res.df.en.barplot

# cross validation for ridge
cat("processing cross-validation for ridge:\n")
ridge.cv               =     cv.glmnet(X.train, y.train, alpha = 0, family = "gaussian",intercept = T, type.measure = "mae", nfolds = 10)
ridge.fit              =     glmnet(X.train, y.train, alpha = 0, family = "gaussian", intercept = T, lambda = ridge.cv$lambda.min)
y.train.hat.ridge      =     X.train %*% ridge.fit$beta + ridge.fit$a0  #same as: y.train.hat_  =    predict(ridge.fit, newx = X.train, type = "response", ridge.cv$lambda.min)
y.test.hat.ridge       =     X.test %*% ridge.fit$beta  + ridge.fit$a0  #same as: y.test.hat_  =    predict(ridge.fit, newx = X.test, type = "response", ridge.cv$lambda.min)
y.train.hat.ridge      =     as.vector(y.train.hat.ridge)
y.test.hat.ridge       =     as.vector(y.test.hat.ridge)

res.df.ridge           =     data.frame(c(rep("train", n.train),rep("test", n.test)), 
                                        c(1:n),
                                        c(y.train.hat.ridge - y.train, y.test.hat.ridge - y.test))
colnames(res.df.ridge) =     c("type", "NO.", "residual")

# res.df.ridge.barplot   =     ggplot(res.df.ridge, aes(x=type, y=residual, fill=type)) + 
#   geom_boxplot(outlier.size = 0.1) + 
#   ggtitle("Ridge") + 
#   theme(legend.position="bottom")
# res.df.ridge.barplot

# rf
cat("processing rf:\n")
rf.fit              =     randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)
y.train.hat.rf      =     predict(rf.fit, X.train)
y.test.hat.rf       =     predict(rf.fit, X.test)
y.train.hat.rf      =     as.vector(y.train.hat.rf)
y.test.hat.rf       =     as.vector(y.test.hat.rf)

res.df.rf           =     data.frame(c(rep("train", n.train),rep("test", n.test)), 
                                     c(1:n),
                                     c(y.train.hat.rf - y.train, y.test.hat.rf - y.test))
colnames(res.df.rf) =     c("type", "NO.", "residual")

# res.df.rf.barplot   =     ggplot(res.df.rf, aes(x=type, y=residual, fill=type)) + 
#   geom_boxplot(outlier.size = 0.1) + 
#   ggtitle("Random Forest") + 
#   theme(legend.position="bottom")
# res.df.rf.barplot


# build giant residual data frame for better plot
res.df              =     data.frame(c(rep("Lasso",n), rep("Elastic-net",n), rep("Ridge",n), rep("Random Forest",n)),
                                     rbind(res.df.lasso, res.df.en, res.df.ridge, res.df.rf))
colnames(res.df)    =     c("method", "type", "NO.", "residual")

# change the order of factor levels
res.df$method       = factor(res.df$method, 
                             levels=c("Lasso", "Elastic-net", "Ridge", "Random Forest"))
res.df$type         = factor(res.df$type, levels=c("train", "test"))

# res.df.barplot      =     ggplot(res.df, aes(x=method, y=residual, fill=type)) +
#   geom_boxplot(outlier.size = 0.1) +
#   ggtitle("Boxplots of Train and Test Residuals with 4 methods")
#   # + theme(legend.position="bottom")
# res.df.barplot

res.df.barplot = ggplot(res.df) +
  aes(x=type, y=residual, fill=type) +
  geom_boxplot() +
  facet_wrap(~ method, ncol=4) +
  ggtitle("Boxplots of Train and Test Residuals with 4 Methods)")
res.df.barplot

res.df.barplot2 = ggplot(res.df) + 
  aes(x=method, y=residual, fill=method) + 
  geom_boxplot() + 
  facet_wrap(~ type, ncol=2) + 
  ggtitle("Boxplots of Train and Test Residuals with 4 Methods)") 
res.df.barplot2 

# grid.arrange(res.df.lasso.barplot, res.df.en.barplot, res.df.ridge.barplot, res.df.rf.barplot)
# grid.arrange(res.df.lasso.barplot, res.df.en.barplot, res.df.ridge.barplot, res.df.rf.barplot, nrow=1,
#              top = textGrob("Boxplots of Train and Test Residuals",gp=gpar(fontsize=20,font=3)))

#################################################################
## 4.(e) Bar-plots (with bootstrapped error bars) of the       ##
## estimated coefficients, andthe importance of the parameters ##
#################################################################
set.seed(1)

bootstrapSamples   =     100
beta.rf.bs         =     matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.lasso.bs      =     matrix(0, nrow = p, ncol = bootstrapSamples)         
beta.en.bs         =     matrix(0, nrow = p, ncol = bootstrapSamples)         
beta.ridge.bs      =     matrix(0, nrow = p, ncol = bootstrapSamples)         

for (m in 1:bootstrapSamples){
  # record time
  ptm              =     proc.time()
  
  bs_indexes       =     sample(n, replace=T)
  X.bs             =     X[bs_indexes, ]
  y.bs             =     y[bs_indexes]
  
  # model 1: random forest
  # fit bs rf
  rf                  =     randomForest(X.bs, y.bs, mtry = sqrt(p), importance = TRUE)
  beta.rf.bs[,m]      =     as.vector(rf$importance[,1])
  # model 2: lasso  alpha = 1
  # fit bs lasso
  a                   =     1   # lasso
  cv.fit              =     cv.glmnet(X.bs, y.bs, alpha = a, nfolds = 10)
  fit                 =     glmnet(X.bs, y.bs, alpha = a, lambda = cv.fit$lambda.min)  
  beta.lasso.bs[,m]   =     as.vector(fit$beta)
  # model 3: elastic-net alpha = 0.5
  # fit bs elastic-net
  a                   =     0.5 # elastic-net
  cv.fit              =     cv.glmnet(X.bs, y.bs, alpha = a, nfolds = 10)
  fit                 =     glmnet(X.bs, y.bs, alpha = a, lambda = cv.fit$lambda.min)  
  beta.en.bs[,m]      =     as.vector(fit$beta)
  # model 4: ridge  alpha = 0 
  # fit bs ridge
  a                   =     0   # ridge
  cv.fit              =     cv.glmnet(X.bs, y.bs, alpha = a, nfolds = 10)
  fit                 =     glmnet(X.bs, y.bs, alpha = a, lambda = cv.fit$lambda.min)  
  beta.ridge.bs[,m]   =     as.vector(fit$beta)
  
  # output time
  ptm    =     proc.time() - ptm
  time   =     ptm["elapsed"]
  
  # print bootstrap log and time
  cat(sprintf("Bootstrap Sample: %3.f | time: %0.3f(sec)\n", m, time))
}

# save files
write.csv(data.frame(beta.rf.bs), 
          "D:/d/Courses/STA/STA 9890/Project/Results/beta.rf.bs.csv", row.names=FALSE)
write.csv(data.frame(beta.lasso.bs), 
          "D:/d/Courses/STA/STA 9890/Project/Results/beta.lasso.bs.csv", row.names=FALSE)
write.csv(data.frame(beta.en.bs), 
          "D:/d/Courses/STA/STA 9890/Project/Results/beta.en.bs.csv", row.names=FALSE)
write.csv(data.frame(beta.ridge.bs), 
          "D:/d/Courses/STA/STA 9890/Project/Results/beta.ridge.bs.csv", row.names=FALSE)


# calculate bootstrapped standard errors / alternatively you could use qunatiles to find upper and lower bounds
rf.bs.sd    = apply(beta.rf.bs, 1, "sd")
lasso.bs.sd = apply(beta.lasso.bs, 1, "sd")
en.bs.sd    = apply(beta.en.bs, 1, "sd")
ridge.bs.sd = apply(beta.ridge.bs, 1, "sd")


set.seed(1)
# fit rf to the whole data
rf                     =     randomForest(X, y, mtry = sqrt(p), importance = TRUE)

betaS.rf               =     data.frame(as.character(c(1:p)), as.vector(rf$importance[,1]), 2*rf.bs.sd)
colnames(betaS.rf)     =     c( "feature", "value", "err")

# fit lasso to the whole data
a=1   # lasso
cv.fit                 =     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit                    =     glmnet(X, y, alpha = a, lambda = cv.fit$lambda.min)

betaS.lasso            =     data.frame(as.character(c(1:p)), as.vector(fit$beta), 2*lasso.bs.sd)
colnames(betaS.lasso)  =     c( "feature", "value", "err")

# fit en to the whole data
a=0.5 # elastic-net
cv.fit                 =     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit                    =     glmnet(X, y, alpha = a, lambda = cv.fit$lambda.min)

betaS.en               =     data.frame(as.character(c(1:p)), as.vector(fit$beta), 2*en.bs.sd)
colnames(betaS.en)     =     c( "feature", "value", "err")

# fit ridge to the whole data
a=0   # ridge
cv.fit                 =     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit                    =     glmnet(X, y, alpha = a, lambda = cv.fit$lambda.min)

betaS.ridge            =     data.frame(as.character(c(1:p)), as.vector(fit$beta), 2*ridge.bs.sd)
colnames(betaS.ridge)  =     c( "feature", "value", "err")

# save files
write.csv(betaS.rf, 
          "D:/d/Courses/STA/STA 9890/Project/Results/betaS.rf.csv", row.names=FALSE)
write.csv(betaS.lasso, 
          "D:/d/Courses/STA/STA 9890/Project/Results/betaS.lasso.csv", row.names=FALSE)
write.csv(betaS.en, 
          "D:/d/Courses/STA/STA 9890/Project/Results/betaS.en.csv", row.names=FALSE)
write.csv(betaS.ridge, 
          "D:/d/Courses/STA/STA 9890/Project/Results/betaS.ridge.csv", row.names=FALSE)


rfPlot    =  ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.4)

lassoPlot =  ggplot(betaS.lasso, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.4)

enPlot    =  ggplot(betaS.en, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.4)

ridgePlot =  ggplot(betaS.ridge, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.4)

grid.arrange(rfPlot, lassoPlot, enPlot, ridgePlot, nrow = 4)

# we need to change the order of factor levels by specifying the order explicitly.
betaS.rf$feature     =  factor(betaS.rf$feature,    levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.lasso$feature  =  factor(betaS.lasso$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.en$feature     =  factor(betaS.en$feature,    levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.ridge$feature  =  factor(betaS.ridge$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])

rfPlot    =  ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="#619CFF", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.4) + 
  ggtitle("Importance of Parameters - RF") + 
  theme(plot.title  = element_text(size = 10, face = "bold"), 
        axis.text.x = element_text(size=7)) 

lassoPlot =  ggplot(betaS.lasso, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="#F8766D", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.4) + 
  ggtitle("Estimated Coefficients - Lasso") + 
  theme(plot.title  = element_text(size = 10, face = "bold"), 
        axis.text.x = element_text(size=7)) +
  ylim(-0.76, 0.76)  

enPlot    =  ggplot(betaS.en, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="#F8766D", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.4) + 
  ggtitle("Estimated Coefficients - Elastic-net (alpha = 0.5)") + 
  theme(plot.title  = element_text(size = 10, face = "bold"), 
        axis.text.x = element_text(size=7)) +
  ylim(-0.76, 0.76)  

ridgePlot =  ggplot(betaS.ridge, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="#F8766D", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.4) + 
  ggtitle("Estimated Coefficients - Ridge") + 
  theme(plot.title  = element_text(size = 10, face = "bold"), 
        axis.text.x = element_text(size=7)) +
  ylim(-0.76, 0.76)  

grid.arrange(rfPlot, lassoPlot, enPlot, ridgePlot, nrow = 4)


# Another Plot
betaS =     data.frame(c(rep("Random Forest",p), rep("Lasso",p), rep("Elastic-net (alpha = 0.5)",p), rep("Ridge",p)),
                       rbind(betaS.rf,betaS.lasso,betaS.en,betaS.ridge))
colnames(betaS)     =     c( "method", "feature", "value", "err")

# we need to change the order of factor levels by specifying the order explicitly.
betaS$method   =  factor(betaS$method, levels=c("Random Forest", "Lasso", "Elastic-net (alpha = 0.5)", "Ridge"))
betaS$feature  =  factor(betaS$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])

betaS.Plot    =  ggplot(betaS, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=0.1) +
  facet_wrap(~ method, nrow=4) +
  ggtitle("Bar-plots of Estimated Coefficients (Lasso, EN, Ridge) and Importance of Parameters (RF)")
betaS.Plot
  




