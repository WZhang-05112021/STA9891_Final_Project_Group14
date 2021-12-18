#STA9891 Final Group 14

rm(list=ls())
cat("\014")
graphics.off()

# install.packages("dplyr")
# install.packages("gridExtra")
# install.packages("glmnet")
# install.packages("caret")
# install.packages("tictoc")
# install.packages("tree")
# install.packages("randomForest")
# install.packages("svMisc")

library(ggplot2)
library(gridExtra)
library(glmnet)
library(caret)
library(readr)
library(tictoc)
library(MASS)
library(tree)
library(randomForest)
library(httr)
library(dplyr)
library(svMisc)
library(pROC)

# R code to load the clean data file
df <- read.csv("C:/Users/xzhang/Desktop/9891_Final/Data/telecom_churn_data_clean.csv")

df<- as.data.frame(df)

dim(df)

# Count positive and negative observations
df %>% count(churn)

set.seed(1)

dim(df)

n = dim(df)[1]
p = ncol(subset( df, select = -c(churn) ))

X <- data.matrix(subset( df, select = -c(churn) ))
y <- data.matrix(df$churn)

n.train        =     floor(0.9*n)
n.test         =     n-n.train

# 50 times sampling and CV for EN, Lasso and Ridge

M              =     50
auc.test.en    =     rep(0,M)  
auc.train.en   =     rep(0,M)
runtime.en     =     rep(0,M)

auc.test.ls    =     rep(0,M) 
auc.train.ls   =     rep(0,M)
runtime.ls     =     rep(0,M)

auc.test.rid   =     rep(0,M)
auc.train.rid  =     rep(0,M)
runtime.rid    =     rep(0,M)

#Q3(b) 

# record run time - start
tic('50-sample')
for (m in c(1:M)) {
  
  # progress bar
  progress(m, progress.bar = TRUE)
  Sys.sleep(0.01)
  if (m == M) cat("Done!\n")

  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train,]
  y.train          =     y[train]
  X.test           =     X[test,]
  y.test           =     y[test]
  
  
  n.P                 =        sum(y.train)
  n.N                 =        n.train - n.P
  ww                  =        rep(1,n.train)
  ww[y.train==1]      =        n.N/n.P
  
  # Elastic Net
  a=0.5 # elastic-net 0<a<1
  
  start_time <- Sys.time() # record start time
  cv.fiten         =     cv.glmnet(X.train, y.train, family = "binomial",intercept = TRUE, alpha = a, nfolds = 10, weights = ww, type.measure = "auc")
  end_time <- Sys.time() # record end time
  runtime.en[m] = end_time - start_time
  
  fit              =     glmnet(X.train, y.train,family = "binomial",intercept = TRUE, alpha = a, lambda = cv.fiten$lambda.min, weights = ww)
  
  beta0.hat               =        fit$a0
  beta.hat                =        as.vector(fit$beta)
  prob.train              =        exp(X.train %*% beta.hat +  beta0.hat  )/(1 + exp(X.train %*% beta.hat +  beta0.hat  ))
  prob.test               =        exp(X.test %*% beta.hat +  beta0.hat  )/(1 + exp(X.test %*% beta.hat +  beta0.hat  ))
  dt                      =        0.01
  thta                    =        1-seq(0,1, by=dt)
  thta.length             =        length(thta)
  FPR.train               =        matrix(0, thta.length)
  TPR.train               =        matrix(0, thta.length)
  FPR.test                =        matrix(0, thta.length)
  TPR.test                =        matrix(0, thta.length)
  
  for (i in c(1:thta.length)){
    # calculate the FPR and TPR for train data 
    y.hat.train             =        ifelse(prob.train > thta[i], 1, 0) #table(y.hat.train, y.train)
    FP.train                =        sum(y.train[y.hat.train==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.train                =        sum(y.hat.train[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
    P.train                 =        sum(y.train==1) # total positives in the data
    N.train                 =        sum(y.train==0) # total negatives in the data
    FPR.train[i]            =        FP.train/N.train # false positive rate = type 1 error = 1 - specificity
    TPR.train[i]            =        TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity
    
    # calculate the FPR and TPR for test data 
    y.hat.test              =        ifelse(prob.test > thta[i], 1, 0)
    FP.test                 =        sum(y.test[y.hat.test==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.test                 =        sum(y.hat.test[y.test==1] == 1) # true positives = positives in the data that were predicted as positive
    P.test                  =        sum(y.test==1) # total positives in the data
    N.test                  =        sum(y.test==0) # total negatives in the data
    FPR.test[i]             =        FP.test/N.test # false positive rate = type 1 error = 1 - specificity
    TPR.test[i]             =        TP.test/P.test # true positive rate = 1 - type 2 error = sensitivity    
    # print(paste("K=", K, " ki=",ki, ", K-fold CV=", Kfold.CV.err[i]))
  }
  
  #auc.train = auc(FPR.train, TPR.train)
  auc.test.en[m]       =       sum((TPR.train[1:(thta.length-1)] + 0.5 * diff(TPR.train)) * diff(FPR.train))
  auc.train.en[m]      =       sum((TPR.test[1:(thta.length-1)] + 0.5 * diff(TPR.test)) * diff(FPR.test))
  
  
  # Lasso
  a=1 # lasso
  
  start_time <- Sys.time() # record start time
  cv.fitls         =     cv.glmnet(X.train, y.train, family = "binomial",intercept = TRUE, alpha = a, nfolds = 10, weights = ww, type.measure = "auc")
  end_time <- Sys.time() # record end time
  runtime.ls[m] = end_time - start_time
  
  fit              =     glmnet(X.train, y.train,family = "binomial",intercept = TRUE, alpha = a, lambda = cv.fitls$lambda.min, weights = ww)
  
  beta0.hat               =        fit$a0
  beta.hat                =        as.vector(fit$beta)
  prob.train              =        exp(X.train %*% beta.hat +  beta0.hat  )/(1 + exp(X.train %*% beta.hat +  beta0.hat  ))
  prob.test               =        exp(X.test %*% beta.hat +  beta0.hat  )/(1 + exp(X.test %*% beta.hat +  beta0.hat  ))
  dt                      =        0.01
  thta                    =        1-seq(0,1, by=dt)
  thta.length             =        length(thta)
  FPR.train               =        matrix(0, thta.length)
  TPR.train               =        matrix(0, thta.length)
  FPR.test                =        matrix(0, thta.length)
  TPR.test                =        matrix(0, thta.length)
  
  for (i in c(1:thta.length)){
    # calculate the FPR and TPR for train data 
    y.hat.train             =        ifelse(prob.train > thta[i], 1, 0) #table(y.hat.train, y.train)
    FP.train                =        sum(y.train[y.hat.train==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.train                =        sum(y.hat.train[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
    P.train                 =        sum(y.train==1) # total positives in the data
    N.train                 =        sum(y.train==0) # total negatives in the data
    FPR.train[i]            =        FP.train/N.train # false positive rate = type 1 error = 1 - specificity
    TPR.train[i]            =        TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity
    
    # calculate the FPR and TPR for test data 
    y.hat.test              =        ifelse(prob.test > thta[i], 1, 0)
    FP.test                 =        sum(y.test[y.hat.test==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.test                 =        sum(y.hat.test[y.test==1] == 1) # true positives = positives in the data that were predicted as positive
    P.test                  =        sum(y.test==1) # total positives in the data
    N.test                  =        sum(y.test==0) # total negatives in the data
    FPR.test[i]             =        FP.test/N.test # false positive rate = type 1 error = 1 - specificity
    TPR.test[i]             =        TP.test/P.test # true positive rate = 1 - type 2 error = sensitivity    
    # print(paste("K=", K, " ki=",ki, ", K-fold CV=", Kfold.CV.err[i]))
  }
  
  #auc.train = auc(FPR.train, TPR.train)
  auc.test.ls[m]       =       sum((TPR.train[1:(thta.length-1)] + 0.5 * diff(TPR.train)) * diff(FPR.train))
  auc.train.ls[m]      =       sum((TPR.test[1:(thta.length-1)] + 0.5 * diff(TPR.test)) * diff(FPR.test))
  
  # Ridge
  a=0 # ridge
  
  start_time <- Sys.time() # record start time
  cv.fitrid         =     cv.glmnet(X.train, y.train, family = "binomial",intercept = TRUE, alpha = a, nfolds = 10, weights = ww, type.measure = "auc")
  end_time <- Sys.time() # record end time
  runtime.rid[m] = end_time - start_time
  
  fit              =     glmnet(X.train, y.train,family = "binomial",intercept = TRUE, alpha = a, lambda = cv.fitrid$lambda.min, weights = ww)
  
  beta0.hat               =        fit$a0
  beta.hat                =        as.vector(fit$beta)
  prob.train              =        exp(X.train %*% beta.hat +  beta0.hat  )/(1 + exp(X.train %*% beta.hat +  beta0.hat  ))
  prob.test               =        exp(X.test %*% beta.hat +  beta0.hat  )/(1 + exp(X.test %*% beta.hat +  beta0.hat  ))
  dt                      =        0.01
  thta                    =        1-seq(0,1, by=dt)
  thta.length             =        length(thta)
  FPR.train               =        matrix(0, thta.length)
  TPR.train               =        matrix(0, thta.length)
  FPR.test                =        matrix(0, thta.length)
  TPR.test                =        matrix(0, thta.length)
  
  
  for (i in c(1:thta.length)){
    # calculate the FPR and TPR for train data 
    y.hat.train             =        ifelse(prob.train > thta[i], 1, 0) #table(y.hat.train, y.train)
    FP.train                =        sum(y.train[y.hat.train==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.train                =        sum(y.hat.train[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
    P.train                 =        sum(y.train==1) # total positives in the data
    N.train                 =        sum(y.train==0) # total negatives in the data
    FPR.train[i]            =        FP.train/N.train # false positive rate = type 1 error = 1 - specificity
    TPR.train[i]            =        TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity
    
    # calculate the FPR and TPR for test data 
    y.hat.test              =        ifelse(prob.test > thta[i], 1, 0)
    FP.test                 =        sum(y.test[y.hat.test==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.test                 =        sum(y.hat.test[y.test==1] == 1) # true positives = positives in the data that were predicted as positive
    P.test                  =        sum(y.test==1) # total positives in the data
    N.test                  =        sum(y.test==0) # total negatives in the data
    FPR.test[i]             =        FP.test/N.test # false positive rate = type 1 error = 1 - specificity
    TPR.test[i]             =        TP.test/P.test # true positive rate = 1 - type 2 error = sensitivity    
    # print(paste("K=", K, " ki=",ki, ", K-fold CV=", Kfold.CV.err[i]))
  }
  
  #auc.train = auc(FPR.train, TPR.train)
  auc.test.rid[m]       =       sum((TPR.train[1:(thta.length-1)] + 0.5 * diff(TPR.train)) * diff(FPR.train))
  auc.train.rid[m]      =       sum((TPR.test[1:(thta.length-1)] + 0.5 * diff(TPR.test)) * diff(FPR.test))
  
}

# record run time - end
toc()


# random forest train test auc
# random forest train test auc

M             =     50

auc.test.rf   =     rep(0,M)
auc.train.rf  =     rep(0,M)
runtime.rf    =     rep(0,M)


tic('50-sample for Random Forest')

for (m in c(1:M)) {
  
  # progress bar
  progress(m, progress.bar = TRUE)
  Sys.sleep(0.01)
  if (m == M) cat("Done!\n")
  
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     as.matrix(X[train,])
  y.train          =     as.factor(y[train,])
  X.test           =     as.matrix(X[test,])
  y.test           =     as.factor(y[test,])
  d_train          =     data.frame(x=X.train, y=y.train)
  d_test           =     data.frame(x=X.test, y=y.test)
  
  # Random Forest
  start_time <- Sys.time() # record start time
  rf.fit              =    randomForest(y~., data = d_train , mtry = sqrt(p))
  end_time <- Sys.time() # record end time
  runtime.rf[m]       =    end_time - start_time
  
  #AUC for train 
  roc_train           =    roc(rf.fit$predicted, as.numeric(y.train))
  auc.train.rf[m]     =    auc(roc_train)
  
  #AUC for test
  y.hat               =    predict(rf.fit, d_test)
  test_roc            =    roc(y.hat, as.numeric(y.test))
  auc.test.rf[m]     =     auc(test_roc)
}  

toc()  

# Train Test AUC Summary
auc.test.en
auc.train.en
runtime.en

auc.test.ls 
auc.train.ls
runtime.ls

auc.test.rid
auc.train.rid
runtime.rid

auc.test.rf
auc.train.rf
runtime.rf


# Q3(b) Boxplots of the 50 AUCs (train and test) for ntrain = 0:9n. Specically show two plots, one for test AUCs and train AUCs, respectively. Make sure everything is clearly visible and legible.

par(mfrow = c(1, 2))

# train r^2 box plot
boxplot(auc.train.en, auc.train.ls, auc.train.rid, auc.train.rf, main="Train AUC", ylab="AUC", names = c("Elastic Net","Lasso", "Ridge", "Random Forest"), col = terrain.colors(4))

# test r^2 box plot
boxplot(auc.test.en, auc.test.ls, auc.test.rid, auc.test.rf, main="Test AUC", ylab="AUC", names = c("Elastic Net","Lasso", "Ridge", "Random Forest"), col = terrain.colors(4) )

# 3 (c) one of the 50 samples

# take the last sample in the 50 loops

# Cross validation curve
# stack the three plots
par(mfrow = c(3, 1))

plot(cv.fiten,  sub = "Elastic Net")
plot(cv.fitls,  sub = "Lasso")
plot(cv.fitrid, sub = 'Ridge')

# Runtime

runtime.en_51  <-tail(runtime.en , n = 1)
runtime.ls_51  <-tail(runtime.ls , n = 1)
runtime.rid_51 <-tail(runtime.rid , n = 1)

oneof50_runtime <- rbind(runtime.en_51 ,   runtime.ls_51,   runtime.rid_51)
colnames(oneof50_runtime) <- "Runtime for 1 of 50 Samples"
rownames(oneof50_runtime) <- 1:nrow(oneof50_runtime)
rownames(oneof50_runtime) <- c(1,2,3)
rownames(oneof50_runtime) <- c("Elastic Net","Lasso", "Ridge")

oneof50_runtime

# 3(d)
# Fit the whole data set 

n.P                 =        sum(y)
n.N                 =        n - n.P
ww                  =        rep(1,n)
ww[y==1]            =        n.N/n.P

a=0.5 # elastic-net 0<a<1
# record run time - start
tic('Fit single Elastic Net')
start_time <- Sys.time() # record start time
cv.fiten         =     cv.glmnet(X, y, family = "binomial",intercept = TRUE, alpha = a, nfolds = 10, weights = ww)
fit.en           =     glmnet(X, y, family = "binomial", intercept = TRUE, alpha = a, lambda = cv.fiten$lambda.min, weights = ww, type.measure = "auc")
# record run time - end
toc()
end_time <- Sys.time() # record end time
runtime.en.wd <- end_time - start_time


a=1 # lasso
# record run time - start
tic('Fit single Lasso')
start_time <- Sys.time() # record start time
cv.fiten         =     cv.glmnet(X, y, family = "binomial",intercept = TRUE, alpha = a, nfolds = 10, weights = ww)
fit.ls           =     glmnet(X, y, family = "binomial", intercept = TRUE, alpha = a, lambda = cv.fiten$lambda.min, weights = ww, type.measure = "auc")
# record run time - end
toc()
end_time <- Sys.time() # record end time
runtime.ls.wd <- end_time - start_time


a=0  # ridge
# record run time - start
tic('Fit single Ridge')
start_time <- Sys.time() # record start time
cv.fiten         =     cv.glmnet(X, y, family = "binomial",intercept = TRUE, alpha = a, nfolds = 10, weights = ww)
fit.rid          =     glmnet(X, y, family = "binomial", intercept = TRUE, alpha = a, lambda = cv.fiten$lambda.min, weights = ww, type.measure = "auc")
# record run time - end
toc()
end_time <- Sys.time() # record end time
runtime.rid.wd <- end_time - start_time


#  random Forrest
tic('Fit Single Random Forest')
start_time <- Sys.time() 
rf.fit              =    randomForest( df$churn ~., data =df, mtry = sqrt(p))
toc()
end_time <- Sys.time() # record end time
runtime.rf.wd <- end_time - start_time

#Q3(d)
# part 1
# add random Forrest

# Create a table 4 x 2 table, the 4 rows corresponding to the 4 methods, and the two columns for test AUCs and time.
# The First column should show the median of test AUCs among the the 50 samples
# the second column the time it takes to fit the model on all the data

# median of test 50 AUCs
median.auc.test.en  <- median(auc.test.en)
median.auc.test.ls  <- median(auc.test.ls)
median.auc.test.rid <- median(auc.test.rid)

median.auc.test.en 
median.auc.test.ls
median.auc.test.rid

median.auc.test.rf <- median(auc.test.rf)

# runtime to fit models on all the data
runtime.en.wd
runtime.ls.wd
runtime.rid.wd
runtime.rf.wd


# combine median auc and runtime

plot_df <- cbind(rbind(median.auc.test.en ,    median.auc.test.ls,    median.auc.test.rid,  median.auc.test.rf), rbind(runtime.en.wd,  runtime.ls.wd,  runtime.rid.wd, runtime.rf.wd))
plot_df <- as.data.frame(plot_df)

names(plot_df)[1] <- "Median of 50 Test AUCs"
names(plot_df)[2] <- "Runtime"

rownames(plot_df) <- c('Elastic Net','Lasso', 'Ridge', 'Random Forest')

plot_df

# 3(d)
# Part 2 Coefficient Barplot

# elestic net  
betaS.en               =     data.frame(c(1:p), as.vector(fit.en$beta))
colnames(betaS.en)     =     c( "feature", "value")
betaS.en$feature       =     row.names(fit.en$beta)

#lasso
betaS.ls               =     data.frame(c(1:p), as.vector(fit.ls$beta))
colnames(betaS.ls)     =     c( "feature", "value")
betaS.ls$feature       =     row.names(fit.ls$beta)

#rid
betaS.rid              =   data.frame(c(1:p), as.vector(fit.rid$beta))
colnames(betaS.rid)    =     c( "feature", "value")
betaS.rid$feature      =     row.names(fit.rid$beta)

#rf
betaS.rf               =   data.frame(c(1:p), as.vector(rf.fit$importance[,1]))
colnames(betaS.rf)     =     c( "feature", "value")
betaS.rf$feature       =     row.names(rf.fit$importance)

# Resort the columns by the elastict-net coefficient value
betaS.ls$feature     =  factor(betaS.ls$feature, levels = betaS.en$feature[order(betaS.en$value, decreasing = TRUE)])
betaS.en$feature     =  factor(betaS.en$feature, levels = betaS.en$feature[order(betaS.en$value, decreasing = TRUE)])
betaS.rid$feature    =  factor(betaS.rid$feature, levels = betaS.rid$feature[order(betaS.en$value,decreasing = TRUE)])
betaS.rf$feature    =  factor(betaS.rf$feature, levels = betaS.rf$feature[order(betaS.rf$value,decreasing = TRUE)])

colnames(betaS.en)[2] <- "en.coef"
colnames(betaS.ls)[2] <- "ls.coef"
colnames(betaS.rid)[2] <- "rid.coef"
colnames(betaS.rf)[2] <- "rf.imp"

merge1 <- merge(betaS.en,betaS.ls,by="feature")
merge2 <- merge(merge1,betaS.rid,by="feature")
merge3 <- merge(merge2,betaS.rf,by="feature")

# plot.data <- merge2[order(- merge2$en.coef),]

plot.data <- merge3[order(- merge3$en.coef),]

# Plot coefficient and Randomr Forest Importance by Elastic Net

enPlot3 =  ggplot(plot.data[, 1:2], aes(x=feature, y=en.coef)) +
  geom_bar(stat = "identity", fill="tan3", colour="black")+
  ggtitle("Elastic Net Coefficient")+ 
  theme(axis.text.x = element_blank(), axis.title.x=element_blank())


lsPlot3 =  ggplot(plot.data[c("feature","ls.coef")], aes(x=feature, y=ls.coef),) +
  geom_bar(stat = "identity", fill="plum2", colour="black")+
  ggtitle("Lasso Coefficient") +
  theme(axis.text.x = element_blank(), axis.title.x=element_blank())


ridPlot3 =ggplot(plot.data[c("feature","rid.coef")], aes(x=feature, y=rid.coef))+
  geom_bar(stat="identity",fill="slategray1",colour="black")+
  ggtitle("Ridge Coefficient") +
  theme(axis.text.x = element_blank(), axis.title.x=element_blank())

rfPlot3 =ggplot(plot.data[c("feature","rf.imp")], aes(x=feature, y=rf.imp))+
  geom_bar(stat="identity",fill="palegreen",colour="black")+
  ggtitle("Random Forest Importance") + 
  theme(axis.text.x = element_text(color = "grey20", size = 5, angle = 90, hjust = .5, vjust = .5, face = "plain"), axis.title.x=element_blank())

# Generate Plot 2:
grid.arrange( enPlot3, lsPlot3, ridPlot3, rfPlot3,  ncol=1)

