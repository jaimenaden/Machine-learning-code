## Descriptive statistics of “message” file 

#import packages
library(moments)
#import data
mydata = read.csv("sample42.csv") 
attach(mydata)
#get the five number summary, sd, skewness and kurtosis
summary(Price)
summary(Size)
skewness(Price)
skewness(Size)
kurtosis(Price)
kurtosis(Size)
sd(Price)
sd(Size)


## Depth at Five Levels of Limit Order Book #import packages
library(moments)
#import data
mydata = read.csv("sample42.csv") 
attach(mydata)
#get the five number summary, sd, skewness and kurtosis
summary(ASKs1)
summary(ASKs2)
summary(ASKs3)
summary(ASKs4)
summary(ASKs5)

sd(ASKs1)
sd(ASKs2)
sd(ASKs3)
sd(ASKs4)
sd(ASKs5)


summary(BIDs1)
summary(BIDs2)
summary(BIDs3)
summary(BIDs4)
summary(BIDs5)

sd(BIDs1)
sd(BIDs2)
sd(BIDs3)
sd(BIDs4)
sd(BIDs5)



## Spread between Adjacent Levels of Limit Order Book 
#import packages
library(moments)
#import data
mydata = read.csv("sample42.csv") 
# get the absolute difference of different depth 
a<-BIDp1-BIDp2
b<-BIDp2-BIDp3
c<-BIDp3-BIDp4
d<-BIDp4-BIDp5
e<-ASKp1-BIDp1
f<-ASKp2-ASKp1
g<-ASKp3-ASKp2
h<-ASKp4-ASKp3
i<-ASKp5-ASKp4
#get the five number summary and sd

summary(a)
summary(b)
summary(c)
summary(d)
summary(e)
summary(f)
summary(g)
summary(h)
summary(i)
sd(a)
sd(b)
sd(c)
sd(d)
sd(e)
sd(f)
sd(g)
sd(h)
sd(i)



## static analysis assumption : split the data into half so we can check if samples are iid
# import data
mydata = read.csv("sample42.csv") 
attach(mydata)
#import packages
library(moments)
#split the data into half
smp_size <- floor(0.5 * nrow(mydata))
# set seed
set.seed(123)

train_ind <- sample(seq_len(nrow(mydata)), size = smp_size)
#train is first half of data
train <- mydata[train_ind, ]
#test is second half of data
test <- mydata[-train_ind, ]
train
summary(train)
summary(test)


## Autocorrelation Function 
# plot the ACF of BIDs1, ASKs1, BIDp1, ASKp1
#import packages
library(forecast)
library(ggplot2)
library(moments)
library(tseries)
library('tseries')

#import data
mydata = read.csv("sample42.csv") 
attach(mydata)
head(mydata)

#plot the acf
par(mfrow=c(2,2))
p1<-Acf(BIDs1)
p2<-Acf(ASKs1)
p3<-Acf(ASKp1)
p4<-Acf(BIDp1)
p6<-Acf(quotedspread)
adf.test(mydata$quotedspread, alternative = "stationary")
par(mfrow=c(2,2))
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

multiplot(p1,p2,p3,p4,cols=2)


## Time Series data of Bid and ask prices with time (days) 
#import data
prices = read.csv("sample42.csv") 
# import packages
library(ggplot2)
# plot bid and ask prices
askprice<-prices[,15]
bidprice<-prices[,16]
n = length(askprice)
df<-data.frame(time = rep(((1:n)/n*31),2),value = c(askprice,bidprice),bidask = rep(c("ask","bid"),c(n,n)))
ggplot(df,aes(time,value, colour = bidask))+geom_line()+scale_color_manual(values=c("red", "blue",))+ggtitle("Bid&Ask Prices") + xlab("Days") + ylab("Bid/Ask Price")



## Time Series data of Bid and ask size with time (days) 
#import data
prices = read.csv("sample42.csv") 
# import packages
library(ggplot2)
# plot bid and ask sizes
asksize<-prices[,5]
bidsize<-prices[,6]
n = length(asksize)
df<-data.frame(time = rep(((1:n)/n*31),2),value = c(asksize,bidsize),bidask = rep(c("ask","bid"),c(n,n)))
ggplot(df,aes(time,value, colour = bidask))+geom_line()+scale_color_manual(values=c("red", "blue",))+ggtitle("Bid&Ask Sizes") + xlab("Days") + ylab("Bid/Ask Size")


## Time Series data of Size with time (days) 
#import data
prices = read.csv("sample42.csv") 
# import packages
library(ggplot2)
# plot the data
n=dim(prices)[1]
size<-prices[,2]
#plot the time series data on quoted spread
# we have compress each day and then plot the whole quoted spread of the whole month
df5 = data.frame(time = (1:n)/n*31, Price = size)
ggplot(df5,aes(time,Price))+geom_line(color = "red")+ggtitle("Size") + xlab("Days") + ylab("Size")






## Time Series data of quoted spread with time (days) 
#import data
prices = read.csv("merged.csv") 
# import packages
library(ggplot2)
# plot the data
n=dim(prices)[1]
quotedspread<-prices[,24]
#plot the time series data on quoted spread
# we have compress each day and then plot the whole quoted spread of the whole month
df5 = data.frame(time = (1:n)/n*31, Price = size)
ggplot(df5,aes(time,Price))+geom_line(color = "red")+ggtitle("Size") + xlab("Days") + ylab("Size")


## Time Series data of relative spread with time (days) 
#import data
prices = read.csv("merged.csv") 
# import packages
library(ggplot2)
# plot the data
n=dim(prices)[1]
relativespread<-prices[,29]
#plot the time series data on relative spread
# we have compress each day and then plot the whole relative spread of the whole month
df5 = data.frame(time = (1:n)/n*31, Price = size)
ggplot(df5,aes(time,Price))+geom_line(color = "red")+ggtitle("relative spread") + xlab("Days") + ylab("Size")



## Time Series data of effective spread with time (days) 
#import data
prices = read.csv("merged.csv") 
# import packages
library(ggplot2)
# plot the data
n=dim(prices)[1]
effectivespread<-prices[,34]
#plot the time series data on effective spread
# we have compress each day and then plot the whole effective spread of the whole month
df5 = data.frame(time = (1:n)/n*31, Price = quotedspread)
ggplot(df5,aes(time,Price))+geom_line(color = "red")+ggtitle("Effective spread") + xlab("Days") + ylab("Quoted spread")





## Time Series data of percentage spread with time (days) 
#import data
prices = read.csv("merged.csv") 
# import packages
library(ggplot2)
# plot the data
n=dim(prices)[1]
percentagespread<-prices[,39]
#plot the time series data on percentage spread
# we have compress each day and then plot the whole percentage spread of the whole month
df5 = data.frame(time = (1:n)/n*31, Price = size)
ggplot(df5,aes(time,Price))+geom_line(color = "red")+ggtitle("Percentage spread") + xlab("Days") + ylab("Size")






# import packages
library(ggplot2)
#import data
prices = read.csv("sample.csv") 
#get the column on effective spread
effectivespread<-prices[,1]
#plot the time series data on effective spread
# we have compress each day and then plot the whole effective spread of the whole month
n=dim(prices)[1]
df5 = data.frame(time = (1:n)/n*31, Price = effectivespread)
ggplot(df5,aes(time,Price))+geom_line(color = "red")+ggtitle("Effective spread") + xlab("Days") + ylab("Effective spread")

# import packages
library(ggplot2)
#import data
prices = read.csv("sample1.csv") 
#get the column on relative spread
relativespread<-prices[,1]
#plot the time series data on relative spread
# we have compress each day and then plot the whole relative spread of the whole month
n=dim(prices)[1]
df5 = data.frame(time = (1:n)/n*31, Price = relativespread)
ggplot(df5,aes(time,Price))+geom_line(color = "red")+ggtitle("Relative spread") + xlab("Days") + ylab("Relative spread")


# import packages
library(ggplot2)
#import data
prices = read.csv("sample3.csv") 
#get the column on midquotes
percentagespread<-prices[,1]
#plot the time series data on percentage spread
# we have compress each day and then plot the whole percentage spread of the whole month
n=dim(prices)[1]
df5 = data.frame(time = (1:n)/n*31, Price = percentagespread)
ggplot(df5,aes(time,Price))+geom_line(color = "red")+ggtitle("Percentage spread") + xlab("Days") + ylab("Percentage spread")





## Do descriptive statistics/moments on measures for 5 levels 
#import data
mydata = read.csv("merged.csv") 
attach(mydata)

#import packages
library(moments)

#use summary() function in R to get five number summary, first quadrant of relative spread, effective spread and quoted spread at five levels etc


summary(quotespread1)
summary(quotespread2)
summary(quotespread3)
summary(quotespread4)
summary(quotespread5)

summary(relativespread1)
summary(relativespread2)
summary(relativespread3)
summary(relativespread4)
summary(relativespread5)


summary(effectivespread1)
summary(effectivespread2)
summary(effectivespread3)
summary(effectivespread4)
summary(effectivespread5)

summary(percentagespread1)
summary(percentagespread2)
summary(percentagespread3)
summary(percentagespread4)
summary(percentagespread5)

# get skewness and kurtosis and standard deviation of all midquotes
skewness(quotespread1)
kurtosis(quotespread1)
sd(quotespread1)

skewness(quotespread2)
kurtosis(quotespread2)
sd(quotespread2)

skewness(quotespread3)
kurtosis(quotespread3)
sd(quotespread3)

skewness(quotespread4)
kurtosis(quotespread4)
sd(quotespread4)

skewness(quotespread5)
kurtosis(quotespread5)
sd(quotespread5)


# get skewness and kurtosis and standard deviation of all relative spread

skewness(relativespread1)
kurtosis(relativespread1)
sd(relativespread1)

skewness(relativespread2)
kurtosis(relativespread2)
sd(relativespread2)

skewness(relativespread3)
kurtosis(relativespread3)
sd(relativespread3)

skewness(relativespread4)
kurtosis(relativespread4)
sd(relativespread4)

skewness(relativespread5)
kurtosis(relativespread5)
sd(relativespread5)


# get skewness and kurtosis and standard deviation of all effective spread

skewness(effectivespread1)
kurtosis(effectivespread1)
sd(effectivespread1)

skewness(effectivespread2)
kurtosis(effectivespread2)
sd(effectivespread2)

skewness(effectivespread3)
kurtosis(effectivespread3)
sd(effectivespread3)

skewness(effectivespread4)
kurtosis(effectivespread4)
sd(effectivespread4)

skewness(effectivespread5)
kurtosis(effectivespread5)
sd(effectivespread5)




# get skewness and kurtosis and standard deviation of all percentage spread

skewness(percentagespread1)
kurtosis(percentagespread1)
sd(percentagespread1)

skewness(percentagespread2)
kurtosis(percentagespread2)
sd(percentagespread2)

skewness(percentagespread3)
kurtosis(percentagespread3)
sd(percentagespread3)

skewness(percentagespread4)
kurtosis(percentagespread4)
sd(percentagespread4)

skewness(percentagespread5)
kurtosis(percentagespread5)
sd(percentagespread5)







## fit a linear regression to predict quoted spread
#import data
prices = read.csv("sample42.csv") 
attach(prices)
#split type into its factors since its categorical
prices$Type.f <- factor(prices$Type)
is.factor(prices$Type.f)
prices$Type.f[1:15]
#fit a linear regression
regression=lm(quotedspread~Type.f+Size+Price+TradeDirection+ASKs1+BIDs1+ASKs2+BIDs2+ASKs3+BIDs3+ASKs4+BIDs4+ASKs5+BIDs5+BIDp1+ASKp2+BIDp2+ASKp3+BIDp3+ASKp4+BIDp4+ASKp5+BIDp5,data=prices)
fit=summary(regression)

## fit a linear regression on log(quotedspread)
regression=lm(log(quotedspread)~Type.f+Size+Price+TradeDirection+ASKs1+BIDs1+ASKs2+BIDs2+ASKs3+BIDs3+ASKs4+BIDs4+ASKs5+BIDs5+BIDp1+ASKp2+BIDp2+ASKp3+BIDp3+ASKp4+BIDp4+ASKp5+BIDp5,data=prices)
fit=summary(regression)

## Arima model
# import data
data = read.csv("sample42.csv") 
attach(data)

#impoer packages
library(forecast)
library(ggplot2)
library(moments)

# get acf and pacf of quoted spread

Acf(quotedspread)
Pacf(quotedspread)

#use auto.arima() to see which model r suggest
auto.arima(mydata$quotedspread)


#fit an arima model to see which has lower aic
fit <- Arima(data$quotedspread, order=c(1,0,2))
tsdiag(fit)
fit <- Arima(data$quotedspread, order=c(1,0,0))
summary(fit)
fit <- Arima(data$quotedspread, order=c(1,0,1))
summary(fit)
fit <- Arima(data$quotedspread, order=c(1,1,3))
summary(fit)
fit <- Arima(data$quotedspread, order=c(1,0,3))
summary(fit)
fit <- Arima(data$quotedspread, order=c(0,0,1))
summary(fit)
fit <- Arima(data$quotedspread, order=c(0,0,0))
summary(fit)
fit <- Arima(data$quotedspread, order=c(1,0,4))
summary(fit)
fit <- Arima(data$quotedspread, order=c(1,0,5))
summary(fit)
fit <- Arima(data$quotedspread, order=c(1,0,6))
summary(fit)

#get acf of residuals in model (1,0,6)
Acf(residuals(fit))

#plot forecast model (1,0,6)
plot(forecast(fit))

# to get Ljung box plot 
tsdiag(fit)



#do adf test
adf.test(mydata$quotedspread, alternative = "stationary")



## Train vs validation loss graph of machine learning model
# plot train vs validation loss from our results or refer to python code
#import packages
library(ggplot2)
#import loss data dervied from the results of machine learning fitted model
prices = read.csv("loss.csv") 
attach(prices)
prices[,1]
#plot train vs validation loss
df<-data.frame(epoch = rep(((1:n)/n*31),2),value = c(askprice,bidprice), Loss= rep(c(" Training loss","validation loss"),c(n,n)))
df
ggplot(df,aes(epoch,value, colour = Loss))+geom_line()+ggtitle("Model train vs validation loss") + xlab("Epoch") + ylab("loss")



