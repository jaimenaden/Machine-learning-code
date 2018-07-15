# code adaped from lobster sample code
# load all packages
library(graphics)
 
# Google stock is my ticker
ticker  <- "GOOGL"     
# look at data on 2017-08-09                 
demodate = "2017-08-09"
# start time is 9.30 pm
starttime <- 34200000 
# end time is 4pm
endtime <- 57600000
# Levels
nlevels         = 10;
# set up my Orderbook
ORDERBOOK <- paste(paste(ticker , demodate ,starttime,endtime,"orderbook" ,nlevels ,sep = "_"),"csv",sep = ".")

# set up my MSGbook
MSGBOOK <- paste(paste(ticker , demodate ,starttime,endtime,"message" ,nlevels ,sep = "_"),"csv",sep = ".")

# Load data
datamessage <- read.csv ( MSGBOOK )   

# Name the columns 
columns <- c ( "Time" , "Type" , "OrderID" , "Size" , "Price" , "TradeDirection" )
colnames ( datamessage ) <- columns
 
# start and end trading hours
startTrad   = 9.5*60*60;       # 9:30:00.000 in ms after midnight
endTrad     = 16*60*60;        # 16:00:00.000 in ms after midnight


# remove observations outside of trading hours
datamessage_part = datamessage[datamessage$Time>=startTrad & datamessage$Time<=endTrad,]


# look for halts in trading

tradehaltIdx = which(datamessage[,2] == 7 & datamessage[,5] == -1 );
tradequoteIdx = which(datamessage[,2] == 7 & datamessage[,5] == 0 );
traderesumeIdx = which(datamessage[,2] == 7 & datamessage[,5] == 1 );
  
		
if(length(tradehaltIdx)==0 & length(tradequoteIdx)==0  & length(traderesumeIdx)==0 )
	print("No trading halts detected.")
		
if(length(tradehaltIdx) !=0)
	cat("Data contains trading halt! at time stamp(s)", datamessage[tradehaltIdx,1],"\n" )
		
if(length(tradequoteIdx) !=0)
	cat(" Data contains quoting message! at time stamp(s)", datamessage[tradequoteIdx,1], "\n")
			
if(length(traderesumeIdx) !=0)
	cat(" Data resumes trading! at time stamp(s) ", datamessage[traderesumeIdx,1],"\n")
		


#plot intrday bounds

# interval length
freq = 5*60;   # Interval length in ms 5 minutes

# No of intervals from 9:30am to 4:00pm
noint= (endTrad-startTrad)/freq
datamessage_part$index = seq(from=1,to=dim(datamessage_part)[1])

# Variables for 'for' loop
j= 0
l =0
bound =0               	  # Variable for inverval bound
visible_count = 0         # visible_count calculates the number of visible trades in an interval of 5 min
hidden_count = 0          # hidden_count calculates the number of visible trades in an interval of 5 min
visible_size = 0          # Total volume of visible trades in an interval of 5 minutes
hidden_size = 0           # Total volume of hidden trades in an interval of 5 minutes

# Set Bounds for Intraday Intervals
for(j in c(1:noint)) {

	bound[j+1] = startTrad + j * freq
	}
bound[1] = startTrad


# Calculate number of visible, hidden trades and its volume

for(l in c(1:noint)) 
{
         visible_count[l] = nrow(datamessage_part[datamessage_part$Time > bound[l] & datamessage_part$Time < bound[l+1] & datamessage_part$Type == 4,])
         visible_size[l] = sum(datamessage_part[datamessage_part$Time > bound[l] & datamessage_part$Time < bound[l+1] & datamessage_part$Type == 4,4])/100
         
         hidden_count[l] = nrow(datamessage_part[datamessage_part$Time > bound[l] & datamessage_part$Time < bound[l+1] & datamessage_part$Type == 5,])
         hidden_size[l] = sum(datamessage_part[datamessage_part$Time > bound[l] & datamessage_part$Time < bound[l+1] & datamessage_part$Type == 5,4])/100
         
}
      
# Split area of plot into two windows
par(mfrow=c(1,2) )
	  
# Plot number of visible trades
plot(c(1:noint),visible_count , type ='h' , lwd = 5 , col = 'red' , ylim = c(-max(hidden_count), max(visible_count)) ,ylab ="Number of Executions" ,xlab = "Interval" )
title(sprintf("Number of Executions by Interval for %s " ,ticker ,cex = 0.8 ) )

# No of hidden trades
lines(c(1:noint),-hidden_count, type ='h' , lwd = 5 , col = 'blue' )

# Legend
legend("top", c('Hidden' ,'Visible'),  col=c('blue','red'), horiz = TRUE , lty = 1 ,  inset = .05)

# Second plot of visible volume
plot(c(1:noint),visible_size, type ='h' , lwd = 5 , col = 'red' , ylim = c(-max(hidden_size)-200, max(visible_size)) , ylab ="Volume of Trades(x100 shares)" ,xlab ="Interval")
title( sprintf("Trade Volume by Interval for %s " ,ticker ,cex = 0.8 ))

# Hidden volume in an interval
lines(c(1:noint),-hidden_size, type ='h' , lwd = 5 , col = 'blue' )

# Legend
legend("top", c('Hidden' ,'Visible'),  col=c('blue','red'), horiz = TRUE , lty = 1 ,  inset = .05)

# Load data
dataorder <- read.csv( ORDERBOOK )   
                                  
	# Note: The file contains more than 250 000 entries. It takes a few seconds to load.

columns2 <- c("ASKp1" , "ASKs1" , "BIDp1",  "BIDs1")

# naming the columns of data frame                                          
if (nlevels > 1)
{
	for ( i in 2:nlevels )
	{ 
  		columns2 <- c (columns2,paste("ASKp",i,sep=""), paste("ASKs",i,sep=""),paste("BIDp",i,sep=""),paste("BIDs",i,sep="")) 
	}
}
	
colnames ( dataorder) <- columns2


# Trading hours start and end
timeindex <-datamessage$Time>=startTrad & datamessage$Time<=endTrad

dataorder_part = dataorder[timeindex,]
# Convert prices into dollars
#    Note: LOBSTER stores prices in dollar price times 10000

for ( i in c(seq(from = 1, length=2*nlevels, by = 2)) ) 
{ 
	dataorder_part[,i ]  = dataorder_part[ ,i]/10000 
}

#plot snapshot of lob

par(mfrow=c(1,2))
# Note: Pick a random row/event from the order book
totalrows <- nrow(dataorder_part)
random_no <- sample(1:totalrows, 1, replace=F)


# Select colour for ASK and BID prices bars, greeen for BID and Red for ASK
colmatrix = matrix(0,1,4)
colmatrix = c("red","green","red","green","red","green","red","green" )#

# Plot
plot(x=as.numeric(dataorder_part[random_no,seq(from=1,by=2,length=2*nlevels)]),y=as.numeric(dataorder_part[random_no,seq(from=2,by=2,length=2*nlevels)]),type="h" , lwd = 5,col = colmatrix ,xlab ="Price($)", ylab = "Volume" )

# Title 
title(sprintf("Limit Order Book Volume for %s at %s ms" ,ticker,datamessage_part[random_no,1]) , cex = 0.8)

# Legend
legend("top",c('BID', 'ASK'), lty = 1, col=c('green','red'),ncol=1 , , horiz = TRUE,inset = .03)


#plot relative depth

# Plot variables
percASK =0
x <- c(1:nlevels)

# Separating volume data from data_frame DataOB
randoms = dataorder_part[random_no,seq(from=2,by=2,length=2*nlevels)]

# ASK side cumulative sum
totalsize = cumsum( c(randoms[1,seq(from=1,by=2,length=nlevels),]))
percASK = totalsize/totalsize[nlevels]

# Plot
plot(x,percASK ,type = "s" , col = 'red' ,lwd = 4 , ylim= c(-1,1) , xlab = "Level" , ylab = "% of volume" )

# Title 
title(sprintf("Relative Depth in the Limit Order Book for %s at %s ms for %s levels" ,ticker,dataM_part[random_no,1] , nlevels) , cex = 0.8 )

# BID side
totalsize2 = cumsum(c(randoms[1,seq(from=2,by=2,length=nlevels),] ))
percBID = -totalsize2/totalsize2[nlevels]

# BID side
lines(x,percBID ,type = "s" , col = 'green' , lwd = 4)

# Legend
legend("bottomleft",c('Ask','Bid' ), lty = 1, col=c('red','green' ),ncol=1 ,horiz = TRUE,  inset = .05)

#plot intrday evolution of depth
par(mfrow=c(1,1))
# Calculate the max/ min volume to set limit of y-axis
maxaskvol = max(max(dataorder_part$ASKs1/100), max(dataorder_part$ASKs2/100),  max(dataorder_part$ASKs3/100) )  # calculate the maximum ask volume

# Calculate the max Bid volume , we use negative here and calculate min as we plot Bid below X-axis
maxbidvol = min(min(-dataorder_part$BIDs1/100), min(-dataorder_part$BIDs2/100),  min(-dataorder_part$BIDs3/100) )

# Plot ASK VOLUME level 3
plot ( (datamessage_part$Time/(60*60)), ((dataorder_part$ASKs1/100)+(dataorder_part$ASKs2/100)+ (dataorder_part$ASKs3/100)),type = "h" , col = 'red' , axes=FALSE,ylim = c(maxbidvol ,maxaskvol) , ,ylab = "BID              No of Shares(x100)               ASK" , xlab = "Time",frame=TRUE)

# ASK VOLUME level 2
lines( (datamessage_part$Time/(60*60)), ((dataorder_part$ASKs1/100)+(dataorder_part$ASKs2/100)),type = "h" , col = 'green')

# ASK VOLUME level 1
lines ( (datamessage_part$Time/(60*60)),(dataorder_part$ASKs1/100),type = "h" , col = 'blue')

# BID VOLUME level 3
lines ( (datamessage_part$Time/(60*60)), (-(dataorder_part$BIDs1/100)-(dataorder_part$BIDs2/100)-(dataorder_part$BIDs3/100)),type = "h" , col = 'burlywood1' )

# BID VOLUME level 2
lines( (datamessage_part$Time/(60*60)), -(dataorder_part$BIDs2/100)-(dataorder_part$BIDs1/100),type = "h" , col = 'purple')

# BID VOLUME level 1
lines ( (datamessage_part$Time/(60*60)), -(dataorder_part$BIDs1/100),type = "h" , col = 'cyan')


# Labels
axis(side=1, at=c(10,11,12,13,14,15,16), label=c(10,11,12,13,14,15,16), lwd=1,cex.axis=0.8,padj=0.55)
axis(side=2, at=c(maxbidvol,0,maxaskvol), label=c(-maxbidvol,0,maxaskvol), lwd=1,cex.axis=0.8,padj=0.55)
title(sprintf("Intraday Evolution of Depth for %s for %s levels" ,ticker, nlevels-7) , cex = 0.8 )

# Legend
legend("bottom",c('ASK 1','ASK 2', 'ASK 3', 'BID 1' , 'BID 2' , 'BID 3'),  col=c('blue','green','red','cyan','purple','darkorange'), horiz = TRUE , lty = 1 ,  inset = .05,cex = 0.7)






