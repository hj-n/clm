library(MASS)
library(ggplot2)

### --- DO NOT EDIT --- ### 
### These codes reproduces plots shown during experiment 1 of paper ClustMe
### https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.13684
###
### M. Aupetit - QCRI - Feb 3, 2022


##############################################################################
##############################################################################
#### Functions 
##############################################################################
##############################################################################
# gmm_sample
# plotPoint


### Sampling of the GMM wrt to parameters
gmm_sample = function(Nsample, MuAx, MuAy, SigmaAx, SigmaAy, ThetaA,
                      MuBx, MuBy, SigmaBx,  SigmaBy, ThetaB, Tau = 0.5,
                      seedval=0,
                      LabelA = 1, LabelB = 2){
  
  # init of the random generator's seed value
  set.seed(seedval)
  
  # Tau is the proportion of points in component A
  
  # label values for the components:
  # LabelA = 1  # label of the points sampled from component A (MuA, SigmaA, ThetaA)
  # LabelB = 2  # label of the points sampled from component B (MuB, SigmaB, ThetaB)
  
  
  ##generation of (n x 3) matrix. 
  ##Each column is a random sample of size n from a single component of the mixture
  MuA = c(MuAx, MuAy)
  SigmaA_matrix = matrix(c(SigmaAx,0, 0,SigmaAy), nrow=2)
  A_points = MASS::mvrnorm(n=Nsample, mu=MuA, Sigma=covFromRotAndScale(theta = ThetaA, Sigma = SigmaA_matrix))
  
  MuB = c(MuBx, MuBy)
  SigmaB_matrix = matrix(c(SigmaBx,0, 0,SigmaBy), nrow=2)
  B_points = MASS::mvrnorm(n=Nsample, mu=MuB, Sigma=covFromRotAndScale(theta = ThetaB, Sigma = SigmaB_matrix))
  
  temp_x = cbind(A_points[,1],B_points[,1])
  temp_y = cbind(A_points[,2],B_points[,2])
  temp_label = cbind(rep(LabelA,Nsample),rep(LabelB,Nsample))
    
  ## random shuffling of the indices and prior probability
  id = sample(1:2,Nsample,rep = T,prob = c(Tau,1-Tau))  # columns where to read the label in temp_label, 1 is comp A, 2 is comp B
  id = cbind(1:Nsample,id)
  
  return(data.frame(x=as.numeric(temp_x[id]),y=as.numeric(temp_y[id]),label=as.numeric(temp_label[id]),stringsAsFactors = FALSE))
  
}

#### Utility function to compute the covariance matrix from the Sigma and Theta parameters
covFromRotAndScale = function(theta, Sigma){
  rot<- rotMat(theta) 
  identity_matrix = matrix(c(1,0, 0,1), nrow=2)
  return(rot %*% Sigma %*% t(rot))
}

rotMat<- function(ang){
  return(matrix(c(cos(ang),sin(ang),-sin(ang),cos(ang)),nrow=2))
}
  
# rotate 2D points x,y by angle ang
rotatePoints <- function(inPointsXY,ang){
  # inPointsXY data frame with list x and y (other list are copied in the result)
  # return outPointsXY: same points rotated by ang around (0,0)
  dt=cbind(inPointsXY$x,inPointsXY$y)
  res=t(rotMat(ang)%*%t(dt))
  outPointsXY = inPointsXY
  outPointsXY$x=res[,1]
  outPointsXY$y=res[,2]
  return(outPointsXY)
}

#### Plotting the data as a scatterplot as it was during the Experiment 1 for collecting the human judgments
#### DO NOT MODIFY #### AS IT IS THE SETTING USED DURING THE EXPERIMENT
plotPoints<- function(pointsXY,labelCol=NULL,titleTxt=NULL,titleSz=10,titleCol="black"){
  ### pointXY = dataframe with column names "x" "y" "label"; label is an integer > 0 used t index the colors in labelCol if labelCol not NULL)
  ### labelCol = c("blue", "red") to display color-coded generating component
  
  # get axis range of equal size on both axes
  minx = min(pointsXY$x)  # actual min of x axis
  miny = min(pointsXY$y)  # actual min of y axis
  maxx = max(pointsXY$x)  # actual max of x axis
  maxy = max(pointsXY$y)  # actual max of y axis
  dmxn <- max(c(maxx-minx,maxy-miny))    # max range on all axes
  cmxn = 0.5*(minx+maxx)   # center of x axis range
  cmyn = 0.5*(miny+maxy)   # center of y axis range
  xmin = cmxn-0.6*dmxn  # let 10% margin on x axis left
  ymin = cmyn-0.6*dmxn  # let 10% margin on y axis bottom
  xmax = cmxn+0.6*dmxn  # let 10% margin on x axis right
  ymax = cmyn+0.6*dmxn  # let 10% margin on y axis top

  ### PLOT

   
  gplot <- ggplot()
    
    
    if (!is.null(labelCol) && !is.null(pointsXY$label)) {
      # color code the points
      pointsXY$labelCol=labelCol[pointsXY$label]
      gplot <- gplot + geom_point(data=pointsXY,aes(x=x,y=y,color=labelCol),shape=19,alpha=1) # plot the data as a scatterplot
    } else {
      gplot <- gplot + geom_point(data=pointsXY,aes(x=x,y=y),color="black",shape=19,alpha=1) # plot the data as a scatterplot
    }
        
    if (!is.null(titleTxt)) {
      # add a title
      TXT_LABEL=titleTxt
      TXT_COLOR=titleCol
      TXT_SIZE=titleSz
      gplot <- gplot + ggplot2::annotate("text",label=TXT_LABEL,x=0.5*(xmin+xmax),y=ymax,color=TXT_COLOR,size=TXT_SIZE)
    }
  
  gplot <- gplot + 
    coord_fixed()+ # make sure 1 unit in x and 1 unit in y are display same length on the screen
    xlim(xmin,xmax)+ # set the x axis range
    ylim(ymin,ymax)+  # set the y axis range
    theme(aspect.ratio = 1,
          axis.line.x=element_blank(),
          axis.line.y=element_blank(),
          axis.text.x=element_blank(),
          axis.text.y=element_blank(),
          axis.ticks=element_blank(),
          axis.title.x=element_blank(),
          axis.title.y=element_blank(),
          legend.position="none",
          panel.background=element_blank(),
          #panel.border=element_blank(),
          panel.border = element_rect(colour = "black", fill=NA, size=2),
          panel.grid.major=element_blank(),
          panel.grid.minor=element_blank(),
          plot.background=element_blank(),
          plot.margin=unit(c(0,0,0,0), "cm"))
  
  return(gplot)
}

### GenePointsAsDisplayed
genePointsXY <- function(df,seedval=0,LabelA = 1, LabelB = 2){
  
  # df, dataframe (1 row) containing all parameters
  # [1] "Nsample"    "MuA1"       "MuA2"       "SigmaA1"    "SigmaA2"    "ThetaA"     "MuB1"       "MuB2"       "SigmaB1"    "SigmaB2"   
  # [11] "ThetaB"     "Tau"        "Alpha"      
  
  ## generate samples from gmm parameters and number of points (Nsample)
  pointsXYtmp=gmm_sample(Nsample=df$Nsample, 
                         MuAx=df$MuA1, 
                         MuAy=df$MuA2, 
                         SigmaAx=df$SigmaA1, 
                         SigmaAy=df$SigmaA2, 
                         ThetaA=df$ThetaA,
                         MuBx=df$MuB1, 
                         MuBy=df$MuB2, 
                         SigmaBx=df$SigmaB1, 
                         SigmaBy=df$SigmaB2, 
                         ThetaB=df$ThetaB, 
                         Tau=df$Tau,
                         seedval = seedval,
                         LabelA=LabelA, 
                         LabelB=LabelB)
  
  ## rotate the result by angle Alpha 
  pointsXY = rotatePoints(pointsXYtmp,df$Alpha)
  
  return(pointsXY)
  
}


  