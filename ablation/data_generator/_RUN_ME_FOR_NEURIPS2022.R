### Generate MD points from 2D points by adding normal noise on M-2 coordinates
### Sample N data points from a 2-components GMM with each of the 1000 sets of parameters 
# These data have been used to generate the scatterplots in EXP1 of ClustMe paper
# https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.13684


##################
parameters_file = "1000_2gaussians_param_ClustMe_EXP1.Rdata" 
N = 100 # NUMBER OF SAMPLE POINTS 
MAXSPL = 0 # generate MAXSPL+1 sampling of the same distribution 
M = 500 # output dimension: first 2 are GMM, last M-2 are noise  
resultfolderpath = "./" # same path as this code
KMAX= nrow(gmm_param) #10 # set to nrow(gmm_param) to generate all datasets 

### NEURIPS SETTINGS ###
# N = 10000
# MAXSPL = 0
# M = 500
# resultfolderpath = "./"
# KMAX= nrow(gmm_param) 


show2Dplot = FALSE  # show a 2D scatterplot of the data (Don't do that for all data!)
show3Dplot = FALSE  # show a 3D scatterplot of the data (Don't do that for all data!)

###################

library(rgl)

### get scripts to sample from GMM based on parameters, and to plot the data
source("GenePointsXYfromGMMparam.R")


# load original parameters
load(parameters_file) # 1000 sets of parameters -> dataframe gmm_param

# colnames(gmm_param)
# [1] "Nsample"    "MuA1"       "MuA2"       "SigmaA1"    "SigmaA2"    "ThetaA"     "MuB1"       "MuB2"       "SigmaB1"    "SigmaB2"   
# [11] "ThetaB"     "Tau"        




## generate x,y, noise1...noiseM-2, label coordinates
for (seedval in 0:MAXSPL){
  for (k in 1:KMAX){
    
    
    print(paste0("Seed: ",seedval," -- ",k,"/",KMAX))
    
    LabelA="1"
    LabelB="2"
    ## generate points to display in the scatterplot following parameters of the model
    df_parameters = gmm_param[k,]
    df_parameters$Nsample = N
    df_parameters$Alpha = 0
    
    pointsXY = genePointsXY(df_parameters,seedval=seedval, LabelA=LabelA,LabelB=LabelB)
    
    indptsA = which(pointsXY$label==LabelA)
    indptsB = which(pointsXY$label==LabelB)
    numptsA = length(indptsA)
    numptsB = length(indptsB)
    
    ## compute noise val: 
    
    ## take minimum of variance of component 1 
    SigmaA = min(gmm_param$SigmaA1[k],gmm_param$SigmaA2[k])
    
    ## take minimum of variance of component 2 
    SigmaB = min(gmm_param$SigmaB1[k],gmm_param$SigmaB2[k])
    
    ## generate normal noise on M-2 coordinates by adding noise with variance v1 or v2 to points generate by components 1 or 2.
    SigmaA_matrix = diag(SigmaA, M-2)
    
    noiseA = MASS::mvrnorm(n=numptsA, mu=rep(0,M-2), Sigma=SigmaA_matrix)
    
    SigmaB_matrix = diag(SigmaB, M-2)
    
    noiseB = MASS::mvrnorm(n=numptsB, mu=rep(0,M-2), Sigma=SigmaB_matrix)
    
    ## concatenate to x,y data to form MD data
    numpts = numptsA+numptsB
    noiseAB = matrix(0,numpts,M-2)
    noiseAB[indptsA,] = as.numeric(noiseA)
    noiseAB[indptsB,] = as.numeric(noiseB)
    pointsXYnoise = cbind(pointsXY$x,pointsXY$y,noiseAB)
    
    ## add the label as last column
    pointsXYnoiseLabel = as.data.frame(cbind(pointsXYnoise,as.numeric(pointsXY$label)))
    colnames(pointsXYnoiseLabel)=c("x","y",paste0("noise",1:(M-2)),"label")
    
    ## RE-SAMPLED - COLOR
    ## plot RE-SAMPLED points from same parameterized GMM as seen during the experiment  
    ## with color-coded label of generating component (Use $x, $y and $label)
    ## The distribution displayed corresponds to the one of the original data, up to resampling 
    if (show2Dplot == TRUE){
      print(plotPoints(pointsXYnoiseLabel,
                       labelCol=c("red", "blue"),
                       titleTxt=paste0(rownames(gmm_param[k,])," - RE-SAMPLED"),
                       titleSz=5))
    }
    
    ## 3D view to check how added noise impact the distribution
    if (show3Dplot == TRUE){
      ptsX=pointsXYnoiseLabel$x
      ptsY=pointsXYnoiseLabel$y
      ptsZ=pointsXYnoiseLabel$noise10 # pick on of the noisy dimension
      mnx=min(ptsX)
      mxx=max(ptsX)
      mny=min(ptsY)
      mxy=max(ptsY)
      mnz=min(ptsZ)
      mxz=max(ptsZ)
      mn=min(c(mnx,mny,mnz))
      mx=max(c(mxx,mxy,mxz))
      
      plot3d(ptsX, ptsY, ptsZ, col=pointsXYnoiseLabel$label, size=3,
             xlim = c(mn,mx), ylim = c(mn,mx), zlim = c(mn,mx))
    }
    
    ## Generate the csv files of x,y,...,c data in the scatter plot
    fn = rownames(gmm_param[k,])
    pth = resultfolderpath 
    write.csv(pointsXYnoiseLabel,paste0(pth,fn,"_noise498_num",N,"_seed",seedval,".csv"), row.names=FALSE)
  }
}
