#Running GLM of spatial variables across main survey
	#also creates table required for plotting bubble map

#############################################################################
#copying from t1 v2:
#1. Libraries
library(data.table) #eg fread, fwrite
library(tidyverse) #eg %>%, ggplot

#2. Loading 25cols data
##Loading predictions from classifier 
    ##NB: These are very large files so they weren't uploaded to github
raw_data  = list.files(pattern="*table_25cols*.csv") %>% map_df(~fread(.))
dim(raw_data) #18614775       25 with 1970 row    
colnames(raw_data)

#3. Tidying dataset
#a. Removing 1970 row
crop_data <- raw_data[-(which(raw_data$WAV_date == "1970-01-01")),]
dim(crop_data) #18614774       25
#b. Removing 22:00 rows
Audio_table <- crop_data[-which(crop_data$clip_start_hour == "22"),]
dim(Audio_table) #18614742       25
#c. Tidying R memory
rm(crop_data)
gc()

#============================================================
#4. Sampling Effort
#a) Site
#getting values
SE_site <- Audio_table[, .N, by =.(AudioMoth)]
#Number of 5s clips*5 gives number of seconds
#Divided by 60 gives number of minutes
#Divided again by 60 gives number of hours
SE_site$Hours <- (SE_site$N*5/60)/60
# #Days equivalent  #4:30-12:00 is 7h30m
# SE_site$Days <- SE_site$Hours*(7.5/24)

#============================================================
#5. Total number of detections - raw number - By Site
#a) Site gc
setkey(Audio_table, AudioMoth)
#gc detections grouped by site, including zero counts
Site_gc <- Audio_table[MT_gc=="gc", .N, by=.(AudioMoth)][CJ(unique(Audio_table$AudioMoth))][is.na(N), N:=0L]
Site_gc[, "Category" := "gc"]
#colnames(Site_gc)[2] <- "gc_N"

#b) mm
Site_mm <- Audio_table[MT_mm=="mm", .N, by=.(AudioMoth)][CJ(unique(Audio_table$AudioMoth))][is.na(N), N:=0L]
Site_mm[, "Category" := "mm"]
#colnames(Site_mm)[2] <- "mm_N"

#c) sc
Site_sc <- Audio_table[MT_sc=="sc", .N, by=.(AudioMoth)][CJ(unique(Audio_table$AudioMoth))][is.na(N), N:=0L]
Site_sc[, "Category" := "sc"]
#colnames(Site_sc)[2] <- "sc_N"

#d) background
Site_background <- Audio_table[MT_background=="background", .N, by=.(AudioMoth)][CJ(unique(Audio_table$AudioMoth))][is.na(N), N:=0L]
Site_background[, "Category" := "background"]
#colnames(Site_background)[2] <- "background_N"

#e) all categories
# setkey(Audio_table)
# ##, c(AudioMoth,MT_gc, MT_mm, MT_sc, MT_background)
# Site_category <- Audio_table[, .N, by=.(AudioMoth, MT_gc, MT_mm, MT_sc, MT_background)][CJ(unique(Audio_table$AudioMoth))][is.na(N), N:=0L]
Site_category <- rbind(rbind(rbind(Site_gc, Site_mm), Site_sc), Site_background)

############################################################

##Extra
# #f) Saving counts by site
# Counts_by_site <- data.frame(AudioMoth=SE_site$AudioMoth,
#            Sampling_Hours= SE_site$Hours, background_counts = Site_background$N, 
#            gc_counts = Site_gc$N, mm_counts = Site_mm$N, sc_counts = Site_sc$N)
# write.csv(Counts_by_site, "D1_Output_counts_by_site_table.csv", row.names = F)

#======================================================

#6. Loading spatial data
###Spatial <- read.csv("../../7_Spatial/16.Summary_metadata_55rows,_with_distance_from_water_pixel.csv")
Spatial_metadata <- read.csv("../data/A2_Main_17d)_Summary_2021_metadata_55coordinates.csv")

#7. Combining counts with spatial data
# GLM_table <- data.frame(AudioMoth=SE_site2$AudioMoth,
#                         Sampling_Hours= SE_site2$Hours, background_counts = Site_background2$N, 
#                         gc_counts = Site_gc2$N, mm_counts = Site_mm2$N, sc_counts = Site_sc2$N, 
#                         X=Spatial3$X, Y=Spatial3$Y, elevation=Spatial3$elevation,
#                         tree_cover=Spatial3$tree_cover, provisional_edge_dist=Spatial3$Distance_from_edge,
#                         Forest_500m_binary=Spatial3$Forest_edge_500m)
#write.csv(GLM_table, "GLM_table_12cols.csv", row.names = FALSE)
GLM_table <- data.frame(AudioMoth=SE_site$AudioMoth,
                        Sampling_Hours= SE_site$Hours, background_counts = Site_background$N, 
                        gc_counts = Site_gc$N, mm_counts = Site_mm$N, sc_counts = Site_sc$N, 
                        Spatial_metadata[,c(1:13)])
##GLM_table$AudioMoth == GLM_table$Folder.name
write.csv(GLM_table, "../data/C1_GLM_table_19cols.csv", row.names = FALSE)

#====================================================================================

#8. GLM
#a) gc counts
#i) poisson
      # summary(glm(gc_counts~elevation+tree_cover+Distance_from_edge+offset(log(Sampling_Hours)),data=GLM_table, family="poisson"))
      #   #Residual deviance: 59451  on 51  degrees of freedom
      # #Dispersion Parameter
      # #residual deviance/residual degrees of freedom
      # 59451/51 #1165.706
#therefore overdispersed
summary(glm(gc_counts~elevation+tree_cover+Distance_from_edge+tree_height+slope+aspect+offset(log(Sampling_Hours)),data=GLM_table, family="poisson"))
# Residual deviance: 52662  on 48  degrees of freedom
52662/48 # 1097.125
#ii) using quasipoisson
      # summary(glm(gc_counts~elevation+tree_cover+Distance_from_edge+offset(log(Sampling_Hours)),data=GLM_table, family="quasipoisson"))
      #     # Estimate Std. Error t value Pr(>|t|)  
      #     # (Intercept)         2.466481   2.538255   0.972   0.3358  
      #     # elevation          -0.004527   0.003113  -1.454   0.1520  
      #     # tree_cover         -0.003092   0.015295  -0.202   0.8406  
      #     # Distance_from_edge  0.001450   0.000759   1.911   0.0617 .
#diagnostic plots
  #Eg for interpretation see: https://www.statology.org/diagnostic-plots-in-r/
#par(mfrow=c(2,2))
plot(glm(gc_counts~elevation+tree_cover+Distance_from_edge+tree_height+slope+aspect+offset(log(Sampling_Hours)),data=GLM_table, family="quasipoisson"))
#running glm
summary(glm(gc_counts~elevation+tree_cover+Distance_from_edge+tree_height+slope+aspect+offset(log(Sampling_Hours)),data=GLM_table, family="quasipoisson"))
    # Estimate Std. Error t value Pr(>|t|)  
    # (Intercept)         1.5035500  2.4340159   0.618    0.540  
    # elevation          -0.0061015  0.0030758  -1.984    0.053 .
    # tree_cover         -0.0181750  0.0166889  -1.089    0.282  
    # Distance_from_edge  0.0009939  0.0007468   1.331    0.189  
    # tree_height         0.1877024  0.1165117   1.611    0.114  
    # slope               0.0077713  0.0225041   0.345    0.731  
    # aspect              0.0016430  0.0024971   0.658    0.514
##(Dispersion parameter for quasipoisson family taken to be 1933.536)

#b) mm counts
#i) dispersal
      # summary(glm(mm_counts~elevation+tree_cover+Distance_from_edge+offset(log(Sampling_Hours)),data=GLM_table, family="poisson"))
      # #Residual deviance: 36402  on 51  degrees of freedom
      # 36402/51 #713.7647
summary(glm(mm_counts~elevation+tree_cover+Distance_from_edge+tree_height+slope+aspect+offset(log(Sampling_Hours)),data=GLM_table, family="poisson"))
#Residual deviance: 33725  on 48  degrees of freedom
33725/48 #702.6042
#ii) quasipoisson
        # summary(glm(mm_counts~elevation+tree_cover+Distance_from_edge+offset(log(Sampling_Hours)),data=GLM_table, family="quasipoisson"))
        #     # Coefficients:
        #     # Estimate Std. Error t value Pr(>|t|)   
        #     # (Intercept)        -2.8606913  1.3332924  -2.146  0.03669 * 
        #     # elevation           0.0038234  0.0013243   2.887  0.00569 **
        #     # tree_cover          0.0100857  0.0081492   1.238  0.22152   
        #     # Distance_from_edge  0.0004752  0.0002632   1.806  0.07684 . 
        #     # ---
        #     #   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
        #     # 
        #     # (Dispersion parameter for quasipoisson family taken to be 761.949)
#diagnostic plots
plot(glm(mm_counts~elevation+tree_cover+Distance_from_edge+tree_height+slope+aspect+offset(log(Sampling_Hours)),data=GLM_table, family="quasipoisson"))
#running glm
summary(glm(mm_counts~elevation+tree_cover+Distance_from_edge+tree_height+slope+aspect+offset(log(Sampling_Hours)),data=GLM_table, family="quasipoisson"))
    # Estimate Std. Error t value Pr(>|t|)   
    # (Intercept)        -2.8669365  1.3510813  -2.122  0.03903 * 
    # elevation           0.0036669  0.0013122   2.794  0.00745 **
    # tree_cover          0.0096320  0.0084875   1.135  0.26208   
    # Distance_from_edge  0.0005603  0.0002724   2.057  0.04517 * 
    # tree_height         0.0275249  0.0387723   0.710  0.48119   
    # slope              -0.0138391  0.0084847  -1.631  0.10942   
    # aspect             -0.0006505  0.0008678  -0.750  0.45717   
    # ---
    #   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    # 
    # (Dispersion parameter for quasipoisson family taken to be 736.2983)

#c) sc counts
#i) dispersal
        # summary(glm(sc_counts~elevation+tree_cover+Distance_from_edge+offset(log(Sampling_Hours)),data=GLM_table, family="poisson"))
        # #Residual deviance: 143007  on 51  degrees of freedom
        # 143007/51 # 2804.059
summary(glm(sc_counts~elevation+tree_cover+Distance_from_edge+tree_height+slope+aspect+offset(log(Sampling_Hours)),data=GLM_table, family="poisson"))
#Residual deviance: 136489  on 48  degrees of freedom
136489/48 #2843.521
#ii) quasipoisson
        # summary(glm(sc_counts~elevation+tree_cover+Distance_from_edge+offset(log(Sampling_Hours)),data=GLM_table, family="quasipoisson"))
        #   # Coefficients:
        #   # Estimate Std. Error t value Pr(>|t|)  
        #   # (Intercept)        -0.2710935  1.2502370  -0.217   0.8292  
        #   # elevation           0.0029519  0.0013233   2.231   0.0301 *
        #   # tree_cover          0.0023810  0.0075979   0.313   0.7553  
        #   # Distance_from_edge  0.0004933  0.0002706   1.823   0.0741 .
        #   # #(Dispersion parameter for quasipoisson family taken to be 3161.617)
#diagnostic plots
plot(glm(sc_counts~elevation+tree_cover+Distance_from_edge+tree_height+slope+aspect+offset(log(Sampling_Hours)),data=GLM_table, family="quasipoisson"))
#running glm
summary(glm(sc_counts~elevation+tree_cover+Distance_from_edge+tree_height+slope+aspect+offset(log(Sampling_Hours)),data=GLM_table, family="quasipoisson"))
  # Estimate Std. Error t value Pr(>|t|)  
  # (Intercept)        -0.6003836  1.2994306  -0.462   0.6461  
  # elevation           0.0027745  0.0013466   2.060   0.0448 *
  # tree_cover         -0.0007649  0.0081687  -0.094   0.9258  
  # Distance_from_edge  0.0004624  0.0002806   1.648   0.1059  
  # tree_height         0.0518833  0.0411441   1.261   0.2134  
  # slope              -0.0023276  0.0088869  -0.262   0.7945  
  # aspect             -0.0006266  0.0009095  -0.689   0.4942  
  # ---
  #   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
  # 
  # (Dispersion parameter for quasipoisson family taken to be 3140.823)
            # #==========================
            # summary(glm(sc_counts~elevation+Distance_from_edge+offset(log(Sampling_Hours)),data=GLM_table, family="quasipoisson"))
            #   # Estimate Std. Error t value Pr(>|t|)  
            #   # (Intercept)        -0.0861100  1.0735976  -0.080   0.9364  
            #   # elevation           0.0029229  0.0013003   2.248   0.0289 *
            #   # Distance_from_edge  0.0005083  0.0002627   1.935   0.0585 .

#More mm and sc calls detected when microphone is at higher elevation, 
#more mm calls detected when microphone is further from edge

################################################################################



