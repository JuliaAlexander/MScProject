#Looking at tree height and slope data
    #adapting from 3rd

#1. Libraries for vector and raster data
library(sf) #vector GIS package
library(terra) #raster GIS package
library(tidyverse) #needed for loading readxls library
library(readxl) #loading xls metadata

#2. Loading spatial data files
#a) vector data
forest_outline <- st_read("c_Outline_forest_shapefile_TrungKhanh_forestblock_smooth.gpkg")
Vietnam_China_border <- st_read("d_International boundary,_forest_crosses_Vietnam_China_border")
Vietnam_protected_area <- st_read("e_Vietnamese_protected_area,_Cao vit gibbon conservation area")
China_protected_area <- st_read("f_China_protected_area,_Bangliang National Nature Reserve")
#b) raster data
elevation <- rast("g_Digital_elevation_model,_ALOS PALSAR_clip.tif")
tree_cover <- rast("h_tree cover,_from_Hansen_2013.tif")
land_type <- rast("i_landcover_made_by_Ollie,_pixel_value_1_is_forest,_REMAP TK landcover_proj_updated_v2.tif")
tree_height <- rast("m_tree height.tif")
slope <- rast("n_ALOS Palsar slope.tif")
#c) csv data - AudioMoth coordinates
Main_metadata <- read_excel("k_2021_AudioMoth deployment locations all_withfoldernames.xls")
Prel_metadata <- read_excel("l_2020_Preliminary AudioMoth deployment.xls")
      #raw_metadata <- read.csv("a_AudioMoth deployment locations all.csv")
#converting data frame to an sf object
      #AudioMoth_coordinates <- st_as_sf(raw_metadata, coords = c('X','Y'), crs=32648)
Main_coordinates <- st_as_sf(Main_metadata, coords = c('X','Y'), crs=32648)
Prel_coordinates <- st_as_sf(Prel_metadata, coords = c('X','Y'), crs=32648)
detach(package:tidyr) ##to avoid terra extract function being masked by tidyr

#3. Examining tree height and slope layers
#a) resolution
tree_height
    # class       : SpatRaster 
    # dimensions  : 790, 1407, 1  (nrow, ncol, nlyr)
    # resolution  : 28, 28  (x, y)
    # extent      : 632647.3, 672043.3, 2522752, 2544872  (xmin, xmax, ymin, ymax)
    # coord. ref. : WGS 84 / UTM zone 48N (EPSG:32648) 
    # source      : m_tree height.tif 
    # name        : m_tree height 
##tree_cover and land_type also have 28, 28 resolution
slope
  # class       : SpatRaster 
  # dimensions  : 2484, 3190, 1  (nrow, ncol, nlyr)
  # resolution  : 12.5, 12.5  (x, y)
  # extent      : 632191.9, 672066.9, 2518475, 2549525  (xmin, xmax, ymin, ymax)
  # coord. ref. : WGS 84 / UTM zone 48N (EPSG:32648) 
  # source      : n_ALOS Palsar slope.tif 
  # name        : n_ALOS Palsar slope 
##elevation also has 12.5 , 12.5 resolution
      # #b) plot full raster #not saved
      # plot(tree_height, main="Tree height")
      # plot(slope, main="Slope")
      # #c) cropping raster layer to forest outline extent
      # small_tree_height <- crop(tree_height, forest_outline)
      # small_slope <- crop(slope, forest_outline)
      # #d) plotting cropped raster with forest outline
      # pdf("../17a)_Tree_height_2021_sites_forest_outline_map.pdf")
      # plot(small_tree_height)
      # plot(st_geometry(forest_outline), axes=TRUE, add=TRUE)
      # plot(st_geometry(Main_coordinates), axes=TRUE, add=TRUE, col="blue", pch=1, lwd=1)
      # ##plot(st_geometry(Prel_coordinates), axes=TRUE, add=TRUE, col="black", pch=1, lwd=1)
      # dev.off()
      # pdf("../17b)_Slope_2021_sites_forest_outline_map.pdf")
      # plot(small_slope)
      # plot(st_geometry(forest_outline), axes=TRUE, add=TRUE)
      # plot(st_geometry(Main_coordinates), axes=TRUE, add=TRUE, col="blue", pch=1, lwd=1)
      # ##plot(st_geometry(Prel_coordinates), axes=TRUE, add=TRUE, col="black", pch=1, lwd=1)
      # dev.off()

#e) terrain function
#slope aspect
SA_layer <- terrain(elevation, v=c('slope', 'aspect'))
    # class       : SpatRaster 
    # dimensions  : 2484, 3190, 2  (nrow, ncol, nlyr)
    # resolution  : 12.5, 12.5  (x, y)
    # extent      : 632191.9, 672066.9, 2518475, 2549525  (xmin, xmax, ymin, ymax)
    # coord. ref. : WGS 84 / UTM zone 48N (EPSG:32648) 
    # source      : memory 
    # names       :    slope, aspect 
    # min values  :  0.00000,      0 
    # max values  : 79.97946,    360 
SA_layer$slope
#...
# min value   :  0.00000 
# max value   : 79.97946 
SA_layer$aspect
#...
# min value   :      0 
# max value   :    360 
##Aspect - 0 is north facing, 90 is east facing, 180 is south facing, 270 is west facing

      # #f) plot cropped aspect
      # small_aspect <- crop(SA_layer$aspect, forest_outline)
      # pdf("../17c)_Aspect_2021_sites_forest_outline_map.pdf")
      # plot(small_aspect)
      # plot(st_geometry(forest_outline), axes=TRUE, add=TRUE)
      # plot(st_geometry(Main_coordinates), axes=TRUE, add=TRUE, col="blue", pch=1, lwd=1)
      # ##plot(st_geometry(Prel_coordinates), axes=TRUE, add=TRUE, col="black", pch=1, lwd=1)
      # dev.off()

#--------------------------------------------------
#4. Extracting raster values at sites
  #Adapting 1st method
  #https://www.gisremotesensing.com/2012/10/extract-raster-values-from-points.html
#a) Extracting raster values by point
#i) 2021 sites
  ####tidyverse loads extract() too, so need to specify terra extract method
  # Main_RasVal_r28 <- terra::extract(c(tree_cover, land_type, tree_height),Main_metadata[,c(8,9)])
Main_RasVal_r28 <- extract(c(tree_cover, land_type, tree_height), Main_metadata[,c(9,10)])
colnames(Main_RasVal_r28) <- c("ID","tree_cover", "land_type", "tree_height")
Main_RasVal_r12.5 <- extract( c(elevation, slope, SA_layer$aspect), Main_metadata[,c(9,10)])
colnames(Main_RasVal_r12.5) <- c("ID","elevation", "slope","aspect")
#ii) 2020 sites
Prel_RasVal_r28 <- extract(c(tree_cover, land_type, tree_height), Prel_metadata[,c(4,5)])
colnames(Prel_RasVal_r28) <- c("ID","tree_cover", "land_type", "tree_height")
Prel_RasVal_r12.5 <- extract( c(elevation, slope, SA_layer$aspect), Prel_metadata[,c(4,5)])
colnames(Prel_RasVal_r12.5) <- c("ID","elevation", "slope","aspect")

#b) Combining raster values with points
Main_RasVal_metadata <-  data.frame(Main_metadata[,c(1:4,9:10)],
                                    Main_RasVal_r12.5[,c(2,3,4)],
                                    Main_RasVal_r28[,c(2,3,4)])
Prel_RasVal_metadata <-  data.frame(Prel_metadata[,c(1:5)],
                                    Prel_RasVal_r12.5[,c(2,3,4)],
                                    Prel_RasVal_r28[,c(2,3,4)])
#c) Subset of unique locations for 2021 sites
nrow(unique(Main_RasVal_metadata[,c(3,5:6)] )) #55 unique site
AD_Main_Coord_Ras_table <- unique(Main_RasVal_metadata[,c(2:3,5:12)] )

            #d) saving csv
            # write.csv(AD_Main_Coord_Ras_table, "../17d)_Summary_2021_metadata_55coordinates_raster,_incl_treeheight_slope,_aspect.csv", row.names = F)
            # write.csv(Prel_RasVal_metadata, "../17e)_Summary_2020_metadata_18rows_raster,_incl_treeheight_slope,_aspect.csv", row.names = F)


#5. Adding protected area column
#a) Number of AudioMoths in Chinese protected area
PA_China_11AD <- Main_coordinates[China_protected_area,]
#b) Number of AudioMoths in Vietnamese protected area
PA_Vietnam_44AD <- Main_coordinates[Vietnam_protected_area,]
## all 2020 sites are in Vietnam
Prel_coordinates[China_protected_area,] #0 features
Prel_PA_Vietnam <- Prel_coordinates[Vietnam_protected_area,] #18 features
#c) Adding protected area column
Main_PA_Coord_Ras_table <- data.frame(AD_Main_Coord_Ras_table)
Main_PA_Coord_Ras_table$Protected_area <- NA
for (i in (1:nrow(Main_PA_Coord_Ras_table))){
  if(Main_PA_Coord_Ras_table[i,1] %in% c(PA_Vietnam_44AD$StationID)){
    Main_PA_Coord_Ras_table$Protected_area[i] <- "Vietnam"
  }
  else if(Main_PA_Coord_Ras_table[i,1] %in% c(PA_China_11AD$StationID)){
    Main_PA_Coord_Ras_table$Protected_area[i] <- "China"
  }
}
#2020 sites
Prel_PA_Coord_Ras <- data.frame(Prel_RasVal_metadata)
Prel_PA_Coord_Ras$Protected_area <- NA
for (i in (1:nrow(Prel_PA_Vietnam))){
  if(Prel_PA_Coord_Ras[i,1] %in% c(Prel_PA_Vietnam$StationID)){
    Prel_PA_Coord_Ras$Protected_area[i] <- "Vietnam"
  }
}

        # #d) saving csv
        # write.csv(Main_PA_Coord_Ras_table, "../17d)_Summary_2021_metadata_55coordinates_raster_PA,_incl_treeheight_slope,_aspect.csv", row.names = F)
        # write.csv(Prel_PA_Coord_Ras, "../17e)_Summary_2020_metadata_18rows_raster_PA,_incl_treeheight_slope,_aspect.csv", row.names = F)


#6. Examining 500m from forest
 ###From 2nd 7) Alternative method

#a) Adding 500m inside forest outline
m2_forest_centre <- st_buffer(st_geometry(forest_outline), -500)
plot(m2_forest_centre, col="red")

#b) making forest edge
m2_forest_edge <- st_difference(forest_outline, m2_forest_centre)
plot(st_geometry(m2_forest_edge))
#plot
plot(st_geometry(m2_forest_edge), col="blue", axes=T)
plot(st_geometry(m2_forest_centre), col="red", add=T)
#plot other way round
plot(st_geometry(forest_outline), axes=T)
plot(st_geometry(m2_forest_centre), col="red", axes=T, add=T)
plot(st_geometry(m2_forest_edge), col="blue", axes=T, add=T)

#c) Coordinates in edge and centre polygons
Main_border <- Main_coordinates[m2_forest_edge,]
    # Simple feature collection with 8 features and 8 fields
    # Geometry type: POINT
    # Dimension:     XY
    # Bounding box:  xmin: 654799 ymin: 2533660 xmax: 657984.9 ymax: 2537131
    # Projected CRS: WGS 84 / UTM zone 48N
    # # A tibble: 8 × 9
    # DeploymentID   StationID Folder nam…¹ Phase Start               End                 Hours Effort
    # <chr>          <chr>     <chr>        <chr> <dttm>              <dttm>              <dbl>  <dbl>
    #   1 AU21_Phase 1-2 AU21      AU21         Phas… 2021-10-22 04:30:00 2021-11-28 11:00:20 257.   37.3 
    # 2 AU21_Phase 3   AU21      AU21         Phas… 2021-12-12 04:30:00 2022-01-21 07:00:02 304.   40.1 
    # 3 AU25_Phase 1-2 AU25      AU25         Phas… 2021-10-18 11:05:44 2021-11-26 11:00:20 293.   39.0 
    # 4 AU25_Phase 3   AU25      AU25         Phas… 2021-12-08 11:00:20 2022-01-07 04:30:00 220.   29.7 
    # 5 AU26_Phase 1-2 AU26      AU26         Phas… 2021-10-22 09:00:10 2021-12-01 10:00:15 299.   40.0 
    # 6 AU26_Phase 3   AU26      AU26         Phas… 2021-12-09 09:00:10 2022-01-20 04:30:00 312.   41.8 
    # 7 AU41_Phase 3   AU41      AU41 file f… Phas… 2021-12-20 10:00:15 2021-12-24 08:00:07  29.0   3.92
    # 8 LP26_Phase 3   LP26      LP26 listen… Phas… 2021-12-17 07:00:02 2022-01-15 05:30:05 217.   28.9 
    # # … with 1 more variable: geometry <POINT [m]>, and abbreviated variable name ¹​`Folder name`
Main_centre <- Main_coordinates[m2_forest_centre,]
#2020 sites
Prel_centre <- Prel_coordinates[m2_forest_centre,]
    # Simple feature collection with 17 features and 7 fields
    # Geometry type: POINT
    # Dimension:     XY
    # Bounding box:  xmin: 656288.5 ymin: 2534273 xmax: 657452 ymax: 2535121
    # Projected CRS: WGS 84 / UTM zone 48N
    # # A tibble: 17 × 8
Prel_border <- Prel_coordinates[m2_forest_edge,]
    # Simple feature collection with 1 feature and 7 fields
    # Geometry type: POINT
    # Dimension:     XY
    # Bounding box:  xmin: 657431.7 ymin: 2535433 xmax: 657431.7 ymax: 2535433
    # Projected CRS: WGS 84 / UTM zone 48N
    # # A tibble: 1 × 8
    # StationID AudioMothID Bearing Start End   Hours Effort           geometry
    # <chr>           <dbl>   <dbl> <lgl> <lgl> <lgl> <lgl>         <POINT [m]>
    #   1 D10                18     200 NA    NA    NA    NA     (657431.7 2535433)


#d) Adding forest edge column
#500m from edge
#2021 ites
Main_edge_table <- data.frame(Main_PA_Coord_Ras_table)
Main_edge_table$Forest_500m_region <- NA
for (i in (1:nrow(Main_edge_table))){
  if(Main_edge_table[i,1] %in% c(Main_border$StationID)){
    Main_edge_table$Forest_500m_region[i] <- "Border"
  }
  else if(Main_edge_table[i,1] %in% c(Main_centre$StationID)){
    Main_edge_table$Forest_500m_region[i] <- "Centre"
  }
}
###Main_edge_table <- Coord_edge_table

#2020 sites
Prel_edge_table <- data.frame(Prel_PA_Coord_Ras)
Prel_edge_table$Forest_500m_region <- NA
for (i in (1:nrow(Prel_edge_table))){
  if(Prel_edge_table[i,1] %in% c(Prel_border$StationID)){
    Prel_edge_table$Forest_500m_region[i] <- "Border"
  }
  else if(Prel_edge_table[i,1] %in% c(Prel_centre$StationID)){
    Prel_edge_table$Forest_500m_region[i] <- "Centre"
  }
}

    # #e) saving csv
    # write.csv(Main_edge_table, "../17d)_Summary_2021_metadata_55coordinates_raster_PA,_incl_treeheight_slope,_aspect.csv", row.names = F)
    # write.csv(Prel_edge_table, "../17e)_Summary_2020_metadata_18rows_raster_PA,_incl_treeheight_slope,_aspect.csv", row.names = F)


#f) 2020 sites Edge - Plot with station IDs
pdf("../17f)_Forest_Edge_2020_station_ID,_forest_outline_map.pdf")
plot(st_geometry(forest_outline), axes=TRUE, col="blue")
plot(st_geometry(China_protected_area), axes=TRUE, add=TRUE, col="darkgreen")
plot(st_geometry(Vietnam_protected_area), axes=TRUE, add=TRUE, col="red")
plot(st_cast(st_geometry(forest_outline), 'LINESTRING'), axes=TRUE, add=TRUE, col="blue", lwd=2)
plot(st_geometry(Vietnam_China_border), axes=TRUE,add=TRUE, lwd=2)
text(as.data.frame(st_coordinates(Prel_centre))$X,
     as.data.frame(st_coordinates(Prel_centre))$Y, 
     labels=st_drop_geometry(Prel_centre)$StationID, col="yellow", cex=0.9)
text(as.data.frame(st_coordinates(Prel_border))$X,
     as.data.frame(st_coordinates(Prel_border))$Y, 
     labels=st_drop_geometry(Prel_border)$StationID, col="pink", cex=0.9)
dev.off()


#7. Finding distance from sites to forest edge
#a) Examining for first site row
min(st_distance(st_geometry(Main_border)[1], 
                st_cast(st_geometry(forest_outline), 'POINT')))
##382.061 [m]
min(st_distance(st_geometry(Main_border)[1], 
                st_cast(st_geometry(forest_outline), 'LINESTRING')))
##382.061 [m]

#b) Finding min distance to edge
    # Main_min_dist_edge <- c()
    # for (i in 1:nrow(Main_coordinates)){
    #   distances_i <- st_distance(st_geometry(Main_coordinates)[i], 
    #                              st_cast(st_geometry(forest_outline), 'LINESTRING'))
    #   Main_min_dist_edge <- c(Main_min_dist_edge, min(distances_i))
    #   #print(min_dist)
    # }
    # Main_min_dist_edge
    #     # [1] 1314.49971 1036.18585  878.08644 1019.07563 1741.83252 1419.16201 1197.32057  914.90031
    #     # [9]  713.13485 1260.80432 1076.26622 1596.71332 1596.71332 1525.02526 1525.02526 1147.19621
    #     # [17] 1147.19621  896.45960  896.45960  697.70474  697.70474 1092.50391 1092.50391  837.09237
    #     # [25]  837.09237  742.61639  742.61639  604.28548  604.28548  934.23447  934.23447 1436.57567
    #     # [33] 1436.57567 1198.89648 1198.89648 1027.13499 1027.13499 1258.86197 1258.86197 1194.70001
    #     # [41] 1194.70001  638.48527  638.48527 1049.39564 1049.39564  630.82751  630.82751  689.04400
    #     # [49]  689.04400  382.06103  382.06103  544.29476  544.29476  731.31493  731.31493  596.14342
    #     # [57]  596.14342  231.25197  231.25197  467.73760  467.73760  911.35672  911.35672 1532.33573
    #     # [65] 1532.33573  784.11176  784.11176 1307.97956 1307.97956 1195.79915 1195.79915  777.53485
    #     # [73]  777.53485 1499.99235 1499.99235  674.93047  674.93047 1412.43267 1412.43267  546.19594
    #     # [81]  546.19594 1097.77051 1097.77051  537.09123  537.09123 1267.08530 1267.08530  914.97934
    #     # [89]  914.97934  322.33429   57.88121  727.03116  760.10430  760.10430  841.65500  841.65500

##2021 border
Main_min_dist_border <- c()
for (i in 1:nrow(Main_border)){
  distances_i <- st_distance(st_geometry(Main_border)[i], 
                             st_cast(st_geometry(forest_outline), 'LINESTRING'))
  Main_min_dist_border <- c(Main_min_dist_border,min(distances_i))
  #print(min_dist)
}
Main_min_dist_border
##LINESTRNG: 382.06103 382.06103 231.25197 231.25197 467.73760 467.73760 322.33429  57.88121
##POINT:  382.06103 382.06103 231.25197 231.25197 467.73760 467.73760 322.33429  57.88121

#2021 centre
Main_min_dist_centre <- c()
for (i in 1:nrow(Main_centre)){
  distances_i <- st_distance(st_geometry(Main_centre)[i], 
                             st_cast(st_geometry(forest_outline), 'LINESTRING'))
  Main_min_dist_centre <- c(Main_min_dist_centre,min(distances_i))
  #print(min_dist)
}
Main_min_dist_centre
    # [1] 1314.4997 1036.1859  878.0864 1019.0756 1741.8325 1419.1620 1197.3206  914.9003  713.1349
    # [10] 1260.8043 1076.2662 1596.7133 1596.7133 1525.0253 1525.0253 1147.1962 1147.1962  896.4596
    # [19]  896.4596  697.7047  697.7047 1092.5039 1092.5039  837.0924  837.0924  742.6164  742.6164
    # [28]  604.2855  604.2855  934.2345  934.2345 1436.5757 1436.5757 1198.8965 1198.8965 1027.1350
    # [37] 1027.1350 1258.8620 1258.8620 1194.7000 1194.7000  638.4853  638.4853 1049.3956 1049.3956
    # [46]  630.8275  630.8275  689.0440  689.0440  544.2948  544.2948  731.3149  731.3149  596.1434
    # [55]  596.1434  911.3567  911.3567 1532.3357 1532.3357  784.1118  784.1118 1307.9796 1307.9796
    # [64] 1195.7992 1195.7992  777.5349  777.5349 1499.9924 1499.9924  674.9305  674.9305 1412.4327
    # [73] 1412.4327  546.1959  546.1959 1097.7705 1097.7705  537.0912  537.0912 1267.0853 1267.0853
    # [82]  914.9793  914.9793  727.0312  760.1043  760.1043  841.6550  841.6550

#2020 centre
min_dist_funct <- function(Sites){
  min_dist_values <- c()
  for (i in 1:nrow(Sites)){
    distances_i <- st_distance(st_geometry(Sites)[i], 
                               st_cast(st_geometry(forest_outline), 'LINESTRING'))
    min_dist_values <- c(min_dist_values,min(distances_i))
    #print(min_dist)
  }
  return(min_dist_values)
}
min_dist_funct(Prel_centre)
# [1]  699.3024  699.3024  939.7583  939.7583  903.4203  903.4203 1136.6969 1136.6969 1327.4074
# [10] 1526.6207 1526.6207 1476.4728 1476.4728 1236.9416 1236.9416 1529.5028 1529.5028

#2020 border
min_dist_funct(Prel_border) #445.1877


#c) adding distance column - from edge
#2021 table
Main_dist_table <- data.frame(Main_edge_table)
Main_dist_table$Distance_from_edge <- NA
for (i in (1:nrow(Main_dist_table))){
  #Coord_dist_table$Forest_edge_500m[i] <- "Border"
  point_i <- st_as_sf(Main_dist_table[i,], coords = c('X','Y'), crs=32648)
  distances_i <- st_distance(st_geometry(point_i), 
                             st_cast(st_geometry(forest_outline), 'LINESTRING'))
  #min_dist_centre <- c(min_dist_centre,min(distances_i))
  Main_dist_table$Distance_from_edge[i] <- min(distances_i)
}
#2020 table
Prel_dist_table <- data.frame(Prel_edge_table)
Prel_dist_table$Distance_from_edge <- NA
for (i in (1:nrow(Prel_dist_table))){
  #Coord_dist_table$Forest_edge_500m[i] <- "Border"
  point_i <- st_as_sf(Prel_dist_table[i,], coords = c('X','Y'), crs=32648)
  distances_i <- st_distance(st_geometry(point_i), 
                             st_cast(st_geometry(forest_outline), 'LINESTRING'))
  #min_dist_centre <- c(min_dist_centre,min(distances_i))
  Prel_dist_table$Distance_from_edge[i] <- min(distances_i)
}

#d) saving csv
write.csv(Main_dist_table, "..data/A2_Main_17d)_Summary_2021_metadata_55coordinates.csv", row.names = F)
write.csv(Prel_dist_table, "../data/A1_Prel_17e)_Summary_2020_metadata_18rows.csv", row.names = F)


    # #8. Examining spatial variables
    # plot(Main_dist_table$tree_cover, Main_dist_table$tree_height)
    # plot(Main_dist_table$slope, Main_dist_table$tree_height)




