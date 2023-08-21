#Bubble map t2
  #counts per 30 mins of sampling

#1. library
library(sf)
library(raster)
library(ggplot2)

#2. load dataset
#setwd
counts_by_site <- read.csv("../data/C1_GLM_table_19cols.csv")

#3. Counts per 30 min of sampling
#mm
counts_by_site$mm_count_30m <- counts_by_site$mm_counts/(counts_by_site$Sampling_Hours*2)
counts_by_site$mm_count_30m
    # [1] 2.52868611 1.51939710 2.40597800 2.79074640 1.36628491 8.66193018 1.43338870 1.46921642
    # [9] 0.09119638 1.13070400 2.49770732 2.76305373 0.64953476 2.25789679 2.92284627 0.66033465
    # [17] 3.24097273 2.97992070 2.64395217 3.54656066 1.63460380 3.80083775 4.11698932 3.89178247
    # [25] 2.06793940 3.47842495 2.55580848 1.34895237 1.90355782 2.62051962 1.92033853 1.20461325
    # [33] 0.44449599 1.61822296 1.57055295 0.60924528 1.08356774 2.21823209 0.68231543 1.57223447
    # [41] 4.51604495 0.64185481 0.87446326 0.67379955 0.61458488 0.18510282 0.07661096 0.54998284
    # [49] 0.08869404 1.05760276 1.89873418 2.25168531 1.14324680 2.11804633 1.23210459
#sc
counts_by_site$sc_count_30m <- counts_by_site$sc_counts/(counts_by_site$Sampling_Hours*2)
counts_by_site$sc_count_30m
    # [1]  4.7451132  3.7104851  4.6346310  5.6778699  5.0773558  0.7465071  1.0166591  3.0163913
    # [9]  0.1436343  7.9028133  7.9077689  9.9601470  2.9988261 10.4226983  7.4997578  5.9816574
    # [17] 10.0041268  7.7970857  4.0517494  7.1533550  7.0098553 16.9630440 34.4359811 16.3886722
    # [25]  6.7998072 10.5537351  8.1382919  6.4733750  4.7054238 14.6107034  9.6615915  6.4234468
    # [33]  2.1189672  4.0134597  6.1095485  4.5853292  4.6530176  9.8412182  6.0950459  8.3015718
    # [41]  9.0304142  2.8853790  6.0205419  4.6642783  2.7518211  3.6863227  3.6211446  2.2547438
    # [49]  1.0021598  8.9610541  3.8319908  3.4800868  1.5675807  5.1520046  6.4599622
#mm and sc
counts_by_site$mm_sc_count_30m <- (counts_by_site$mm_counts+counts_by_site$sc_counts)/(counts_by_site$Sampling_Hours*2)
counts_by_site$mm_sc_count_30m
    # [1]  7.2737993  5.2298822  7.0406090  8.4686163  6.4436407  9.4084373  2.4500478  4.4856077
    # [9]  0.2348307  9.0335173 10.4054762 12.7232007  3.6483608 12.6805951 10.4226041  6.6419920
    # [17] 13.2450995 10.7770064  6.6957016 10.6999157  8.6444591 20.7638817 38.5529704 20.2804546
    # [25]  8.8677466 14.0321601 10.6941003  7.8223273  6.6089817 17.2312230 11.5819300  7.6280601
    # [33]  2.5634632  5.6316827  7.6801015  5.1945745  5.7365854 12.0594503  6.7773613  9.8738062
    # [41] 13.5464591  3.5272338  6.8950051  5.3380778  3.3664060  3.8714255  3.6977556  2.8047267
    # [49]  1.0908538 10.0186569  5.7307250  5.7317721  2.7108275  7.2700509  7.6920668
#as sf object
sf_counts_by_site <- st_as_sf(counts_by_site, coords = c('X','Y'), crs=32648)

#4. load forest outline
## spatial data files
#a) vector data
forest_outline <- st_read("../data/Spatial_layers/c_Outline_forest_shapefile_TrungKhanh_forestblock_smooth.gpkg")
Vietnam_China_border <- st_read("../data/Spatial_layers/d_International boundary,_forest_crosses_Vietnam_China_border")
Vietnam_protected_area <- st_read("../data/Spatial_layers/e_Vietnamese_protected_area,_Cao vit gibbon conservation area")
China_protected_area <- st_read("../data/Spatial_layers/f_China_protected_area,_Bangliang National Nature Reserve")
# #b) raster data
# elevation <- rast("g_Digital_elevation_model,_ALOS PALSAR_clip.tif")
# tree_cover <- rast("h_tree cover,_from_Hansen_2013.tif")
# land_type <- rast("i_landcover_made_by_Ollie,_pixel_value_1_is_forest,_REMAP TK landcover_proj_updated_v2.tif")
# tree_height <- rast("m_tree height.tif")
# slope <- rast("n_ALOS Palsar slope.tif")
#Cropping Vietnam China border
crop_VC_border <- st_crop(Vietnam_China_border, forest_outline)

#5. bubble map - combined mm and sc
#with VC border
ggplot()+
  geom_sf(data=forest_outline, fill="grey90")+
  geom_sf(data=crop_VC_border, alpha=0.55)+
  theme_bw()+
  geom_sf(data=sf_counts_by_site, aes(size=mm_sc_count_30m),
          color="deepskyblue2", alpha=0.5)+
  scale_size_area(max_size = 12)+
  labs(x="Longitude", y="Latitude", size="Counts per 30 mins")
ggsave("../results/C2_15a.Bubble_map_mm_plus_sc_count_per_sampling_30_mins,_area_size,_VC_border.png", width=9, height=6)

#saving table
write.csv(sf_counts_by_site, "../results/C3_Counts_by_site_per_30mins_table.csv", row.names = F)




