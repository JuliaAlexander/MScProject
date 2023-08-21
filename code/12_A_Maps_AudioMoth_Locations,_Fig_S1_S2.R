#Map of 2020 2021 AudioMoths

#Creating supplementary figures S1 and S2

#1. Libraries for vector and raster data
library(sf) #vector GIS package
library(terra) #raster GIS package
library(ggplot2) #for map plotting
#setwd

#2. Loading spatial data files
#a) vector data
forest_outline <- st_read("../data/Spatial_layers/c_Outline_forest_shapefile_TrungKhanh_forestblock_smooth.gpkg")
Vietnam_protected_area <- st_read("../data/Spatial_layers/e_Vietnamese_protected_area,_Cao vit gibbon conservation area")
China_protected_area <- st_read("../data/Spatial_layers/f_China_protected_area,_Bangliang National Nature Reserve")
#b) China protetcted area inside forest
China_PArea_inside_forest <- st_intersection(forest_outline,China_protected_area)


#3. Getting coordinates
#a) Loading csv with coordinates
Prel_csv <- read.csv("../data/A1_Prel_17e)_Summary_2020_metadata_18rows.csv")
Main_csv <- read.csv("../data/A2_Main_17d)_Summary_2021_metadata_55coordinates.csv")
#b) converting dataframe to sf
Prel_coordinates <- st_as_sf(Prel_csv, coords = c('X','Y'), crs=32648)
Main_coordinates <- st_as_sf(Main_csv, coords = c('X','Y'), crs=32648)


#4. Map - 2020 and 2021 station ID

#a) 2020 sites - China and Vietnam protected part of forest - Figure S1
#yellow
ggplot()+
  geom_sf(data=forest_outline, fill="darkblue")+
  theme_bw()+
  geom_sf(data=China_PArea_inside_forest, fill="darkgreen")+
  geom_sf(data=Vietnam_protected_area, fill="red")+
  #geom_sf_text(data=Main_coordinates, aes(label=StationID), color="yellow", size=2.5)+
  geom_sf_text(data=Prel_coordinates, aes(label=StationID), color="yellow", size=2.5)+
  labs(x="Longitude", y="Latitude")+
  theme(axis.title = element_text(size=10))
ggsave("../results/A3_18d.StationID_2020_map,_blue_forest_green_China_red_Vietnam,_yellow_10sites.png", width=9, height=6)

#b) 2021 sites - China and Vietnam protected part of forest - Figure S2
ggplot()+
  # geom_sf(data=forest_outline, fill="grey90")+
  geom_sf(data=forest_outline, fill="darkblue")+
  # geom_sf(data=crop_VC_border, alpha=0.55)+
  theme_bw()+
  geom_sf(data=China_PArea_inside_forest, fill="darkgreen")+
  geom_sf(data=Vietnam_protected_area, fill="red")+
  #geom_sf(data=crop_VC_border, alpha=0.65)
  #geom_sf_text(data=Prel_coordinates, aes(label=StationID), color="blue")+
  geom_sf_text(data=Main_coordinates, aes(label=StationID), color="yellow", size=2.5)+
  #geom_sf_text(data=Prel_coordinates, aes(label=StationID), color="pink", size=2.5)+
  labs(x="Longitude", y="Latitude")+
  theme(axis.title = element_text(size=10))
ggsave("../results/A4_18b.StationID_2021_map,_blue_forest_green_China_red_Vietnam,_yellow_55sites.png", width=9, height=6)


