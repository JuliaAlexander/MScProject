#Selecting clips to validate from across gibbon range

#============================================================
#Section 1 
##Loading 2021 predictions dataset

#1. Libraries
library(data.table) #eg fread, fwrite
library(tidyverse) #eg %>%, ggplot

#2. Loading 25cols data
#setwd
#w3_v2
raw_data  = list.files(pattern="*table_24cols_w3_v2.csv") %>% map_df(~fread(.))
dim(raw_data) ##118614775       25
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
###Section 2

#1. Finding row numbers for each category
gc_rows <- which(Audio_table$MT_gc=="gc")
mm_rows <- which(Audio_table$MT_mm=="mm")
sc_rows <- which(Audio_table$MT_sc=="sc")
background_rows <- which(Audio_table$MT_background=="background")


#2. random seed
set.seed(100)

#3. Selecting 50 clips from each category
gc_50r <- sample(gc_rows,50)
mm_50r <- sample(mm_rows,50)
sc_50r <- sample(sc_rows,50)
bg_50r <- sample(background_rows,50)


#4. Creating table of validating clip details
# raw table
Val_200T_raw <- Audio_table[c(gc_50r,mm_50r,sc_50r,bg_50r),]
##colnames(Val_200T_raw)
#adding row number column
Val_200T_raw$row_number <- c(gc_50r,mm_50r,sc_50r,bg_50r)


#5. Adding external drive file path
      ##table(Val_200T_raw$Folder) #China_side   U_Vietnam_Phase_1-2     Vietnam_Phase_3 
#a) Getting External drive folder name
for (i in 1:nrow(Val_200T_raw)){
  if(Val_200T_raw$Folder[i] == "U_Vietnam_Phase_1-2"){
    Val_200T_raw$Original_folder[i] <- "Phase_1-2"
  } else if(Val_200T_raw$Folder[i] == "Vietnam_Phase_3"){
    Val_200T_raw$Original_folder[i] <- "Phase_3"
  } else if (Val_200T_raw$Folder[i] == "China_side"){
    Val_200T_raw$Original_folder[i] <- "China_side"
  }
}
#b) Making file path column
Val_200T_raw$Path_original <- with(Val_200T_raw, paste(Original_folder,AudioMoth,WAV_name, sep="/"))

  
#6. Shuffling rows and removing prediction cols of Val table  
#Shuffling row order
Val_200T_shuffled <- Val_200T_raw[sample(1:nrow(Val_200T_raw)),]
    ###colnames(Val_200T_shuffled)
#Validation detail table - without model predictions
Val_200T_unpred <- Val_200T_shuffled[,c(1:3,13:28)]

# #saving tables
# write.csv(Val_200T_raw,"Validating_2021_dataset,_unshuffled,_gc_mm_sc_bg,_with_model_predictions.csv", row.names=F)
# write.csv(Val_200T_shuffled,"Validating_2021_dataset,_shuffled_with_model_predictions.csv", row.names=F)
# write.csv(Val_200T_unpred,"Validating_2021_dataset_details,_to_fill_in.csv", row.names=F)

#=========================================
# #Loading validation set
# Val_200T_unpred <- read.csv("Validating_2021_dataset_details,_to_fill_in.csv")

#7. Making validating clips
  ##adapting from Label_Clips_Making_v3.R
#a) Libraries
library(tuneR) #for readWave() and writeWave()
library(seewave) #for cutw()

#b) Extracting audio clips
	# #Looking at dataset
	# LongSoundFile_Names <- Val_200T_unpred[,"Path_original"]
	# BeginTime <- Val_200T_unpred[,"start_time"]
	# EndTime <- Val_200T_unpred[,"end_time"]
	# RowNumber <- Val_200T_unpred[,"row_number"]

#Setwd on external drive

##Repeats for each of 200 validating clips
for (i in 128:nrow(Val_200T_unpred)){
  #Read in audio file
  # LongSoundFile_i <- readWave(LongSoundFile_Names[i])
  LongSoundFile_i <- readWave(Val_200T_unpred$Path_original[i])
  
  #Isolating 5s segment
  # Clip_i <- cutw(LongSoundFile_i, from=BeginTime[i],
  #                to=EndTime[i], output="Wave")
  Clip_i <- cutw(LongSoundFile_i, from=Val_200T_unpred$start_time[i],
                 to=Val_200T_unpred$end_time[i], output="Wave")
  
  #Saving audio segment to folder
  #non-extensble format
  writeWave(Clip_i,
            filename=paste0('Validating_clips/',i,'_clip_',Val_200T_unpred$row_number[i],'_row.WAV'),
            extensible = FALSE)

  #Progress message
  if(i %% 10 == 0){cat(paste0("clip number ",i, '\n'))}
  ###if(i %% 1 == 0){cat(paste0("clip number ",i, '\n'))}
}

#=======================

#8. Making spectrogram
#a) Library
library(av)  #for read_audio_fft()
  #Using av package, https://github.com/ropensci/av/tree/7c9ad17853c644edd0663b943fed9be4ab7a8ad3
#Adapting from Spectrograms_4groups_w2


#b) spectrograms for validating dataset 
File_paths <- Val_200T_unpred$Path_original
Row_numbers <- Val_200T_unpred$row_number
Start_s <- Val_200T_unpred$start_time
End_s <- Val_200T_unpred$end_time

#making pdf
pdf("Validating_spectrograms/1st_200pages,_all_validating_clips,_original_5s_audio.pdf", paper="A4r")
for (i in 1:nrow(Val_200T_unpred)){
# pdf("Validating_spectrograms/1st_2pages,_all_validating_clips,_original_5s_audio.pdf", paper="A4r")
# for (i in 1:2){
  clip_i <- read_audio_fft(File_paths[i], start_time=Start_s[i], end_time=End_s[i])
  plot(clip_i, dark = FALSE)
  mtext(paste0(i,"th_", Row_numbers[i],'_row'), side = 3, cex=2)
  if(i %% 10 == 0){print(paste0("i=",i))}
}
dev.off()


#=============================================================================================



