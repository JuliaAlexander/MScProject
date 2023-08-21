#Examining 2021 P95 predictions
#China and Vietnam


#1st Libraries
library(data.table) #eg fread
library(tidyverse) #eg %>%, ggplot
library(hms) #eg as_hms()
library(lubridate) #eg with_tz, seconds, parse_date_time
#setwd


#2nd Function for making table
make_table_func <- function(input_data){
  #3. Tidying table
  #a) Reorder rows by filename
  TM_ordered <- input_data[order(input_data$file),]
  #b) Removing rows lacking scores #eg WAV file had 0 bytes
  #removing 1st column with unordered numbers
  TM_table <- TM_ordered[-which(is.na(TM_ordered$start_time)), -"V1"]
  # nrow(TM_crop) #18614775
  # nrow(TM_raw) - nrow(TM_crop) #2155 rows removed
  rm(input_data, TM_ordered)
  #c) Adding columns from file path
  TM_table[,c("Folder", "AudioMoth", "WAV_name"):= tstrsplit(file, "/")]
  
  
  #4. Examining timing of clips
  #a) getting raw timing from filenames #removing .csv part
  file_date_time_raw <- tstrsplit(TM_table$WAV_name, ".WAV")[[1]]
  ##file_date_time_raw <- str_split_fixed(TM_table$WAV_name, ".WAV",2)[,1]
  #b) formatting to time object
  file_date_time_UTC <- parse_date_time(file_date_time_raw, "ymdHMS")
  #c) converting to Vietnam time zone (+7 hours)
  file_date_time_V7 <- with_tz(file_date_time_UTC,  "Asia/Ho_Chi_Minh")
  #d) finding start timing of each clip
  TM_start_seconds <- seconds(TM_table$start_time)
  ##gc()
  clip_start_V7 <- file_date_time_V7 + TM_start_seconds
  #e) finding end timing of each clip
  TM_end_seconds <-  seconds(TM_table$end_time)
  clip_end_V7 <- file_date_time_V7 + TM_end_seconds
  
  #5. Add timings columns
  #a) date at start of each WAV file
  TM_table$WAV_date <- as_date(file_date_time_V7)
  #b) time at start of each WAV file in Vietnam time zone
  TM_table$WAV_time <- as_hms(file_date_time_V7)
  #c) time between start of file and start of clip
  TM_table$start_gap <- as_hms(TM_table$start_time)
  #d) start time of each clip
  TM_table$clip_start_clock <- as_hms(clip_start_V7)
  #e) end time of each clip
  TM_table$clip_end_clock <- as_hms(clip_end_V7)
  #f) starting hour of each clip
  TM_table$clip_start_hour <- hour(clip_start_V7)
  
  #6. Adding per 1 min and 5 min columns
  #a) Scores grouped by 1 min
  #Number of minutes between clip start and file start
  TM_table$start_gap_1min <- minute(as_hms(TM_table$start_gap))
  
  #b) Scores grouped by 5 min
  #Multiples of 5
  Fives <- c(1:12)*5 #[1]  5 10 15 20 25 30 35 40 45 50 55 60
  #Grouping each minute into 5 minutes
  start_1min_vector <- TM_table$start_gap_1min
  Five_min_group <- c()
  for(i in 1:length(Fives)){
    for(j in 1:length(start_1min_vector)){
      if(start_1min_vector[j]>=(Fives[i]-5) & start_1min_vector[j]<Fives[i]){
        Five_min_group[j] <- Fives[i]-5
      }
    }
  }
  TM_table$start_5min <- Five_min_group
  
  #c) Grouped by 55 mins
  Fiftyfive_min_group <- c()
  for(j in 1:length(start_1min_vector)){
    if(start_1min_vector[j]<55){
      Fiftyfive_min_group[j] <- "0"
    } else{Fiftyfive_min_group[j] <- "9999"} #Not in 55min group
  }
  TM_table$start_55min <- Fiftyfive_min_group
  
  #d) Grouped by 30 mins
  Thirty_min_group <- c()
  for(j in 1:length(start_1min_vector)){
    if(start_1min_vector[j]<30){
      Thirty_min_group[j] <- "0"
    } else{Thirty_min_group[j] <- "30"}
  }
  TM_table$start_30min <- Thirty_min_group

  return(TM_table)
}


#1st part
#Loading 2021 data
Input_1st <- list.files(pattern="*.csv")[1:32] %>% map_df(~fread(.))
dim(Input_1st) #6000799      25
#Making 25col table
Output_1st <- make_table_func(Input_1st)
#Saving table
write.csv(Output_1st, "../Audio_1st_part,_table_25cols_w3_v2.csv", row.names=FALSE)


#2nd part
#Loading 2021 data
Input_2nd <- list.files(pattern="*.csv")[33:64] %>% map_df(~fread(.))
dim(Input_2nd) #6511215      25
#Making 25col table
Output_2nd_v2 <- make_table_func(Input_2nd)
#Saving table
write.csv(Output_2nd, "../Audio_2nd_part,_table_25cols_w3_v2.csv", row.names=FALSE)

#3rd part
#Loading 2021 data
Input_3rd <- list.files(pattern="*.csv")[65:96] %>% map_df(~fread(.))
dim(Input_3rd) #6104916      25
#Making 25col table
Output_3rd <- make_table_func(Input_3rd)
#Saving table
write.csv(Output_3rd, "../Audio_3rd_part,_table_25cols_w3_v2.csv", row.names=FALSE)

#############################################################################

