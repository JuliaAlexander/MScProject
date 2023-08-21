#Daily calling activity
  #by half hour

#============================================================
#Section 1 
##Loading 2021 predictions dataset

#1. Libraries
library(data.table) #eg fread, fwrite
library(tidyverse) #eg %>%, ggplot

#2. Loading 25cols data
setwd
#w3_v2
raw_data  = list.files(pattern="*table_24cols_w3_v2.csv") %>% map_df(~fread(.))
dim(raw_data) ##118614775       25
colnames(raw_data)
# [1] "file"             "start_time"       "end_time"         "background"       "gc"              
# [6] "mm"               "sc"               "predict_group"    "MT_background"    "MT_gc"           
# [11] "MT_mm"            "MT_sc"            "Folder"           "AudioMoth"        "WAV_name"        
# [16] "WAV_date"         "WAV_time"         "start_gap"        "clip_start_clock" "clip_end_clock"  
# [21] "clip_start_hour"  "start_gap_1min"   "start_5min"       "start_55min"      "start_30min"  


#3. Tidying dataset
#a. Removing 1970 row
crop_data <- raw_data[-(which(raw_data$WAV_date == "1970-01-01")),]
dim(crop_data) #18614774       25
#b. Removing 22:00 rows
Audio_table <- crop_data[-which(crop_data$clip_start_hour == "22"),]
dim(Audio_table) #18614742       25
#c. Tidying R memory
rm(crop_data)
#rm(raw_data)
gc()

#===================================================================

#Section 2
#Examining gibbon calls per half hour

#1. half hours
half_hours <- as_hms(c("4:30:00", "5:00:00", "5:30:00", "6:00:00", "6:30:00",
                       "7:00:00", "7:30:00", "8:00:00", "8:30:00", "9:00:00", "9:30:00",
                       "10:00:00","10:30:00","11:00:00","11:30:00","12:00:00"))

#2. gc
#a) counting by file
#subset of each half hour grouping
gc_30min_table <- data.frame()
for(i in 1:(length(half_hours)-1)){ #1 to 14
  #half hour subset
  time_i <- Audio_table[as_hms(half_hours[i]) <= clip_start_clock_hms
                        & as_hms(half_hours[i+1]) > clip_start_clock_hms]
  
  #Total number of 5s clips by file name
  Total_30m_i <- time_i[,.(Total_5s_clips=.N), by=.(file)]
  #gc counts
  setkey(time_i, file)
  gc_count_i <- time_i[MT_gc=="gc",.N, by=.(file)][CJ(unique(time_i$file))][is.na(N), N:=0L]
  #Combining gc count and total detection
  gc_merged_i = merge(gc_count_i, Total_30m_i)
  
  #Selecting files with 30mins of audio
  gc_30mins_i <- gc_merged_i[Total_5s_clips=="360"]
  #adding half hour category column
  gc_30mins_i$Half_hour <- half_hours[i]
  
  #Storing counts
  gc_30min_table <- rbind(gc_30min_table, gc_30mins_i)
}
nrow(gc_30min_table) #27666 rows ##24224 rows
#write.csv(gc_30min_table, "gc_half_hour_by_file_calls,_27666_rows,_added_9am_part.csv")

#b) gc quantiles
Q_gc_time2 <- gc_30min_table[, .(Mean = mean(N), Min=min(N),
                                Q1= quantile(N, 0.25),Q2=quantile(N, 0.5),
                                Q3=quantile(N, 0.75), Max=max(N)),
                            by=.(Half_hour)]

Q_gc_time2
    # Half_hour      Mean Min Q1 Q2 Q3 Max
    # 1:  04:30:00 0.4103746   0  0  0  0 172
    # 2:  05:00:00 0.5656598   0  0  0  0 135
    # 3:  05:30:00 0.8958991   0  0  0  0  72
    # 4:  06:00:00 0.4274760   0  0  0  0 144
    # 5:  06:30:00 0.8170347   0  0  0  0  67
    # 6:  07:00:00 0.6032577   0  0  0  0 131
    # 7:  07:30:00 0.0000000   0  0  0  0   0
    # 8:  08:00:00 1.4778481   0  0  0  0 170
    # 9:  08:30:00 1.0163671   0  0  0  0 223
    # 10:  09:00:00 0.9937695   0  0  0  0  91
    # 11:  09:30:00 0.5107337   0  0  0  0 117
    # 12:  10:00:00 0.6400000   0  0  0  0  55
    # 13:  10:30:00 0.9636480   0  0  0  0 227
    # 14:  11:00:00 1.0623053   0  0  0  0  97
    # 15:  11:30:00 0.5591329   0  0  0  0 203



#3. mm
#a) counting by file
#subset of each half hour grouping
mm_30min_table <- data.frame()
for(i in 1:(length(half_hours)-1)){ #1 to 14
  #half hour subset
  time_i <- Audio_table[as_hms(half_hours[i]) <= clip_start_clock_hms
                        & as_hms(half_hours[i+1]) > clip_start_clock_hms]
  
  #Total number of 5s clips by file name
  Total_30m_i <- time_i[,.(Total_5s_clips=.N), by=.(file)]
  #mm counts
  setkey(time_i, file)
  mm_count_i <- time_i[MT_mm=="mm",.N, by=.(file)][CJ(unique(time_i$file))][is.na(N), N:=0L]
  #Combining mm count and total detection
  mm_merged_i = merge(mm_count_i, Total_30m_i)
  
  #Selecting files with 30mins of audio
  mm_30mins_i <- mm_merged_i[Total_5s_clips=="360"]
  #adding half hour category column
  mm_30mins_i$Half_hour <- half_hours[i]
  
  #Storing counts
  mm_30min_table <- rbind(mm_30min_table, mm_30mins_i)
}
nrow(mm_30min_table) #27666 rows ##24224 rows
write.csv(mm_30min_table, "mm_half_hour_by_file_calls,_27666_rows,_added_9am_part.csv")

#b) mm quantiles
Q_mm_time2 <- mm_30min_table[, .(Mean = mean(N), Min=min(N),
                                 Q1= quantile(N, 0.25),Q2=quantile(N, 0.5),
                                 Q3=quantile(N, 0.75), Max=max(N)),
                             by=.(Half_hour)]

Q_mm_time2
# Half_hour      Mean Min Q1 Q2 Q3 Max
# 1:  04:30:00 0.1224784   0  0  0  0  20
# 2:  05:00:00 0.2302067   0  0  0  0  23
# 3:  05:30:00 0.4321767   0  0  0  0  46
# 4:  06:00:00 5.0249201   0  0  0  4  70
# 5:  06:30:00 8.0189274   0  0  0  7 273
# 6:  07:00:00 3.9319372   0  0  0  3 304
# 7:  07:30:00 1.1071429   0  0  0  0  21
# 8:  08:00:00 2.6329114   0  0  0  0  55
# 9:  08:30:00 1.2057125   0  0  0  0 188
# 10:  09:00:00 1.6510903   0  0  0  0 200
# 11:  09:30:00 1.2816405   0  0  0  0 198
# 12:  10:00:00 1.3169231   0  0  0  0 349
# 13:  10:30:00 0.4764031   0  0  0  0 185
# 14:  11:00:00 1.8193146   0  0  0  0 326
# 15:  11:30:00 0.4179152   0  0  0  0 191



#4. sc
#a) counting by file
#subset of each half hour grouping
sc_30min_table <- data.frame()
for(i in 1:(length(half_hours)-1)){ #1 to 14
  #half hour subset
  time_i <- Audio_table[as_hms(half_hours[i]) <= clip_start_clock_hms
                        & as_hms(half_hours[i+1]) > clip_start_clock_hms]
  
  #Total number of 5s clips by file name
  Total_30m_i <- time_i[,.(Total_5s_clips=.N), by=.(file)]
  #sc counts
  setkey(time_i, file)
  sc_count_i <- time_i[MT_sc=="sc",.N, by=.(file)][CJ(unique(time_i$file))][is.na(N), N:=0L]
  #Combining sc count and total detection
  sc_merged_i = merge(sc_count_i, Total_30m_i)
  
  #Selecting files with 30mins of audio
  sc_30mins_i <- sc_merged_i[Total_5s_clips=="360"]
  #adding half hour category column
  sc_30mins_i$Half_hour <- half_hours[i]
  
  #Storing counts
  sc_30min_table <- rbind(sc_30min_table, sc_30mins_i)
}
nrow(sc_30min_table) #27666 rows
#write.csv(sc_30min_table, "sc_half_hour_by_file_calls,_27666_rows,_added_9am_part.csv")

#b) sc quantiles
Q_sc_time2 <- sc_30min_table[, .(Mean = mean(N), Min=min(N),
                                Q1= quantile(N, 0.25),Q2=quantile(N, 0.5),
                                Q3=quantile(N, 0.75), Max=max(N)),
                            by=.(Half_hour)]
Q_sc_time2
    # Half_hour       Mean Min Q1 Q2    Q3 Max
    # 1:  04:30:00  2.5806916   0  0  0  2.00 256
    # 2:  05:00:00  4.8925278   0  0  0  3.00 283
    # 3:  05:30:00  0.9842271   0  0  0  1.00  19
    # 4:  06:00:00 14.1543131   0  0  2 14.00 224
    # 5:  06:30:00 13.7981073   0  0  2 17.00 121
    # 6:  07:00:00 12.7286213   0  0  2 12.00 292
    # 7:  07:30:00  3.1428571   0  0  1  3.25  31
    # 8:  08:00:00  5.6613924   0  0  0  4.00 108
    # 9:  08:30:00  5.6039795   0  0  1  4.00 229
    # 10:  09:00:00  3.4953271   0  0  0  2.00  76
    # 11:  09:30:00  5.8917014   0  0  1  4.00 211
    # 12:  10:00:00  1.8000000   0  0  0  2.00  55
    # 13:  10:30:00  3.9247449   0  0  1  3.00 237
    # 14:  11:00:00  2.6479751   0  0  0  1.00 119
    # 15:  11:30:00  4.0717246   0  0  1  3.00 266


#===================================

#5. Comparing mm and sc barplot
#a) adding category column
#Q_gc_time2$Category <- "gc"
Q_mm_time2$Category <- "mm"
Q_sc_time2$Category <- "sc"

#b) finding half hour range
hm_half_hours <- format(strptime(half_hours, "%H:%M:%S"), "%H:%M")
# [1] "04:30" "05:00" "05:30" "06:00" "06:30" "07:00" "07:30" "08:00" "08:30" "09:00" "09:30"
# [12] "10:00" "10:30" "11:00" "11:30" "12:00"
Time_range2 <- c()
for(i in 1:(length(hm_half_hours)-1)){
  Time_range2[i] <- paste0(hm_half_hours[i],"-", hm_half_hours[i+1])
}
Time_range2
# [1] "04:30-05:00" "05:00-05:30" "05:30-06:00" "06:00-06:30" "06:30-07:00" "07:00-07:30"
# [7] "07:30-08:00" "08:00-08:30" "08:30-09:00" "09:00-09:30" "09:30-10:00" "10:00-10:30"
# [13] "10:30-11:00" "11:00-11:30" "11:30-12:00"

#c) adding half hour range column
#Q_gc_time2$Category <- Time_range2
Q_mm_time2$Time <- Time_range2
Q_sc_time2$Time <- Time_range2

#d) Combining mm and sc mean counts
Q_time_mm_sc2 <- rbind(Q_mm_time2,Q_sc_time2)

#e) Mean to 1dp
Q_time_mm_sc2$Mean_1dp <- formatC(round(Q_time_mm_sc2$Mean,1), digits=1, format="f")

#f) saving mm sc combined quantiles
#write.csv(Q_time_mm_sc2, "Quatile_mm_sc_combined,_with_9am_part.csv", row.names=F)

#g) graph
ggplot(Q_time_mm_sc2, aes(as.factor(Time), Mean, fill=Category))+
  geom_bar(stat="identity", position="dodge")+
  theme_bw()+
  scale_x_discrete(guide=guide_axis(angle=45))+
  #labs(y="Mean call rate (number of detections per 30 minute period)", fill="Call type")+
  labs(x="Time", y="Mean call rate (number of detections per 30 minute period)",
       fill="Call type")+
  geom_text(aes(label=Mean_1dp),
            position=position_dodge(.9), size=10/.pt)
ggsave("L1_13d.Ncall_vs_half_hour,_mm_sc_dodge_Barplot,_mean_call_rate_per_30mins,_with_9am_part.png", width=9, height=6)




