library(REddyProc)
library(dplyr)
library(readr)

silenceR <- function(){
  sink(file("./.r2py_out_sink.log", "w"), type="message")
}

sink.reset <- function(){
    for(i in seq_len(sink.number())){
        sink(NULL)
    }
}

REddyProcsMDSGapFill = function(EProc, cols_to_apply=c('NEE'), verbosity=0){
  EProc$sEstimateUstarScenarios(
    nSample = 200L,
    probs = c(0.05, 0.5, 0.95))
  #add Ustar threshold to the dataset
  #(uStarTh <- EProc$sEstUstarThold()$uStarTh)
  EProc$sGetUstarScenarios()
  (uStarThAgg <- EProc$sGetEstimatedUstarThresholdDistribution())
  EProc$useSeaonsalUStarThresholds()

  # inspect the changed thresholds to be used
  #uStarSuffixes <- colnames(EProc$sGetUstarScenarios())[-1]
  #print(uStarSuffixes)

  for (c in cols_to_apply){
    #EProc$sMDSGapFill(c, FillAll=FALSE, isVerbose=F)
    #EProc$sMDSGapFillAfterUstar(c)
    EProc$sMDSGapFillUStarScens(c, isVerbose=F)#, FillAll=FALSE)
  }
  #EProc$sFillVPDFromDew()  # fill longer gaps still present in VPD_f (important for day-time partition)
  return(EProc)
}

library(assertthat)
do_gapfill = function(path_to_file, met_path="", NEE_name="co2_flux",
                      climcols=c('Tair', 'VPD', 'Rg'),
                      fluxcols=c('NEE'),
                      Lat=43, Lon=47.5,
                      partitioning="MRF",
                      save=F,
                      path_to_output_file="",
                      verbosity=1){
  all_cols <- c(climcols, fluxcols)

  # load csv from path
  if (is.character(path_to_file)){
    data1 <- read_csv(path_to_file, na = c("NAN", "NA", "NaN", -9999))
  } else {data1 <- path_to_file}

  if (met_path!=""){
    meteo <- read_csv(met_path, na = c("NAN", "NA", "NaN", -9999)) %>%
      mutate("TIMESTAMP" = as.POSIXct(as.character(TIMESTAMP_END), tryFormat=c("%Y%m%d%H%M"), tz="GMT")) %>%
      # rename(any_of(c("Rg"="SW_IN", "rH"="RH"))) %>%
      select(c("TIMESTAMP", "SW_IN", "RH"))
    data1 <- merge(data1, meteo, by="TIMESTAMP", all.x=T, suffixes=c("_", ""))
  }

  data1 <- data1 %>%
    # guarantee time is time
    mutate(TIMESTAMP = as.POSIXct(TIMESTAMP, tz='UTC')) %>%
    #filter(TIMESTAMP < as.POSIXct(as.character("201904010000"), tryFormat=c("%Y%m%d%H%M"), tz="UTC")) %>%
    # guarantee time continuity
    merge(tibble("TIMESTAMP"=seq(min(data1$TIMESTAMP, na.rm=T), max(data1$TIMESTAMP, na.rm=T), by="30 min")),
          by="TIMESTAMP", all=T) %>%
    # rename variable names
    rename(any_of(c("Tair"="air_temperature", "Rg"="SW_IN", "rH"="RH",
                    "Ustar"="ustar", "Ustar"="us", "Ustar"="u*", "VPD"="vpd"))) %>%
    # verify units
    mutate(Tair_Av = mean(.$Tair, na.rm=T), VPD_Av = mean(.$VPD, na.rm=T),
           Tair = ifelse(Tair_Av>100, Tair-273, Tair),
           VPD = ifelse(VPD_Av>100, VPD/1000, VPD),
           Rg = ifelse(Rg < 0, 0, Rg),
           Tsoil = Tair)

  if ("NEE" %in% colnames(data1)) {data2 <- data1 %>%
    rename(DateTime=TIMESTAMP)
  } else {
    data2 <- data1 %>%
      # rename variable for NEE
      rename("NEE":=!!NEE_name) %>%
      # apply flags (value -> nan)
      #mutate(NEE = ifelse(`fITC`>0, NaN, NEE),
      #       NEE = ifelse(`fSTA`>0, NaN, NEE)) %>%
      mutate(NEE = ifelse(`flag(w)`>1, NaN, NEE),
             NEE = ifelse(`flag(w/co2)`>1, NaN, NEE)) %>%
      # exclude duplicates (keep first)
      distinct(TIMESTAMP, .keep_all = T) %>%
      rename(DateTime=TIMESTAMP)
  }

  if (verbosity>1) print(data2 %>% dplyr::select(DateTime, NEE, Ustar, unlist(all_cols)) %>% summary())

  #+++ Initalize R5 reference class sEddyProc for processing of eddy data
  #+++ with all variables needed for processing later
  EProc <- sEddyProc$new(
    "FluxSite", data2, union(c('Rg', 'Ustar'), unlist(all_cols)))

  EProc$sSetLocationInfo(LatDeg = Lat, LongDeg = Lon, TimeZoneHour = 1)
  #Gap-filling of Tair, VPD, NEE and LE
  for (c in climcols){
    EProc$sMDSGapFill(c, FillAll=T, isVerbose=F)
  }

  # run MDS gap fill (REddyProc)
  EProc <- REddyProcsMDSGapFill(EProc, cols_to_apply = fluxcols)

  # run partitioning (REddyProc)
  if (partitioning == "MRF"){
    EProc$sMRFluxPartitionUStarScens()
  } else if (partitioning == "TKF"){
    EProc$sTKFluxPartitionUStarScens()
  } else if (partitioning == "GLF"){
    EProc$sGLFluxPartitionUStarScens()}

  #++ Export gap filled and partitioned data to standard data frame
  FilledEddyData <- EProc$sExportResults()
  FilledEddyData$TIMESTAMP <- data2$DateTime
  #FilledEddyData$uStarThAgg <- uStarThAgg
  #print(uStarThAgg)
  #print(uStarTh)
  #FilledEddyData <- cbind(EddyData, FilledEddyData)
  #FilledEddyData$TIMESTAMP <- lubridate::ymd_hms(FilledEddyData$TIMESTAMP)
  #FilledEddyData <- dplyr::select(FilledEddyData, -c(DateTime))

  #FilledEddyData <- FilledEddyData %>%
  #  select(starts_with('NEE'), starts_with('GPP'), starts_with('Reco'))
  #FilledEddyData <- rbind(colnames(FilledEddyData), FilledEddyData)
  print('FilledEddyData 14:39')
  print(colnames(FilledEddyData))
  if (save){
    write_csv(FilledEddyData,
              ifelse(path_to_output_file!='', path_to_output_file, path_to_file))
    return()
  } else {return(FilledEddyData)}
}

example <- function(SiteName){
  SiteName <- "FR-Gri"
  cwd <- "C:/Users/phherigcoimb/OneDrive/INRAe/thesis-project-1/data/flux/ICOS/"
  path <- paste0(cwd, SiteName,"/",SiteName,"_full_output_flagged.30mn.csv")
  root = paste0("C:/Users/phherigcoimb/OneDrive/INRAe/thesis-project-1/data/flux/ICOS/",
                SiteName,"/",SiteName)

  mpath <- ifelse(SiteName=="FR-Fon", paste0(cwd,SiteName,"/ICOSETC_",SiteName,"_METEO_L2.csv"), "")

  # meteo <- read_csv("C:/Users/phherigcoimb/OneDrive/INRAe/thesis-project-1/data/flux/ICOS/FR-Fon/ICOSETC_FR-Fon_METEO_L2.csv",
  #                   na = c("NAN", "NA", "NaN", -9999)) %>%
  #   mutate("TIMESTAMP" = as.POSIXct(as.character(TIMESTAMP_END), tryFormat=c("%Y%m%d%H%M"), tz="GMT")) %>%
  #   select(c("TIMESTAMP", "SW_IN", "RH"))
  # rg <- read_csv("C:/Users/phherigcoimb/OneDrive/INRAe/thesis-project-1/data/flux/ICOS/FR-Fon/FR-Fon_Rg.csv",
  #                                     na = c("NA", -9999))
  # plot(as.Date(as.character(ICOSETC_FR_Fon_METEO_L2$TIMESTAMP_START),
  #              tryFormat=c("%Y%m%d%H%M")), ICOSETC_FR_Fon_METEO_L2$SW_IN)

  do_gapfill(path, mpath, NEE_name = "dwt_wco2_x", partitioning = "MRF",
             verbosity = 2) %>%
    write_csv(paste0(root, "_full_gapfill_MRF_DWgapswithSTAandITC.30mn.csv"))
  break
  dat_mrf_ec <- do_gapfill(path, mpath, NEE_name = "co2_flux", partitioning = "MRF",
                           verbosity = 2)
  print("ECS MRF done")
  dat_tkf_ec <- do_gapfill(path, mpath, NEE_name = "co2_flux", partitioning = "TKF",
                           verbosity = 2)
  print("ECS TKF done")
  dat_glf_ec <- do_gapfill(path, mpath, NEE_name = "co2_flux", partitioning = "GLF",
                           verbosity = 2)
  print("ECS GLF done")
  dat_mrf_dw <- do_gapfill(path, mpath, NEE_name = "dwt_wco2_x", partitioning = "MRF",
                           verbosity = 2)
  print("DW MRF done")
  dat_tkf_dw <- do_gapfill(path, mpath, NEE_name = "dwt_wco2_x", partitioning = "TKF",
                           verbosity = 2)
  print("DW TKF done")
  dat_glf_dw <- do_gapfill(path, mpath, NEE_name = "dwt_wco2_x", partitioning = "GLF",
                           verbosity = 2)
  print("DW GLF done")

  plot(dat_tkf_ec$TIMESTAMP, dat_tkf_ec$Reco_DT_uStar,
       xlim=c(as.POSIXct("2022/07/11 0000", tz='GMT'), as.POSIXct("2022/07/26 0000", tz='GMT')),
       ylim=c(0,12))
  lines(dat_mrf_ec$TIMESTAMP, dat_mrf_ec$Reco_uStar)

  library(ggplot2)
  library(lubridate)
  dat_mrf_ec %>%
    mutate(TIMESTAMP=round_date(TIMESTAMP, unit="1 days")) %>%
    group_by(TIMESTAMP) %>%
    summarise_all(mean) %>%
    ggplot(aes(x=TIMESTAMP)) +
    geom_line(aes(y=R_ref_uStar), color="red") +
    geom_line(aes(y=Reco_uStar)) +
    geom_line(aes(y=GPP_uStar_f)) +
    geom_line(aes(y=Rg_f/50+15), color='blue') +
    geom_line(aes(y=Tair_f), color='orange') +
    geom_line(aes(y=ifelse(Tair_f>28, Tair_f, 28)), color='red') +
    xlim(as.POSIXct("2021/07/01 0000", tz='GMT'), as.POSIXct("2021/12/01 0000", tz='GMT')) +
    ylim(-5, 40)

  write_csv(dat_mrf_ec, paste0(root, "_full_gapfill_MRF_EP.30mn.csv"))
  write_csv(dat_tkf_ec, paste0(root, "_full_gapfill_TKF_EP.30mn.csv"))
  write_csv(dat_glf_ec, paste0(root, "_full_gapfill_GLF_EP.30mn.csv"))

  write_csv(dat_mrf_dw, paste0(root, "_full_gapfill_MRF_DW.30mn.csv"))
  write_csv(dat_tkf_dw, paste0(root, "_full_gapfill_TKF_DW.30mn.csv"))
  write_csv(dat_glf_dw, paste0(root, "_full_gapfill_GLF_DW.30mn.csv"))
}

