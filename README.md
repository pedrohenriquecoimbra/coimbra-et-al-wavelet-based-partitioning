# Improvement of CO2 flux quality through wavelet-based Eddy Covariance: a new method for partitioning respiration and photosynthesis

Pedro Henrique H. Coimbra1*, Benjamin Loubet1, Olivier Laurent2, Matthias Mauder3,5, Bernard Heinesch4, Jonathan Bitton4 , Jérémie Depuydt1, Pauline Buysse1 

1 ECOSYS, INRAE, AgroParisTech, Université Paris-Saclay, Palaiseau, France<br>
2 Laboratoire des Sciences du Climat et de l'Environnement, CEA, CNRS, Université Paris-Saclay, Gif-sur-Yvette, France<br>
3 Institute of Meteorology and Climate Research - Atmospheric Environmental Research (IMK-IFU), Karlsruhe Institute of Technology, Garmisch-Partenkirchen, Germany<br>
4 Faculté des Sciences Agronomiques de Gembloux, Unité de Physique, Gembloux, Belgium<br>
5 Institute of Hydrology and Meteorology, Technische Universität Dresden, Dresden,
Germany<br>
\* corresponding author: pedro-henrique.herig-coimbra@inrae.fr

## Abstract. 

Eddy Covariance (EC) is praised for producing direct, continuous, and reliable flux monitoring for greenhouse gases. For CO2, the method has been commonly used to derive gross primary productivity (GPP) and ecosystem respiration (Reco) from net ecosystem exchange (NEE). However, standard EC is impacted by non-stationarity, reducing data quality and consequently impacting standard partitioning methods that are constructed on simplistic assumptions.
 
This work proposes a new wavelet-based processing framework for EC tested over two French ICOS ecosystem sites, a mixed forest (FR-Fon) and cropland (FR-Gri), over several years. A new direct partitioning method was also developed, applying conditional sampling in wavelet decomposed signals. This new empirical method splits positive and negative parts of the wavelet decomposed product of the wind vertical component and CO2 dry molar fraction, conditioned by water vapour flux, to compute GPP and Reco. 

Results show 17 to 29 % fewer gaps in wavelet-based than with standard EC, varying between sites, day and night. A good correlation between methods was observed during turbulent and stationary periods (R²=0.99). However, wavelet-based NEE was 9% lower than in standard EC, likely related to low-frequency attenuation led by the detrending nature of wavelet transform.

The new wavelet-based direct partitioning provided daily GPP and Reco very similar to night- and day-time model-based partitioning methods, with the difference between our method and these standard methods smaller than between them. Our method did not produce positive GPP, a common error in the night-time method. It also showed Reco seasonal patterns coherent with management practices at the crop site (growing season, harvest, manure application), which was not always the case for the standard methods. The Reco diel cycle was noticeably different, whereas the standard methods are temperature-driven; our method had a daily pattern correlated to solar radiation and a night-time pattern correlated to soil temperature. 

# Code

- coimbra2023_figures.ipynb<br>
_Function_: Loads data to produce figures used in the paper<br>
_Language_: Python (3.8)<br>
_Requires_: matplotlib, seaborn, numpy, pandas, sklearn, functools
- coimbra2023_fluxcalculation.ipynb<br>
_Function_: Use REddyProc to use model-based gapfilling and partitioning<br>
_Language_: Python (3.8)<br>
_Requires_: yaml, numpy, pandas, zipfile, io, rpy2, pywt, pycwt, fcwt
- coimbra2023_gapfilling.R<br>
_Function_: Use REddyProc to use model-based gapfilling and partitioning<br>
_Language_: R<br>
_Requires_: REddyProc
