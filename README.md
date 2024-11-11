### A Machine-Learning Framework to Enhance, Adapt, and Extend the GPM Combined Radar-Radiometer Algorithm (CORRA)

Principal Investigator (PI) and Co-Investigator (Co-I)

#### 1 Scientific/Technical Management
#### 1.1 Executive Summary
Accurately quantifying uncertainties is critical to deriving optimal precipitation estimates from multiple information sources. This need has always existed but is now even more urgent as the number and diversity of data sources continue to expand rapidly. Recent developments, such as the deployment of miniaturized, cost-effective sensors (e.g., NASA's TROPICS constellation), alongside operational use of deep-learning-based numerical weather prediction models (e.g., FourCastNet, GraphCast), have underscored the need for robust methods to combine diverse information sources. Traditional satellite-based algorithms (from agencies such as NASA and NOAA) for precipitation estimation also contribute to the pressing need for enhanced data integration methods.

The NASA GPM Combined Radar-Radiometer Algorithm (CORRA) is a sophisticated algorithm that combines data from the GPM Dual-frequency Precipitation Radar (DPR) and the GPM Microwave Imager (GMI) to estimate precipitation. CORRA’s modular structure allows rigorous procedures to quantify uncertainties from each source of information [1]. Although CORRA has been refined over time, several limitations remain [2]. For instance, snowfall estimates are biased low relative to ground-based radar (e.g., MRMS) and the algorithm tends to miss light precipitation, particularly at high latitudes, due to limited radar sensitivity.

We propose developing two machine-learning (ML) modules to address CORRA's existing limitations and extend its capabilities for tasks like estimating precipitation from other instrument combinations and supporting data assimilation. The first module will employ Cluster-wise Ridge Regression (CRR) [4] to estimate precipitation rates, particle size distribution (PSD) parameters, and associated uncertainties from DPR-only observations. The radar CRR model will use a customized version of CORRA with an enhanced snowfall retrieval derived from field campaign microphysical observations. A key advantage of the radar CRR model is its computational efficiency relative to the current CORRA ensemble radar-profiling algorithm, with differentiable functions that allow precise uncertainty estimation.

The second module, also using CRR, will refine precipitation estimates and uncertainties from radiometer data. This model will be trained using synthetic brightness temperatures derived from the radar CRR output and CORRA’s surface emissivity estimates, via radiative transfer simulation. A biased sampling strategy, along with the CloudSat-GPM coincidence dataset [3], will support better representation of light precipitation profiles in training data, enabling improved estimates in both precipitation rates and uncertainties over current CORRA capabilities.

A schematic overview of the proposed research is provided in Figure 1.

![Corra Flowchart](CORRA_FlowChart.png)*Figure 1. Schematic illustration of the proposed research. A radar Cluster-wise Ridge Regression (CRR) model will be trained using a customized CORRA with enhanced snowfall retrieval and incorporated to improve CORRA outputs. Additionally, a radiometer CRR model will be trained using synthetic data, enabling refined products and potential applications in deep-learning-based NWP data assimilation.*

#### 1.2 Technical Approach
#### 1.2.1 Radar Cluster-wise Ridge Regression
The GPM CORRA algorithm uses an ensemble filter for precipitation profile estimation [1]. A radar profiling algorithm [5] generates an ensemble of Ku-band radar-consistent precipitation profiles. This ensemble is then used to simulate Ka-band radar and GMI observations, which are iteratively updated, incorporating independent PIA estimates from the surface reference technique (SRT). The radar algorithm is run twice in the current CORRA version, using a PSD relationship between the intercept ($N_w$) and the mean diameter ($D_m$) [6]. Despite efforts to streamline the algorithm, it remains computationally intensive, making further refinement, alternative testing, or adaptation for missions like NASA AOS challenging.

To mitigate this, we propose implementing a more efficient, cluster-wise linear regression approach [4]. In this technique, predictors are grouped into clusters, and a linear regression model is estimated for each cluster. This approach, particularly effective for diverse applications like satellite-based cloud ice estimation [7], is stabilized by adding a ridge regression penalty to reduce overfitting. We refer to this method as Cluster-wise Ridge Regression (CRR). Due to its simplicity, CRR has been independently developed across research areas under various names, including conditional regression, cluster regression, and piece-wise linear regression. Conceptually, CRR is related to Gaussian Mixture Models [8], with crisp cluster boundaries, and may also be seen as a simplified neural network with activation paths corresponding to a unique cluster. CRRs can provide uncertainty estimates, unlike standard neural networks, making them suitable for our precipitation estimation task.

An anonymized description of the radar CRR model implementation is available in a GitHub repository: https://anonymous.4open.science/r/PMMCCST-397B/README.md. Our CRR approach uses the K-Means algorithm for clustering, with stratification by precipitation type (convective or stratiform) and surface type (land or ocean). Predictors in the radar CRR model include Ku-band radar reflectivity profiles and freezing level position, while dependent variables include PSD parameters ($N_w$, $D_m$), precipitation rates, and water content. Each cluster has a unique ridge regression of the form $Y = X\beta $ is derived, where $Y$ is the vector of dependent variables, $X$ is the matrix of predictors, and $\beta$ is the vector of coefficients. The coefficients are estimated using:

$$\hat{\beta} = (X^TX + \lambda I)^{-1}X^TY$$

where $\lambda$ is the regularization parameter chosen by cross-validation. For uncertainty quantification, we calculate $\frac{1}{N-p} (Y-X\hat{\beta})^T(Y-X\hat{\beta})$, with $N$ as data points per cluster and $p$ as the size of $\hat{\beta}$.

Using this model, we derive $N_w$ ensembles by identifying the appropriate cluster, estimating conditional averages with CRR, and sampling from the uncertainty distribution within that cluster. An analysis of CRR-derived ensembles (not shown) indicates strong consistency with CORRA’s dual-frequency retrievals and physical consistency across parameters. For example, higher-than-average $N_w$ estimates correspond to lower $D_m$ values and reduced precipitation rates. Additionally, the joint distribution of $N_w$ and $D_m$ aligns with the structurally imposed $N_w$-$D_m$ relationship [6].

An illustration comparing $N_w$ and $D_m$ estimates derived from the radar CRR model against CORRA V08 reference values is shown in Figure 2. The radar CRR model performs particularly well for $D_m$, despite using only Ku-band data, while the $N_w$ estimates are less strongly correlated to the CORRA V08 reference values given the lack of Ka-band observations.

![](logdNw_and_Dm.png)*Figure 2: Comparison of $N_w$ and $D_m$ estimates derived from the radar CRR model and CORRA V08 references, showing strong consistency, especially for $D_m$.*

To address biases in ice-phase precipitation estimates, we will use a customized CORRA version where “a priori” $N_w$ values for ice-phase PSDs are derived from dual-frequency reflectivity observations based on NASA IMPACTS field campaign microphysical data [11]. Specifically, we will leverage a Ku- and Ka-band reflectivity database, constructed using electromagnetic scattering calculations from [12] and IMPACTS PSDs, to refine relationships between radar observations and PSDs. A point-based CRR, which provides uncertainty estimates, will be trained on this database to supply "a priori" $N_w$ values for the customized CORRA. This version of CORRA will then generate enhanced products for training the radar CRR module across all precipitation and surface type combinations. Data collected from 21 May 2018 onward will be used to ensure dual-frequency coverage across the entire radar scan.

We propose that, in CORRA V9, the radar CRR model replace the computationally intensive first iteration of the radar-profiling algorithm, to enable the generation of rapid, bias-corrected $N_w$ ensembles. Full replacement of CORRA’s radar-profiling with a CRR-based approach will be explored post-V09, potentially benefiting NASA’s AOS and INCUS missions.  

The current CORRA ensemble-based approach might be improved by incorporating a gradient-based optimization algorithm. While the ensemble-smoother avoids explicit gradient calculations, it can be suboptimal for non-linear problems when the initial ensemble is not close to the optimal solution. Developing an efficient method to estimate the Jacobians of radar and radiometer forward models would enable the use of gradient-based algorithms, such as the Gauss-Newton algorithm, to refine the CORRA solution, while reducing the computational effort. We therefore propose an ML-based approach for estimating the Jacobians in these forward models, aimed at enhancing CORRA’s accuracy.

Specifically, we propose the use of a more general formulation of cluster-based regressions called cluster-weighted modeling (CWM) [] to estimate the gradient of the radar and radiometer forward models. That is, instead of hard assignments to clusters, we will use soft assignments, where each data point is associated with all clusters with varying weights [12]. The weights are modeled as probability densities, and learned through an Expectation-Maximization (EM) algorithm. The gradient of the forward model is then estimated as a weighted average of the gradients of the cluster-specific models. Unlike the CRR approach, where the Jacobians are constant within each cluster, the cluster-weighted modeling approach allows for continuous changes in the Jacobians across clusters. Specifically, as dependent variables Y are determined from predictors X using

$$Y = \frac {\sum_{k=1}^{K} f(X \beta_k) p(X|c_m)p(c_m)} {{\sum_{k=1}^{K} p(X|c_m)p(c_m)}}$$

where $f(X\beta_k)$ is the ridge regression for cluster $k$, $p(X|c_m)$, $p(X|c_m)$ is the conditional probability density of $X$ given cluster $c_m$, $p(c_m)$ is the probability of cluster $c_m$ and $K$ is the number of clusters, the Jacobian of Y with respect to X is can be calculated analytically by applying the quotient rule and using the derivatives of $f(X\beta_k)$ and $p(X|c_m)$ which are straightforward to derive. The actual formulas are omitted here for brevity.  Given the Jacobians $\frac {\partial Y}{\partial X}$, the state variables will be updated using the Gauss-Netwon algorithm for solving non-linear least squared problems [DF_Robust].

To fit the CWM model, we will use the CORRA V09 product. However, as CORRA does not save the intermediate radar and radiometer forward model outputs, we will need to reconstruct them through the algorithm's physical model to generate the training data. Given that Jacobian calculations preclude the need for a full ensemble, which makes the approach very efficient, we propose that the CWM model be run in parallel with the equivalent physical model to ensure consistency between the two. Large discrepancies would be an indication of application to points far from any of the existing clusters and would enable adaptive training by addition of aditional clusters. 

A detailed task list for ML radar related activities is presented in Table 1.

| Proposed Activity| Benefits  | Timeline  |
|------------------|-----------|-----------|
|Customize CORRA to derive improved ice-phase PSD estimates from dual-frequency radar observations  | Ubiased precipitation estimates appropriate for the training of a radar CRR model| Feb 2025 - March 2025 |
| Develop Ku-band CRR radar model with uncertainty estimates | Fast generation of unbiased $N_w$ ensemble for CORRA V09 | March 2025 - May 2025 |
| Integration and testing of the Ku-band radar CRR model| Improved CORRA V09 products| May 2025 - December 2025 |
| Develop CWM radar model and Jacobian calculation methdology | Faster and better exploration of the potential solution space  | January 2026 - June 2026 |
| Extend the CWM approach to support DL NWP data assimilation| Enhanced capabilities for future  applications| January 2027 - June 2027  |

#### 1.2.2 Radiometer Cluster-wise Ridge Regression
As previously mentioned, CORRA tends to miss light precipitation (especially at high-latitudes) due to the DPR's limited sensitivity. This limitation can be addressed by developing a radiometer-only module that can estimate light precipitation over oceans. Light precipitation retrievals from radiometer-only observations over land are considerably more challenging due to surface emissivity variations and are not considered for inclusion in V09.

A prototype light precipitation retrieval algorithm has been already developed [2] and is under investigation. The algorithm uses a neural network (NN) trained on the existing CORRA dataset using a sampling strategy optimized to maximize the agreement with the CloudSat-GPM coincident dataset. Specifically, if the exact fraction CORRA precipitation points is used in training, the NN will underestimate light precipitation. By selecting relatively fewer clear-sky points in the training dataset, the NN can be induced to produce more frequent non-zero precipitation estimates for potentially precipitating pixels but with DPR signal below the noise level. Technically, this is an estimation problem involving left-censored data, and the adaptive sampling strategy is a form of multiple imputation [13].

Nevertheless, while the NN approach is promising, it has limitations, as it does not guarantee consistency between the precipitation estimates and the associated simulated brightness temperatures. To address this, we propose developing a radiometer CRR model to estimate the cloud water path (CLWP) and total precipitable water (TWP) consistent the NN light precipitation estimates and observed GMI observations. The radiometer CRR model will be trained using synthetic brightness temperatures derived from WRF simulations. An example of synthetic brightness temperatures at 37 GHz derived from WRF simulations is shown in Figure 3. As apparent in the figure, the difference between the simulated brightness temperatures accounting for cloud water emission and those ignoring it is significant. Therefore, just the inclusion of the NN light precipiation estimates does not guarantee consistency between observed and simulated GMI brightness temperatures. This consistency is required in the development of radiometer-only precipitation retrieval algorithms for GMI as well as other sensors.

![](wrf_tb_37GHz_20240901_08\:00.png)
*Figure 3. Example of synthetic brightness temperatures at 37 GHz derived from WRF simulations. Simulated brightness temperatures in the left panel do not account for emission by cloud water. We propose augmenting the NN light precipitation model with a CRR CLWP and TPW model, ensuring thus the consistency between observed and observed GMI brightness temperatures.*


As previously described, CRR models provide uncertainty estimates, making them suitable for our conditional estimation task. Specifically, the radiometer CRR model provides unconditional estimates of $X_{NP}$={CLWP,TPW} and precipitation ($X_P$) and their uncertainties as a function of observed brightness temperature.  At the same time the NN model provide precipitation estimates $X_{P,NN}$.  We will use the Gaussian conditional estimation theory [MVar] to derive the conditional estimates of $X_{NP}$ given their unconditional estimates and the NN precipitation estimates. The final CLWP and TWP estimates will be derived as:

$$X_{NP}=X_{NP,a}+\Sigma_{NP,P}(Σ_{P,P}+Σ_{P,NN})^{-1}(X_{P,NN}−X_{P,a})$$

where $\Sigma_{NP,P}$ and $\Sigma_{P,P}$ are covariance matrices derived from the CRR model, while $\Sigma_{P,NN}$ is determined during the NN training.

​Shown in Figure 4 is an example of unconditional estimates of CLWP and TWP derived from the application of radiometer CRR model to GMI observations. The CORRA V08 reference precipitation areas are also outlined in black thin lines. As apparent in the figure, there are large areas with significant amounts of cloud not associated with any CORRA precipitation. These areas are likely to be associated with light precipitation that is not detected by the DPR. Moreover, there are precipitation areas detected by CORRA with very low CLWP. While precipitation can occur without significant cloud water, these may be also an indication of scattering in the GMI observations, which illustrates the need for the inclusion of a conditional estimation procedure such as the one described above.

![](retrieved_cldwp_tpw_59684.png)*Figure 4. Example of unconditional estimates of CLWP and TWP derived from the application of radiometer CRR model to GMI observations from orbit 59684 on 1 September 2024. The CORRA V08 reference precipitation areas are outlined in black thin lines.*

Results in Figure 4, are derived using a pre-trained CRR model using two WRF simulation roughly from the region and the period of the GMI observations used in the figure. To ensure other regimes are also well represented in the training data, we will carry out additional WRF simulations (or use existing ones) covering a wide range of conditions. Specifically, we intend to carry out WRF simulations over North Atlantic, North Pacific and some in the regions characterized by a large percentage of storms associated with the lowest height mode according to [Warm]. While WRF simulations are sensitive to multiple factor including microphysical scheme, resolution, etc., we expect that the resulting simulated brightness temperature cover well the real space of brightness temperatures.  Moreover, the CRR model and the subsequent conditional estimation procedure provide uncertainty estimates that are further refined by the current CORRA ensemble smoother.

Similarly to the radar CRR model, the radiometer CRR model will be integrated into CORRA V09. After the release of V09, we propose the development of a CWM model for the radiometer forward model. The CWM model will be trained using the CORRA V09 product, which will incorporate the light precipitation,  CLWP and TPW estimates. This model will provide the Jacobians of the radiometer forward model, enabling the use of gradient-based optimization algorithms for the refinement of the CORRA solution. Similarly to the radar CWM model, the radiometer CWM model will be run in parallel with the equivalent physical model to ensure consistency between the two. The same adaptive training strategy will be used to ensure that the representativeness of the training data.



### 1.3 Impact

### 1.4 Relevance

### 1.5 Work Plan

### 1.6 Schedule

### 2. References

[1] Grecu, M., Olson, W.S., Munchak, S.J., Ringerud, S., Liao, L., Haddad, Z., Kelley, B.L. and McLaughlin, S.F., 2016. The GPM combined algorithm. Journal of Atmospheric and Oceanic Technology, 33(10), pp.2225-2245.

[2] Olson, W. S. and coauthors, 2024, PMM Combined Radar-Radiometer Algorithm, PMM Science Team Meeting, San-Diego, 9-13 September 2024, available https://docs.google.com/presentation/d/e/2PACX-1vSYTmdkLlcvAKa2lVka050RDLewQ-IQHa3mRU4-6Yqu3tM3TLAF_8IqtvnJCl6-JDs8XlevwtUaYsIo/pub?start=false&loop=false&delayms=3000

[3] Turk, F. Joseph, Sarah E. Ringerud, Andrea Camplani, Daniele Casella, Randy J. Chase, Ardeshir Ebtehaj, Jie Gong, Mark Kulie, Guosheng Liu, Lisa Milani, and et al. 2021. "Applications of a CloudSat-TRMM and CloudSat-GPM Satellite Coincidence Dataset" Remote Sensing 13, no. 12: 2264. https://doi.org/10.3390/rs13122264

[4] DeSarbo, W.S. and Cron, W.L., 1988. A maximum likelihood methodology for clusterwise linear regression. Journal of classification, 5, pp.249-282.


[5] Grecu, M., Tian, L., Olson, W.S. and Tanelli, S., 2011. A robust dual-frequency radar profiling algorithm. Journal of applied meteorology and climatology, 50(7), pp.1543-1557.


[6] Olson, W.S., Masunaga, H. and GPM CORRA Team., 2022. GPM combined radar-radiometer precipitation algorithm theoretical basis document (version 7). NASA: Washington, DC, USA, available at https://gpm.nasa.gov/sites/default/files/2023-01/Combined_algorithm_ATBD.V07_0.pdf.


[7] Grecu, M., and J. E. Yorks, 2024: Synergistic Retrievals of Ice in High Clouds from Elastic Backscatter Lidar, Ku-Band Radar, and Submillimeter Wave Radiometer Observations. J. Atmos. Oceanic Technol., 41, 79–93, https://doi.org/10.1175/JTECH-D-23-0028.1.

[8] Reynolds, D.A., 2009. Gaussian mixture models. Encyclopedia of biometrics, 741(659-663).

[9] Bishop, C.M. and Bishop, H., 2023. Deep learning: Foundations and concepts. Springer Nature.

[10] Ahmed, M., Seraj, R. and Islam, S.M.S., 2020. The k-means algorithm: A comprehensive survey and performance evaluation. Electronics, 9(8), p.1295.

[11] McMurdie, L.A., Heymsfield, G.M., Yorks, J.E., Braun, S.A., Skofronick-Jackson, G., Rauber, R.M., Yuter, S., Colle, B., McFarquhar, G.M., Poellot, M. and Novak, D.R., 2022. Chasing snowstorms: The investigation of microphysics and precipitation for Atlantic coast-threatening snowstorms (IMPACTS) campaign. Bulletin of the American Meteorological Society, 103(5), pp.E1243-E1269.

[12] Kuo, K.S., Olson, W.S., Johnson, B.T., Grecu, M., Tian, L., Clune, T.L., van Aartsen, B.H., Heymsfield, A.J., Liao, L. and Meneghini, R., 2016. The microwave radiative properties of falling snow derived from nonspherical ice particle models. Part I: An extensive database of simulated pristine crystals and aggregate particles, and their scattering properties. Journal of Applied Meteorology and Climatology, 55(3), pp.691-708.

Gershenfeld, N.A., 1999. The nature of mathematical modeling. Cambridge university press.

Grecu, M., Tian, L., Olson, W.S. and Tanelli, S., 2011. A robust dual-frequency radar profiling algorithm. Journal of applied meteorology and climatology, 50(7), pp.1543-1557.

Carpenter, J.R., Bartlett, J.W., Morris, T.P., Wood, A.M., Quartagno, M. and Kenward, M.G., 2023. Multiple imputation and its application. John Wiley & Sons.

Johnson, R.A. and Wichern, D.W., 2002. Applied multivariate statistical analysis. Prentice Hall.

Short, D. A., and K. Nakamura, 2000: TRMM Radar Observations of Shallow Precipitation over the Tropical Oceans. J. Climate, 13, 4107–4124, https://doi.org/10.1175/1520-0442(2000)013<4107:TROOSP>2.0.CO;2.

[13] Chase, R. J., S. W. Nesbitt, and G. M. McFarquhar, 2021: A Dual-Frequency Radar Retrieval of Two Parameters of the Snowfall Particle Size Distribution Using a Neural Network. J. Appl. Meteor. Climatol., 60, 341–359, https://doi.org/10.1175/JAMC-D-20-0177.1.

### 3 Open Science and Data Management Plan
### 4 Biographical Sketches
### 5 Summary of Personnel and Work Effort
### 6 Current and Pending Support
### 7 Budget and Budget Justification
#### 7.1 Budget Justification Morgan State University
#### 7.2 Detailed Budget
#### 7.3 NASA Budget Justification: Narrative and Details
### 8 Special Notifications and/or Certifications