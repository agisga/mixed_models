---
title: 'mixed_models: Statistical linear mixed effects models in Ruby'
tags:
  - statistics 
  - linear mixed model 
  - regression 
  - ruby
authors:
 - name: Alexej Gossmann 
   orcid:
   affiliation: Tulane University, New Orleans, LA
 - name: Yu-Ping Wang 
   orcid:
   affiliation: Tulane University, New Orleans, LA
 - name: Pjotr Prins
   orcid: 0000-0002-8021-9162
   affiliation: University Medical Center Utrecht, The Netherlands, University of Tennessee Health Science Center, USA
date: 8 July 2016
bibliography: paper.bib
---

# Summary

The Ruby gem mixed_models can be used to fit statistical linear mixed models (LMM), and perform statistical inference on the model parameters, as well as to predict future observations (including prediction intervals).

Other software packages for LMM include the R packages lme4 [@Bates:2015] and nlme [@Pinheiro:2016], Python statistical library statsmodels, and the Julia package MixedModels.jl.

The parameter estimation in mixed_models is based on the (restricted) maximum likelihood approach developed by the authors of lme4, which is delineated in [@Bates:2015]. One of the goals was to improve code readability compared to the corresponding implementation in lme4. All internal matrix calculations are performed using the gem nmatrix.

Support for the formula language, similar to that of lme4, in mixed_models makes model specification flexible, convenient, and user friendly. Additional flexibility is provided by an option to specify the random effects covariance structure with a Ruby Proc. 

Many types of hypotheses tests for the fixed and random effects are available, such as Chi squared or bootstrap based likelihood ratio tests for nested models.
Also multiple types of confidence intervals for the fixed effects coefficient estimates are provided, based on bootstrap or Wald statistics. In order to speed up computation, all bootstrap based methods are performed in parallel by default.

# References
