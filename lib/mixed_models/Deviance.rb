# Copyright (c) 2015 Alexej Gossmann 
#
# The following implementation is based on the paper Douglas Bates, Martin Maechler, 
# Ben Bolker, Steve Walker, "Fitting Linear Mixed - Effects Models using lme4" (arXiv:1406.5823v1 [stat.CO]. 2014),
# and the corresponding R implementation in the package lme4pureR (https://github.com/lme4/lme4pureR).
#

module MixedModels

  # Generate the linear mixed model profiled deviance function 
  # or the REML criterion as a Proc object
  #
  # === Arguments
  #
  # * +d+    - an object of class LMMData supplying the data
  #   required for the deviance/REML function evaluations
  # * +reml+ - indicator whether to calculate REML criterion 
  #
  # === References
  # 
  # * Douglas Bates, Martin Maechler, Ben Bolker, Steve Walker, 
  #   "Fitting Linear Mixed - Effects Models using lme4". arXiv:1406.5823v1 [stat.CO]. 2014.
  #  
  def MixedModels.mk_lmm_dev_fun(d, reml)
    df     = (reml ? (d.n - d.p) : d.n) # degrees of freedom (depends on REML)
    cu     = NMatrix.new([d.q, 1], dtype: :float64)
    rzx    = NMatrix.new([d.q, d.p], dtype: :float64)
    wtres  = NMatrix.new([d.n, 1], dtype: :float64)
    logdet = Float::INFINITY

    Proc.new do |theta|
      # (1) update the covariance factor +lambdat+ and the Cholesky factor +l+ 
      d.lambdat = d.thfun.call(theta)
      tmp_mat1 = d.lambdat.dot d.ztw
      tmp_mat2 = (tmp_mat1.dot tmp_mat1.transpose) + NMatrix.identity(d.q)
      d.l = tmp_mat2.factorize_cholesky[1] 
      tmp_mat1, tmp_mat2 = nil, nil
      
      # (2) solve the normal equations to get estimates +beta+, +u+ and +b+
      cu = d.l.triangular_solve(:lower, d.lambdat.dot(d.ztwy))
      rzx = d.l.triangular_solve(:lower, d.lambdat.dot(d.ztwx))
      d.rxtrx = d.xtwx - (rzx.transpose.dot(rzx))
      d.beta = d.rxtrx.solve(d.xtwy - rzx.transpose.dot(cu))
      d.u = d.l.transpose.triangular_solve(:upper, (cu - rzx.dot(d.beta)))
      d.b = d.lambdat.transpose.dot(d.u)
      
      # (3) update the predictor of the response and the weighted residuals
      d.mu = (d.x.dot d.beta) + (d.zt.transpose.dot d.b) + d.offset
      wtres = d.sqrtw.dot (d.y-d.mu)

      # (4) evaluate the profiled deviance or the REML criterion
      d.pwrss = (wtres.norm2)**2.0 + (d.u.norm2)**2.0 # penalized weighted residual sum-of-squares
      logdet = 2.0 * Math::log(d.l.det.abs) 
      logdet += Math::log(d.rxtrx.det.abs) if reml
      logdet + df * (1.0 + Math::log(2.0 * Math::PI * d.pwrss) - Math::log(df))
    end
  end
end
