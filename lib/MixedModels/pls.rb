# Copyright (c) 2015 Alexej Gossmann 
#
# The following implementation is largely based on the paper Douglas Bates, Martin Maechler, 
# Ben Bolker, Steve Walker, "Fitting Linear Mixed - Effects Models using lme4" (arXiv:1406.5823v1 [stat.CO]. 2014),
# and the corresponding R implementation in the package lme4pureR (https://github.com/lme4/lme4pureR).
#
 
require 'nmatrix'

# Create linear mixed model profiled deviance function
# using the penalized least squares (PLS) approach.
# The implementation is based on Bates et al. (2014) and the corresponding
# R implementations in the package lme4pureR (https://github.com/lme4/lme4pureR).
#
# === References
# 
# * Douglas Bates, Martin Maechler, Ben Bolker, Steve Walker, 
#   "Fitting Linear Mixed - Effects Models using lme4". arXiv:1406.5823v1 [stat.CO]. 2014.
#
class Deviance 

  # Create linear mixed model profiled deviance function
  # or the REML criterion using the penalized least squares (PLS) approach.
  #
  # === Arguments
  #
  # * +x+ - fixed effects model matrix
  # * +y+ - response
  # * +zt+ - transpose of the random effects model matrix
  # * +lambdat+ - upper triangular Cholesky factor of the
  #               relative covariance matrix of the random effects.
  # * +thfun+ - a +Proc+ object that takes a value of +theta+ and produces
  #             the non-zero elements of +Lambdat+.  The structure of +Lambdat+
  #             cannot change, only the numerical values.
  # * +weights+ - optional array of prior weights
  # * +offset+ - offset
  # * +reml_flag+ - indicator whether to calculate REML criterion 
  #
  def initialize(x:, y:, zt:, lambdat:, thfun:, weights: nil, offset: 0.0, reml_flag: true)
    unless x.is_a?(NMatrix) and y.is_a?(NMatrix) and zt.is_a?(NMatrix) and lambdat.is_a?(NMatrix)
      raise ArgumentError, "x, y, zt, lambdat should be passed as NMatrix objects"
    end
    raise ArgumentError, "y.shape should be of the form [n,1]" unless y.shape[1]==1
    raise ArgumentError, "weights should be passed as an array or nil" unless weights.is_a?(Array) or weights.nil?

    @x, @y, @zt, @lambdat, @thfun, @weights, @offset, @reml_flag = x, y, zt, lambdat, thfun, weights, offset, reml_flag 
    @n, @p, @q = @y.shape[0], @x.shape[1], @zt.shape[0]
 
    unless @x.shape[0]==@n and @zt.shape[1]==@n and @lambdat.shape[0]==@q and @lambdat.shape[1]==@q
      raise ArgumentError, "Dimensions mismatch"
    end

    @sqrtw = if @weights.nil?
               NMatrix.identity(@n, dtype: :float64)
             else
               raise ArgumentError, "weights should have the same length as the response vector y" unless @weights.length==@n
               NMatrix.diagonal(weights.map { |w| Math::sqrt(w) }, dtype: :float64)
             end
    
    wx = @sqrtw.dot @x
    wy = @sqrtw.dot @y
    @ztw = @zt.dot @sqrtw
    @xtwx = wx.transpose.dot wx
    @xtwy = wx.transpose.dot wy
    @ztwx = @ztw.dot wx
    @ztwy = @ztw.dot wy
    wx, wy = nil, nil

    @b = NMatrix.new([@q,1], dtype: :float64)      # conditional mode of random effects
    @beta = NMatrix.new([@p,1], dtype: :float64)   # conditional estimate of fixed-effects
    @cu = NMatrix.new([@q,1], dtype: :float64)     # intermediate solution
    @rxtrx = @xtwx                                 # down-dated xtwx
    tmp_mat1 = @lambdat.dot @ztw
    tmp_mat2 = (tmp_mat1.dot tmp_mat1.transpose) + NMatrix.identity(@q)
    @l = tmp_mat2.factorize_cholesky[1]            # lower triangular Cholesky factor 
    tmp_mat1, tmp_mat2 = nil, nil
    #@lambdat_ini = @lambdat                        # b/c it will be updated
    @mu = NMatrix.new([@n,1], dtype: :float64)     # conditional mean of response
    @rzx = NMatrix.zeros([@q,@p], dtype: :float64) # intermediate matrix in solution
    @u = NMatrix.new([@q,1], dtype: :float64)      # conditional mode of spherical random effects
    @df = @n                                       # degrees of freedom (depends on REML)
    @df -= @p if @reml_flag
  end

  # Evaluate the linear mixed model profiled deviance function
  # or the REML criterion at the value +theta+
  #
  # === Arguments
  #
  # * +theta+ - an array compatible with +thfun+ from Deviance#initialize
  #  
  def eval(theta)
    # update the covariance factor +@lambdat@ and the Cholesky factor +@l+ 
    @lambdat = @thfun.call(theta)
    tmp_mat1 = @lambdat.dot @ztw
    tmp_mat2 = (tmp_mat1.dot tmp_mat1.transpose) + NMatrix.identity(@q)
    @l = tmp_mat2.factorize_cholesky[1] 
    tmp_mat1, tmp_mat2 = nil, nil
    
    # solve the normal equations to get estimates +@beta+, +@u+ and +@b+
    # TODO: use a triangular solve method where appropriate
    @cu = @l.solve(@lambdat.dot @ztwy) 
    #@rzx = @l.solve(@lambdat.dot @ztwx)
    @rzx = @l.inverse.dot(@lambdat.dot @ztwx) #TODO: make a solve method for this
    @rxtrx = @xtwx - (@rzx.transpose.dot @rzx)
    @beta = @rxtrx.solve(@xtwy - (@rzx.transpose.dot @cu))
    @u = @l.transpose.solve(@cu - (@rzx.dot @beta))
    @b = @lambdat.transpose.dot @u
    
    # update the predictor of the response and the weighted residuals
    @mu = (@x.dot @beta) + (@zt.transpose.dot @b) + @offset
    @wtres = @sqrtw.dot (@y-@mu)

    # evaluate the profiled deviance or the REML criterion
    @pwrss = (@wtres.norm2)**2.0 + (@u.norm2)**2.0 # penalized, weighted residual sum-of-squares
    @logdet = 2.0 * Math::log(@l.det.abs) 
    @logdet += Math::log(@rxtrx.det.abs) if @reml_flag
    deviance = @logdet + @df * (1.0 + Math::log(2.0 * Math::PI * @pwrss) - Math::log(@df))

    return deviance
  end
end
