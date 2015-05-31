# Copyright (c) 2015 Alexej Gossmann 
 
require 'nmatrix'

# A class to store all the information required to fit a linear mixed model and the 
# results of the model fit.
# The implementation is partially based on Bates et al. (2014) and the corresponding
# R implementations in the package lme4pureR (https://github.com/lme4/lme4pureR).
#
# === References
# 
# * Douglas Bates, Martin Maechler, Ben Bolker, Steve Walker, 
#   "Fitting Linear Mixed - Effects Models using lme4". arXiv:1406.5823v1 [stat.CO]. 2014.
#
class LMMData

  # Create an object which stores all the information required to fit a linear mixed model
  # as well as the results of the model fit, such as various matrices and some of their 
  # products, the parameterization of the random effects covariance Cholesky factor as a Proc object,
  # various parameter estimates, etc.
  #
  # === Arguments
  #
  # * +x+         - fixed effects model matrix
  # * +y+         - response
  # * +zt+        - transpose of the random effects model matrix
  # * +lambdat+   - upper triangular Cholesky factor of the
  #                 relative covariance matrix of the random effects.
  # * +weights+   - optional array of prior weights
  # * +offset+    - offset
  # * +thfun+     - a +Proc+ object that takes a value of +theta+ and produces
  #                 the non-zero elements of +Lambdat+.  The structure of +Lambdat+
  #                 cannot change, only the numerical values.
  #
  def initialize(x:, y:, zt:, lambdat:, weights: nil, offset: 0.0, &thfun)
    unless x.is_a?(NMatrix) and y.is_a?(NMatrix) and zt.is_a?(NMatrix) and lambdat.is_a?(NMatrix)
      raise ArgumentError, "x, y, zt, lambdat should be passed as NMatrix objects"
    end
    raise ArgumentError, "y.shape should be of the form [n,1]" unless y.shape[1]==1
    raise ArgumentError, "weights should be passed as an array or nil" unless weights.is_a?(Array) or weights.nil?

    @x, @y, @zt, @lambdat, @weights, @offset, @thfun = x, y, zt, lambdat, weights, offset, thfun 
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
    tmp_mat1 = @lambdat.dot @ztw
    tmp_mat2 = (tmp_mat1.dot tmp_mat1.transpose) + NMatrix.identity(@q)
    @l = tmp_mat2.factorize_cholesky[1]            # lower triangular Cholesky factor 
    tmp_mat1, tmp_mat2 = nil, nil
    #@lambdat_ini = @lambdat                       # b/c it will be updated
    @mu = NMatrix.new([@n,1], dtype: :float64)     # conditional mean of response
    @u = NMatrix.new([@q,1], dtype: :float64)      # conditional mode of spherical random effects
  end

  # Attribute readers available for the following attributes:
  #
  # * +x+         - fixed effects model matrix
  # * +y+         - response
  # * +zt+        - transpose of the random effects model matrix
  # * +lambdat+   - upper triangular Cholesky factor of the
  #                 relative covariance matrix of the random effects.
  # * +weights+   - optional array of prior weights
  # * +offset+    - offset
  # * +thfun+     - a +Proc+ object that takes a value of +theta+ and produces
  #                 the non-zero elements of +Lambdat+.  The structure of +Lambdat+
  #                 cannot change, only the numerical values.
  # * +n+         - length of +y+, i.e. size of the data
  # * +p+         - number of columns of +x+, i.e. number of fixed effects covariates
  # * +q+         - number of rows of +zt+, i.e. number of random effects terms
  # * +sqrtw+     - diagonal matrix with the square roots of +weights+ on the diagonal
  # * +ztw+       - matrix product z^T * sqrtw 
  # * +xtwx+      - matrix product x^T * w * x
  # * +xtwy+      - matrix product x^T * w * y
  # * +ztwx+      - matrix product z^T * w * x
  # * +ztwy+      - matrix product z^T * w * y
  #
  attr_reader :x, :y, :zt, :lambdat, :weights, :offset, :thfun, :n, :p, :q, :sqrtw, 
    :ztw, :xtwx, :xtwy, :ztwx, :ztwy

  # Attribute readers and writers available for the following attributes.
  # These attributes will be typically changed during the evaluation of
  # the deviance function or the REML criterion.
  #
  # * +b+       - conditional mode of random effects
  # * +beta+    - conditional estimate of fixed-effects
  # * +l+       - lower triangular Cholesky factor of [lambda^T * z^T * w * z * lambda + identity]
  # * +lambdat+ - upper triangular Cholesky factor of the relative covariance matrix of the random effects.
  # * +mu+      - conditional mean of response
  # * +u+       - conditional mode of spherical random effects
  #
  attr_accessor :lambdat, :b, :beta, :l, :lambdat, :mu, :u

end
