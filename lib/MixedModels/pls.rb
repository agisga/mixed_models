require 'nmatrix'

# Create linear mixed model profiled deviance function
# using the penalized least squares (PLS) approach.
#
class Deviance 

  # Create linear mixed model pprofiled deviance function
  # or the REML criterion the penalized least squares (PLS) approach.
  #
  # === Arguments
  #
  # * +x+ - fixed effects model matrix
  # * +y+ - response
  # * +zt+ - transpose of the sparse model matrix for the random effects
  # * +lambdat+ - upper triangular sparse Cholesky factor of the
  #               relative covariance matrix of the random effects.
  # * +thfun+ - a +Proc+ object that takes a value of +theta+ and produces
  #             the non-zero elements of +Lambdat+.  The structure of +Lambdat+
  #             cannot change, only the numerical values.
  # * +weights+ - optional array of prior weights
  # * +offset+ - offset
  # * +reml_flag+ - indicator whether to calculate REML criterion 
  #
  def initialize(x:, y:, zt:, lambdat:, thfun:, weights: nil, offset: nil, reml_flag: true)
    unless x.is_a?(NMatrix) and y.is_a?(NMatrix) and zt.is_a?(NMatrix) and lambdat.is_a?(NMatrix)
      raise ArgumentError, "x, y, zt, lambdat should be passed as NMatrix objects"
    end
    raise ArgumentError, "y.shape should be of the form [n,1]" unless y.shape[1]==1
    raise ArgumentError, "weights should be passed as an array" unless weights.is_a?(Array)

    @x, @y, @zt, @lambdat, @thfun, @weights, @offset, @reml_flag = x, y, zt, lambdat, thfun, weights, offset, reml_flag 
    @n, @p, @q = @y.shape[0], @X.shape[1], @Zt.shape[0]
 
    unless @x.shape[0]==@n and @zt.shape[1]==@n and @lambdat.shape[0]==@q and 
           @lambdat.shape[1]==@q and @weights.length==@n
      raise ArgumentError, "Dimensions mismatch"
    end

    @sqrtw = if @weights.nil?
               NMatrix.identity(@n, dtype: :float64)
             else
               NMatrix.diagonal(@weights, dtype: :float64)
             end
    
    wx = sqrtw.dot x
    wy = sqrtw.dot y
    @ztw = @zt.dot @sqrtw
    @xtwx = wx.transpose.dot wx
    @xtwy = wx.transpose.dot Wy
    @ztwx = @ztw.dot wx
    @ztwy = @ztw.dot wy
    wx = nil
    wy = nil

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
    @df = @df - @p if @reml_flag
  end

  def eval(theta)
    # update the covariance factor +@lambdat@ and the Cholesky factor +@l+ 
    @lambdat = @thfun.call(theta)
    tmp_mat1 = @lambdat.dot @ztw
    tmp_mat2 = (tmp_mat1.dot tmp_mat1.transpose) + NMatrix.identity(@q)
    @l = tmp_mat2.factorize_cholesky[1] 
    tmp_mat1, tmp_mat2 = nil, nil
    
    # solve the normal equations to get estimates +@beta+, +@u+ and +@b+
    @cu = @l.solve(@lambdat.dot @ztwy) 
    @rzx = @l.solve(@lambdat.dot @ztwx)
    @rxtrx = @xtwx - (@rzx.transpose.dot @rzx)
    @beta = @rxtrx.solve(@xtwy - (@rzx.transpose.dot @cu))
    @u = @l.transpose.solve(@cu - (@rzx.dot @beta))
    @b = @lambdat.transpose.dot @u
    
    # update the predictor of the response and the weighted residuals
    @mu = (@x.dot @beta) + (@zt.transpose.dot @b) + @offset
    @wtres = @sqrtw.dot (@y-@mu)

    # evaluate the profiled deviance
    # TODO: is NMatrix#norm2 more efficient here?
    @pwrss <- (@wtres**2.0).sum + (@u**2).sum # penalized, weighted residual sum-of-squares
    @logdet = 2.0 * Math::log(@l.abs.det) 
    @logdet += Math::log(@rxtrx.abs.det) if @reml_flag
                                # profiled deviance or REML
                                # criterion (eqns. 34, 41)
    deviance = @logdet + @df * (1.0 + Math::log(2.0 * Math::PI * @pwrss) - Math::log(@df))
    return deviance
  end
end
