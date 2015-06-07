# Copyright (c) 2015 Alexej Gossmann 
 
require 'nmatrix'

# Linear mixed effects models.
# The implementation is based on Bates et al. (2014)
#
# === References
# 
# * Douglas Bates, Martin Maechler, Ben Bolker, Steve Walker, 
#   "Fitting Linear Mixed - Effects Models using lme4". arXiv:1406.5823v1 [stat.CO]. 2014.
#
class LMM

  attr_reader :reml, :theta_optimal, :dev_optimal, :dev_fun, :optimization_result, :model_data,
              :sigma2, :sigma_mat, :fix_ef_cov_mat, :ran_ef_cov_mat, :sse, :fix_ef, :ran_ef

  # Fit and store a linear mixed effects model according to the input from the user.
  # Parameter estimates are obtained by the method described in Bates et. al. (2014).
  #
  # === Arguments
  #
  # * +x+              - fixed effects model matrix as a dense NMatrix
  # * +y+              - response vector as a nx1 dense NMatrix
  # * +zt+             - transpose of the random effects model matrix as a dense NMatrix
  # * +lambdat+        - upper triangular Cholesky factor of the relative 
  #                      covariance matrix of the random effects; a dense NMatrix
  # * +weights+        - optional Array of prior weights
  # * +offset+         - an optional vector of offset terms which are known a priori; a nx1 NMatrix
  # * +reml+           - if true than the profiled REML criterion will be used as the objective
  #                      function for the minimization; if false then the profiled deviance 
  #                      will be used; defaults to true
  # * +start_point+    - an Array specifying the initial parameter estimates for the minimization
  # * +lower_bound+    - an optional Array of lower bounds for each coordinate of the optimal solution 
  # * +upper_bound+    - an optional Array of upper bounds for each coordinate of the optimal solution 
  # * +epsilon+        - a small number specifying the thresholds for the convergence check 
  #                      of the optimization algorithm; see the respective documentation for 
  #                      more detail
  # * +max_iterations+ - the maximum number of iterations for the optimization algorithm
  # * +thfun+          - a block or +Proc+ object that takes a value of +theta+ and produces
  #                      the non-zero elements of +Lambdat+.  The structure of +Lambdat+
  #                      cannot change, only the numerical values.
  # === References
  # 
  # * Douglas Bates, Martin Maechler, Ben Bolker, Steve Walker, 
  #   "Fitting Linear Mixed - Effects Models using lme4". arXiv:1406.5823v1 [stat.CO]. 2014.
  #
  def initialize(x:, y:, zt:, lambdat:, weights: nil, offset: 0.0, reml: true, 
                 start_point:, lower_bound: nil, upper_bound: nil, epsilon: 1e-6, 
                 max_iterations: 1e6, &thfun)
    @reml = reml

    ################################################
    # Fit the linear mixed model
    ################################################

    # (1) Create the data structure in a LMMData object
    @model_data = LMMData.new(x: x, y: y, zt: zt, lambdat: lambdat, 
                              weights: weights, offset: offset, &thfun)
    
    # (2) Set up the profiled deviance/REML function
    @dev_fun = MixedModels::mk_lmm_dev_fun(@model_data, @reml)

    # (3) Optimize the deviance/REML
    @optimization_result = MixedModels::NelderMead.minimize(start_point: start_point, 
                                                            lower_bound: lower_bound, 
                                                            upper_bound: upper_bound,
                                                            epsilon: epsilon, 
                                                            max_iterations: max_iterations,
                                                            &dev_fun)

    #################################################
    # Compute and store some output parameters
    #################################################
    
    # optimal solution for theta
    @theta_optimal = @optimization_result.x_minimum 
    # function value at the optimal solution
    @dev_optimal   = @optimization_result.f_minimum 

    # sum of squared residuals
    @sse = 0.0
    self.residuals.each { |r| @sse += r**2 }

    # scale parameter of the covariance (the residuals conditional on the random 
    # effects have variances "sigma2*weights^(-1)"; if all weights are ones then 
    # sigma2 is an estimate of the residual variance)
    @sigma2 = if reml then
               @model_data.pwrss / (@model_data.n - @model_data.p)
             else
               @model_data.pwrss / @model_data.n
             end

    # estimate of the covariance matrix Sigma of the random effects vector b,
    # where b ~ N(0, Sigma).
    @sigma_mat = (@model_data.lambdat.transpose.dot @model_data.lambdat) * @sigma2

    # variance-covariance matrix of the random effects estimates, conditional on the
    # input data, as given in equation (58) in Bates et. al. (2014).
    # TODO: this can be done more efficiently because l is a lower-triangular matrix
    linv = @model_data.l.inverse
    v = linv.transpose.dot linv
    v = v * sigma2
    @ran_ef_cov_mat = (@model_data.lambdat.transpose.dot v).dot @model_data.lambdat

    # variance-covariance matrix of the fixed effects estimates, conditional on the
    # input data, as given in equation (54) in Bates et. al. (2014).
    @fix_ef_cov_mat = @model_data.rxtrx.inverse * @sigma2

    # Construct a Hash containing information about the estimated fixed effects 
    # coefficiants (these estimates are conditional on the estimated covariance parameters).
    @fix_ef = Hash.new
    fix_ef_names = (0...@model_data.beta.shape[0]).map { |i| "x" + i.to_s }
    fix_ef_names.each_with_index { |name, i| @fix_ef[name] = @model_data.beta[i] }
    # TODO: store more info in fix_ef, such as p-values on 95%CI
    
    # Construct a Hash containing information about the estimated mean values of the 
    # random effects (these are conditional estimates which depend on the input data).
    @ran_ef = Hash.new
    ran_ef_names = (0...@model_data.b.shape[0]).map { |i| "z" + i.to_s }
    ran_ef_names.each_with_index { |name, i| @ran_ef[name] = @model_data.b[i] }
    # TODO: store more info in ran_ef, such as p-values on 95%CI
  end

  # An Array containing the fitted response values, i.e. the estimated mean of the response
  # (conditional on the estimates of the covariance parameters, the random effects,
  # and the fixed effects).
  def fitted
    @model_data.mu.to_flat_a
  end

  # An Array containing the model residuals, which are defined as the difference between the
  # response and the fitted values.
  #
  def residuals
    (@model_data.y - @model_data.mu).to_flat_a
  end
end
