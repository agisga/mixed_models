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

  attr_reader :reml, :theta_optimal, :dev_optimal, :dev_fun, :optimization_result, :model_data

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
    @theta_optimal = @optimization_result.x_minimum # optimal solution for theta
    @dev_optimal   = @optimization_result.f_minimum # function value at the optimal solution
  end

  # An Array containing the estimated fixed effects coefficiants.
  # These estimates are conditional on the estimated covariance parameters.
  #
  def fixed_effects 
    @model_data.beta.to_flat_a
  end

  # An Array containing the estimated mean values of the random effects.
  # These are conditional estimates which depend on the input data.
  #
  def random_effects
    @model_data.b.to_flat_a
  end

  # An Array containing the fitted response values, i.e. the estimated mean of the response
  # (conditional on the estimates of the covariance parameters, the random effects,
  # and the fixed effects).
  def fitted_values
    @model_data.mu.to_flat_a
  end

  # An Array containing the model residuals, which are defined as the difference between the
  # response and the fitted values.
  #
  def residuals
    (@model_data.y - @model_data.mu).to_flat_a
  end
  
  # Sum of squared residuals
  #
  def sse
    s = 0.0
    self.residuals.each { |r| s += r**2 }
    return s
  end

  # Mean squared error, defined as the mean
  # of the squares of the differences between the response
  # and the fitted values.
  #
  def mse
    self.sse / @model_data.n
  end
end
