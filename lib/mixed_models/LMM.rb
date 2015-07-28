# Copyright (c) 2015 Alexej Gossmann 
 
# Linear mixed effects models.
# The implementation is based on Bates et al. (2014)
#
# === References
# 
# * Douglas Bates, Martin Maechler, Ben Bolker, Steve Walker, 
#   "Fitting Linear Mixed - Effects Models using lme4". arXiv:1406.5823v1 [stat.CO]. 2014.
#
class LMM

  attr_reader :reml, :formula, :dev_fun, :optimization_result, :model_data,
              :sigma2, :sigma_mat, :fix_ef, :ran_ef, :fix_ef_names, :ran_ef_names

  # Fit and store a linear mixed effects model according to the input from the user.
  # Parameter estimates are obtained by the method described in Bates et. al. (2014).
  #
  # === Arguments
  #
  # * +x+              - fixed effects model matrix as a dense NMatrix
  # * +y+              - response vector as a nx1 dense NMatrix
  # * +zt+             - transpose of the random effects model matrix as a dense NMatrix
  # * +x_col_names     - (Optional) column names for the matrix +x+, i.e. the names of the fixed
  #                      effects terms
  # * +z_col_names     - (Optional) column names for the matrix z, i.e. row names for the matrix
  #                      +zt+, i.e. the names of the random effects terms
  # * +weights+        - (Optional) Array of prior weights
  # * +offset+         - an optional vector of offset terms which are known 
  #                      a priori; a nx1 NMatrix
  # * +reml+           - if true than the profiled REML criterion will be used as the objective
  #                      function for the minimization; if false then the profiled deviance 
  #                      will be used; defaults to true
  # * +start_point+    - an Array specifying the initial parameter estimates for the 
  #                      minimization
  # * +lower_bound+    - an optional Array of lower bounds for each coordinate of the optimal 
  #                      solution 
  # * +upper_bound+    - an optional Array of upper bounds for each coordinate of the optimal 
  #                      solution 
  # * +epsilon+        - a small number specifying the thresholds for the convergence check 
  #                      of the optimization algorithm; see the respective documentation for 
  #                      more detail
  # * +max_iterations+ - the maximum number of iterations for the optimization algorithm
  # * +from_daru_args+ - (! Never used in a direct call of #initialize) a Hash, storinig some 
  #                      arguments supplied to #from_daru (except the data set and the arguments 
  #                      that #from_daru shares with #initialize), if #initilize was originally 
  #                      called from within the #from_daru method
  # * +formula+        - (! Never used in a direct call of #initialize) a String containing the 
  #                      formula used to fit the model, if the model was fit by #from_formula
  # * +thfun+          - a block or +Proc+ object that takes in an Array +theta+ and produces
  #                      the non-zero elements of the dense NMatrix +lambdat+, which is the upper 
  #                      triangular Cholesky factor of the relative covariance matrix of the random 
  #                      effects. The structure of +lambdat+ cannot change, only the numerical values.
  #
  # === References
  # 
  # * Douglas Bates, Martin Maechler, Ben Bolker, Steve Walker, 
  #   "Fitting Linear Mixed - Effects Models using lme4". arXiv:1406.5823v1 [stat.CO]. 2014.
  #
  def initialize(x:, y:, zt:, x_col_names: nil, z_col_names: nil, weights: nil, 
                 offset: 0.0, reml: true, start_point:, lower_bound: nil, upper_bound: nil, 
                 epsilon: 1e-6, max_iterations: 1e6, from_daru_args: nil, formula: nil, &thfun) 
    @from_daru_args = from_daru_args
    @formula        = formula
    @reml           = reml

    ################################################
    # Fit the linear mixed model
    ################################################

    # (1) Create the data structure in a LMMData object
    lambdat_ini = thfun.call(start_point) #initial estimate of +lambdat+
    @model_data = LMMData.new(x: x, y: y, zt: zt, lambdat: lambdat_ini, 
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

    # scale parameter of the covariance; the residuals conditional on the random 
    # effects have variances "sigma2*weights^(-1)"; if all weights are ones then 
    # sigma2 is an estimate of the residual variance
    @sigma2 = if reml then
               @model_data.pwrss / (@model_data.n - @model_data.p)
             else
               @model_data.pwrss / @model_data.n
             end

    # estimate of the covariance matrix Sigma of the random effects vector b,
    # where b ~ N(0, Sigma).
    @sigma_mat = (@model_data.lambdat.transpose.dot @model_data.lambdat) * @sigma2

    # Array containing the names of the fixed effects terms
    @fix_ef_names = if x_col_names.nil? then
                      (0...@model_data.beta.shape[0]).map { |i| "x" + i.to_s } 
                    else
                      x_col_names
                    end
    # Hash containing the estimated fixed effects coefficiants (these estimates are 
    # conditional on the estimated covariance parameters).
    @fix_ef = Hash.new
    @fix_ef_names.each_with_index { |name, i| @fix_ef[name] = @model_data.beta[i] }
    
    # Array containing the names of the random effects terms
    @ran_ef_names = if z_col_names.nil? then 
                      (0...@model_data.b.shape[0]).map { |i| "z" + i.to_s }
                    else
                      z_col_names
                    end
    # Hash containing the estimated conditional mean values of the random effects terms
    # (these are conditional estimates which depend on the input data).
    @ran_ef = Hash.new
    @ran_ef_names.each_with_index { |name, i| @ran_ef[name] = @model_data.b[i] }
  end

  # Fit and store a linear mixed effects model, specified using a formula interface,
  # from data supplied as Daru::DataFrame. Parameter estimates are obtained via 
  # LMM#from_daru and LMM#initialize.
  #
  # The response variable, fixed effects and random effects are specified with a formula,
  # which uses a smaller version of the laguage of the formula interface to the R package lme4. 
  # Compared to lme4 the formula language here is somewhat restricted by not allowing the use
  # of operators "*" and "||" (equivalent formula formulations are always possible in
  # those cases, using only "+", ":" and "|"). Moreover, nested random effects and the 
  # corresponding operator "/" are not supported. Much detail on formula definitions 
  # can be found in the documentation, papers and tutorials to lme4.
  #
  # === Arguments
  #
  # * +formula+        - a String containing a two-sided linear formula describing both, the 
  #                      fixed effects and random effects of the model, with the response on 
  #                      the left of a ~ operator and the terms, separated by + operators, 
  #                      on the right hand side. Random effects specifications are in 
  #                      parentheses () and contain a vertical bar |. Expressions for design 
  #                      matrices are on the left of the vertical bar |, and grouping factors 
  #                      are on the right. 
  # * +data+           - a Daru::DataFrame object, containing the response, fixed and random 
  #                      effects, as well as the grouping variables
  # * +weights+        - optional Array of prior weights
  # * +offset+         - an optional vector of offset terms which are known 
  #                      a priori; a nx1 NMatrix
  # * +reml+           - if true than the profiled REML criterion will be used as the objective
  #                      function for the minimization; if false then the profiled deviance 
  #                      will be used; defaults to true
  # * +start_point+    - an optional Array specifying the initial parameter estimates for the 
  #                      minimization
  # * +epsilon+        - an optional  small number specifying the thresholds for the 
  #                      convergence check of the optimization algorithm; see the respective 
  #                      documentation for more detail
  # * +max_iterations+ - optional, the maximum number of iterations for the optimization 
  #                      algorithm
  #
  # === Usage
  #
  #   df = Daru::DataFrame.from_csv './data/alien_species.csv'
  #   model_fit = LMM.from_formula(formula: "Aggression ~ Age + Species + (Age | Location)", data: df)
  #    
  #   # Print some results:
  #   model_fit.fix_ef # => {:intercept=>1016.2867207696775, :Age=>-0.06531615343468071, :Species_lvl_Human=>-499.69369529020906, :Species_lvl_Ood=>-899.569321353577, :Species_lvl_WeepingAngel=>-199.58895804200768}
  #   model_fit.ran_ef # => {:intercept_Asylum=>-116.68080682806713, :Age_Asylum=>-0.03353391213061963, :intercept_Earth=>83.86571630094411, :Age_Earth=>-0.1361399664446193, :intercept_OodSphere=>32.81508992422786, :Age_OodSphere=>0.1696738785983933}
  #
  def LMM.from_formula(formula:, data:, weights: nil, offset: 0.0, reml: true, 
                       start_point: nil, epsilon: 1e-6, max_iterations: 1e6)
    raise(ArgumentError, "formula must be supplied as a String") unless formula.is_a? String
    raise(NotImplementedError, "The operator * is not supported in formula formulations." +
          "Use the operators + and : instead (e.g. a*b is equivalent to a+b+a:b)") if formula.include? "*"
    raise(NotImplementedError, "Nested random effects are not supported in formula formulation." +
          "The model can still be fit by adding a new column representing the nested grouping structure" +
          "to the data frame.") if formula.include? "/"
    raise(NotImplementedError, "The operator || is not supported in formula formulations." +
          "Reformulate the formula using single vertical lines | instead" +
          "(e.g. (Days || Subj) is equivalent to (1 | Subj) + (0 + Days | Subj))") if formula.include? "||"
    raise(NotImplementedError, "The notation -1 in not supported." +
          "Use 0 instead, in order to denote the exclusion of an intercept term.") if formula.include? "-1"
    raise(ArgumentError, "Cannot have a variable named 'intercept'." +
          "If you want to include an intercept term use '1'." +
          "If you want to exclude an intercept term use '0'.") if formula.include? "intercept"
    raise(ArgumentError, "formula must contain a '~' symbol") unless formula.include? "~"

    original_formula = formula.clone

    # remove whitespaces
    formula.gsub!(%r{\s+}, "")
    # replace ":" with "*", because the LMMFormula class requires this convention
    formula.gsub!(":", "*")

    # deal with the intercept: "intercept" is added to any model specification as a fixed and a random effect;
    # if a "0" term is specified as a fixed or random effect, then "no_intercept" will be included in the formula;
    # later, LMM#from_daru handles the case when both, "intercept" and "no_intercept", are specified.
    formula.gsub!("~", "~intercept+")
    formula.gsub!("(", "(intercept+")
    formula.gsub!("+1", "")
    formula.gsub!("+0", "+no_intercept")

    # extract the response and right hand side
    split = formula.split "~"
    response = split[0].strip
    raise(ArgumentError, "The left hand side of formula cannot be empty") if response.empty?
    response = response.to_sym
    rhs = split[1] 
    raise(ArgumentError, "The right hand side of formula cannot be empty") if rhs.split.empty?

    # get all variable names from rhs
    vars = rhs.split %r{\s*[+|()*]\s*}
    vars.delete("")
    vars.uniq!

    # In the String rhs, wrap each variable name "foo" in "MixedModels::lmm_variable(:foo)":
    # Put whitespaces around symbols "+", "|", "*", "(" and ")", and then
    # substitute "name" with "MixedModels::lmm_variable(name)" only if it is surrounded by white 
    # spaces; this trick is used to ensure that for example the variable "a" is not found within 
    # the variable "year"
    rhs.gsub!(%r{([+*|()])}, ' \1 ')
    rhs = " #{rhs} "
    vars.each { |name| rhs.gsub!(" #{name} ", "MixedModels::lmm_variable(#{name.to_sym.inspect})") } 

    # generate an LMMFormula of the right hand side
    rhs_lmm_formula = eval(rhs)

    #fit the model
    rhs_input = rhs_lmm_formula.to_input_for_lmm_from_daru
    return LMM.from_daru(response: response, 
                         fixed_effects: rhs_input[:fixed_effects],
                         random_effects: rhs_input[:random_effects], 
                         grouping: rhs_input[:grouping],
                         data: data, 
                         weights: weights, offset: offset, reml: reml, 
                         start_point: start_point, epsilon: epsilon, 
                         max_iterations: max_iterations, formula: original_formula)
  end

  # Fit and store a linear mixed effects model from data supplied as Daru::DataFrame.
  # Parameter estimates are obtained via LMM#initialize.
  #
  # The fixed effects are specified as an Array of Symbols (for the respective vectors in
  # the data frame).
  # The random effects are specified as an Array of Arrays, where the variables in each
  # sub-Array correspond to a common grouping factor (given in the Array +grouping+) and are 
  # modeled as correlated.
  # For both, fixed and random effects, +:intercept+ can be used to denote an intercept term; and 
  # +:no_intercept+ denotes the exclusion of an intercept term, even if +:intercept+ is given. 
  # An interaction effect can be included as an Array of length two, containing the 
  # respective variable names. 
  #
  # All non-numeric vectors in the data frame are considered to be categorical variables
  # and treated accordingly.
  #
  # Nested random effects are currently not supported by this interface.
  #
  # === Arguments
  #
  # * +response+       - name of the response variable in +data+
  # * +fixed_effects+  - names of the fixed effects in +data+, given as an Array. An 
  #                      interaction effect can be specified as Array of length two.
  #                      An intercept term can be denoted as +:intercept+; and 
  #                      +:no_intercept+ denotes the exclusion of an intercept term, even 
  #                      if +:intercept+ is given.
  # * +random_effects+ - names of the random effects in +data+, given as an Array of Arrays;
  #                      where the variables in each (inner) Array share a common grouping 
  #                      structure, and the corresponding random effects are modeled as 
  #                      correlated. An interaction effect can be specified as Array of 
  #                      length two. An intercept term can be denoted as +:intercept+; and 
  #                      +:no_intercept+ denotes the exclusion of an intercept term, even 
  #                      if +:intercept+ is given.
  # * +grouping+       - an Array of the names of the variables in +data+, which determine the
  #                      grouping structures for +random_effects+
  # * +data+           - a Daru::DataFrame object, containing the response, fixed and random 
  #                      effects, as well as the grouping variables
  # * +weights+        - optional Array of prior weights
  # * +offset+         - an optional vector of offset terms which are known 
  #                      a priori; a nx1 NMatrix
  # * +reml+           - if true than the profiled REML criterion will be used as the objective
  #                      function for the minimization; if false then the profiled deviance 
  #                      will be used; defaults to true
  # * +start_point+    - an optional Array specifying the initial parameter estimates for the 
  #                      minimization
  # * +epsilon+        - an optional  small number specifying the thresholds for the 
  #                      convergence check of the optimization algorithm; see the respective 
  #                      documentation for more detail
  # * +max_iterations+ - optional, the maximum number of iterations for the optimization 
  #                      algorithm
  # * +formula+        - (! Never used in a direct call of #from_daru) a String containing the 
  #                      formula used to fit the model, if the model was fit by #from_formula
  #
  # === Usage
  #
  #  df = Daru::DataFrame.from_csv './data/categorical_and_crossed_ran_ef.csv'
  #  model_fit = LMM.from_daru(response: :y, fixed_effects: [:intercept, :x, :a], 
  #                            random_effects: [[:intercept, :x], [:intercept, :a]], 
  #                            grouping: [:grp_for_x, :grp_for_a],
  #                            data: df)
  #  # Print some results:
  #  model_fit.dev_optimal # =>	342.7659264122803
  #  model_fit.fix_ef # => {:intercept=>10.146249586724727, :x=>0.6565521213078758, :a_lvl_b=>-4.4565184869223415, :a_lvl_c=>-0.6298761634372705, :a_lvl_d=>-2.9308041789327985, :a_lvl_e=>-1.342758616430962}
  #  model_fit.sigma # =>	0.9459691325482149
  #
  def LMM.from_daru(response:, fixed_effects:, random_effects:, grouping:, data:,
                    weights: nil, offset: 0.0, reml: true, start_point: nil,
                    epsilon: 1e-6, max_iterations: 1e6, formula: nil)
    raise(ArgumentError, "data should be a Daru::DataFrame") unless data.is_a?(Daru::DataFrame)

    given_args = Marshal.load(Marshal.dump({response: response, fixed_effects: fixed_effects,
                                            random_effects: random_effects, grouping: grouping}))

    n = data.size

    ################################################################
    # Adjust +data+, +fixed_effects+, +random_effects+ and 
    # +grouping+ for inclusion or exclusion of an intercept term, 
    # categorical variables, interaction effects and nested 
    # grouping factors
    ################################################################

    adjusted = MixedModels::adjust_lmm_from_daru_inputs(fixed_effects: fixed_effects, 
                                                        random_effects: random_effects, 
                                                        grouping: grouping, data: data)
    fixed_effects  = adjusted[:fixed_effects]
    random_effects = adjusted[:random_effects]
    grouping       = adjusted[:grouping]
    data           = adjusted[:data]

    ################################################################
    # Construct model matrices and vectors, covariance function,
    # and optimization parameters 
    ################################################################

    # construct the response vector
    y = NMatrix.new([n,1], data[response].to_a, dtype: :float64)

    # construct the fixed effects design matrix
    x_frame     = data[*fixed_effects]
    x           = x_frame.to_nm
    x_col_names = fixed_effects.clone # column names of the x matrix

    # construct the random effects model matrix and covariance function 
    num_groups = grouping.length
    raise(ArgumentError, "Length of +random_effects+ mismatches length of +grouping+") unless random_effects.length == num_groups
    ran_ef_raw_mat = Array.new
    ran_ef_grp     = Array.new
    num_ran_ef     = Array.new
    num_grp_levels = Array.new
    0.upto(num_groups-1) do |i|
      xi_frame = data[*random_effects[i]]
      ran_ef_raw_mat[i] = xi_frame.to_nm
      ran_ef_grp[i] = data[grouping[i]].to_a
      num_ran_ef[i] = ran_ef_raw_mat[i].shape[1]
      num_grp_levels[i] = ran_ef_grp[i].uniq.length
    end
    z_result_hash = MixedModels::mk_ran_ef_model_matrix(ran_ef_raw_mat, ran_ef_grp, random_effects)
    z             = z_result_hash[:z] 
    z_col_names   = z_result_hash[:names] # column names of the z matrix
    thfun         = MixedModels::mk_ran_ef_cov_fun(num_ran_ef, num_grp_levels)

    # define the starting point for the optimization (if it's nil),
    # such that random effects are independent with variances equal to one
    q = num_ran_ef.sum
    if start_point.nil? then
      tmp1 = Array.new(q) {1.0}
      tmp2 = Array.new(q*(q-1)/2) {0.0}
      start_point = tmp1 + tmp2
    end

    # set the lower bound on the variance-covariance parameters
    tmp1 = Array.new(q) {0.0}
    tmp2 = Array.new(q*(q-1)/2) {-Float::INFINITY}
    lower_bound = tmp1 + tmp2

    ####################
    # Fit the model
    ####################

    return LMM.new(x: x, y: y, zt: z.transpose,
                   x_col_names: x_col_names, z_col_names: z_col_names,
                   weights: weights, offset: offset, 
                   reml: reml, start_point: start_point, 
                   lower_bound: lower_bound, epsilon: epsilon, 
                   max_iterations: max_iterations, 
                   formula: formula, from_daru_args: given_args, 
                   &thfun)
  end

  # An Array containing the fitted response values, i.e. the estimated mean of the response
  # (conditional on the estimates of the covariance parameters, the random effects,
  # and the fixed effects).
  #
  # === Arguments
  #
  # * +with_ran_ef+ - specifies if random effects coefficients should be used; i.e. whether
  #                   the returned value is X*beta or X*beta+Z*b, where beta are the fixed
  #                   and b the random effects coefficients; default is true 
  #                   
  def fitted(with_ran_ef: true)
    if with_ran_ef then
      @model_data.mu.to_flat_a
    else
      @model_data.x.dot(@model_data.beta).to_flat_a
    end
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
    self.residuals.inject { |sse, r| sse + r**2.0 }
  end

  # Variance-covariance matrix of the estimates of the fixed effects terms, conditional on the
  # estimated covariance parameters, as given in equation (54) in Bates et. al. (2014).
  #
  # === References
  # 
  # * Douglas Bates, Martin Maechler, Ben Bolker, Steve Walker, 
  #   "Fitting Linear Mixed - Effects Models using lme4". arXiv:1406.5823v1 [stat.CO]. 2014.
  #
  def fix_ef_cov_mat
    return @model_data.rxtrx.inverse * @sigma2
  end

  # Returns a Hash containing the standard deviations of the estimated fixed effects 
  # coefficients (these estimates are conditional on the estimated covariance parameters).
  #
  def fix_ef_sd 
    result = Hash.new
    cov_mat = self.fix_ef_cov_mat
    @fix_ef_names.each_with_index { |name, i| result[name] = Math::sqrt(cov_mat[i,i]) }
    return result
  end

  # Returns a Hash containing the Wald z test statistics for each fixed effects coefficient.
  #
  def fix_ef_z
    sd = self.fix_ef_sd
    z  = Hash.new
    sd.each_key { |k| z[k] = @fix_ef[k] / sd[k] }
    return z
  end

  # Returns a Hash containing the p-values of the fixed effects coefficient.
  #
  # === Arguments
  #
  # * +method+ - determines the method used to compute the p-values;
  #              dafault and currently the only possibility is +:wald+,
  #              which denotes the Wald z test
  #
  def fix_ef_p(method: :wald)
    p = Hash.new

    case method
    when :wald
      z = self.fix_ef_z
      z.each_key { |k| p[k] = 2.0*(1.0 - Distribution::Normal.cdf(z[k].abs)) }
    else
      raise(NotImplementedError, "Method #{method} is currently not implemented")
    end

    return p
  end

  # Returns a Hash containing the confidence intervals of the fixed effects
  # coefficients.
  #
  # === Arguments
  #
  # * +level+  - confidence level, a number between 0 and 1
  # * +method+ - determines the method used to compute the confidence intervals;
  #              dafault and currently the only possibility is +:wald+, which 
  #              approximates the confidence intervals based on the Wald z test statistic 
  #
  def fix_ef_conf_int(level: 0.95, method: :wald)
    alpha = 1.0 - level
    conf_int = Hash.new

    case method
    when :wald
      z = Distribution::Normal.p_value(alpha/2.0).abs
      sd = self.fix_ef_sd
      @fix_ef.each_key do |k|
        conf_int[k] = Array.new
        conf_int[k][0] = @fix_ef[k] - z * sd[k]
        conf_int[k][1] = @fix_ef[k] + z * sd[k]
      end
    else
      raise(NotImplementedError, "Method #{method} is currently not implemented")
    end

    return conf_int
  end

  # Conditional variance-covariance matrix of the random effects estimates, based on
  # the assumption that the fixed effects vector beta, the covariance factor Lambda(theta)
  # and the scaling factor sigma are known (i.e. not random; the corresponding estimates are 
  # used as "true" values), as given in equation (58) in Bates et. al. (2014).
  #
  # === References
  # 
  # * Douglas Bates, Martin Maechler, Ben Bolker, Steve Walker, 
  #   "Fitting Linear Mixed - Effects Models using lme4". arXiv:1406.5823v1 [stat.CO]. 2014.
  #
  def cond_cov_mat_ran_ef
    #TODO: this can be more efficient with a method for triangular matrices
    linv = @model_data.l.invert
    v = (linv.transpose.dot linv) * @sigma2
    return (@model_data.lambdat.transpose.dot v).dot(@model_data.lambdat)
  end
    
  # Optimal solution for of the minimization of the deviance function or
  # the REML criterion (whichever was used to fit the model)
  #
  def theta
    return @optimization_result.x_minimum 
  end

  # Value of the deviance function or the REML criterion (whichever was used to fit the model)
  # at the optimal solution
  #
  def deviance
    return @optimization_result.f_minimum 
  end

  # The square root of +@sigma2+. It is the residual standard deviation if all weights are 
  #  equal to one
  #
  def sigma
    Math::sqrt(@sigma2)
  end

  # Predictions from the fitted model on new data, conditional on the estimated fixed and random 
  # effects coefficients. Predictions can be made with ot without the inclusion of random
  # effects terms. The data can be either supplied as a # Daru::DataFrame object +newdata+, 
  # or as raw fixed and random effects model matrices +x+ and +z+. If both, +newdata+ and 
  # +x+ are passed, then an error message is thrown. If neither is passed, then the 
  # predictions are computed with the data that was used to fit the model.
  #
  # === Arguments
  #
  # * +newdata+     - a Daru::DataFrame object containing the data for which the predictions
  #                   will be evaluated
  # * +x+           - fixed effects model matrix, a NMatrix object
  # * +z+           - random effects model matrix, a NMatrix object
  # * +with_ran_ef+ - indicator whether the random effects should be considered in the
  #                   predictions; i.e. whether the predictions are computed as x*beta
  #                   or as x*beta+z*b; default is true
  #
  # === Usage
  #
  #  df = Daru::DataFrame.from_csv './data/alien_species.csv'
  #  model_fit = LMM.from_formula(formula: "Aggression ~ Age + Species + (Age | Location)", data: df)
  #  # Predictions of aggression levels on a new data set:
  #  dfnew = Daru::DataFrame.from_csv './data/alien_species_newdata.csv'
  #  model_fit.predict(newdata: dfnew)
  #   # => [1070.9125752531213,
  #     182.45206492790766,
  #     -17.064468754763425,
  #     384.78815861991046,
  #     876.1240725686444,
  #     674.711339114886,
  #     1092.6985606350875,
  #     871.150885526236,
  #     687.4629975728096,
  #     -4.0162601001437395] 
  #
  def predict(newdata: nil, x: nil, z: nil, with_ran_ef: true)
    raise(ArgumentError, "EITHER pass newdata OR x and z OR nothing") if newdata && (x || z)
    raise(ArgumentError, "If you pass z you need to pass x as well") if z && x.nil?

    # predict from raw model matrices
    if x then
      y = x.dot(@model_data.beta)
      if with_ran_ef then
        raise(ArgumentError, "EITHER pass z OR set with_ran_ef to be false") if z.nil?
        y += z.dot(@model_data.b)
      end
    # predict from Daru::DataFrame
    elsif newdata then
      raise(ArgumentError, "LMM#predict does not work with a Daru::DataFrame," +
            "if the model was not fit using a Daru::DataFrame") if @from_daru_args.nil?
      # to prevent side effects on these parameters:
      fe = Marshal.load(Marshal.dump(@from_daru_args[:fixed_effects]))
      re = Marshal.load(Marshal.dump(@from_daru_args[:random_effects]))
      gr = Marshal.load(Marshal.dump(@from_daru_args[:grouping]))
    
      adjusted = MixedModels::adjust_lmm_from_daru_inputs(fixed_effects: fe, random_effects: re,
                                                          grouping: gr, data: newdata)
      newdata = adjusted[:data]
      fe, re, gr = adjusted[:fixed_effects], adjusted[:random_effects], adjusted[:grouping]
      
      # construct the fixed effects design matrix
      x_frame     = newdata[*@fix_ef_names]
      x           = x_frame.to_nm

      # construct the random effects model matrix and covariance function 
      num_groups     = gr.length
      ran_ef_raw_mat = Array.new
      ran_ef_grp     = Array.new
      num_ran_ef     = Array.new
      num_grp_levels = Array.new
      0.upto(num_groups-1) do |i|
        xi_frame = newdata[*re[i]]
        ran_ef_raw_mat[i] = xi_frame.to_nm
        ran_ef_grp[i] = newdata[gr[i]].to_a
        num_ran_ef[i] = ran_ef_raw_mat[i].shape[1]
        num_grp_levels[i] = ran_ef_grp[i].uniq.length
      end
      z = MixedModels::mk_ran_ef_model_matrix(ran_ef_raw_mat, ran_ef_grp, re)[:z]

      # compute the predictions
      y = x.dot(@model_data.beta)
      y += z.dot(@model_data.b) if with_ran_ef
    # predict on the data that was used to fit the model
    else
      y = with_ran_ef ? @model_data.mu : @model_data.x.dot(@model_data.beta)
    end

    # add the offset
    y += @model_data.offset 

    return y.to_flat_a
  end

  # Predictions and corresponding confidence or prediction intervals computed from the fitted 
  # model on new data. The intervals are computed based on the normal approximation.
  #
  # A confidence interval is an interval estimate of the mean value of the response for given 
  # covariates (i.e. a population parameter). A prediction interval is an interval estimate of 
  # a future observation. For further explanation of this distinction see for example: 
  # https://stat.ethz.ch/education/semesters/ss2010/seminar/06_Handout.pdf
  #
  # Note that both, confidence and prediction intervals, do not take the uncertainty in the
  # random effects estimates in to account. That is, if X is the design matrix, beta is the 
  # vector of fixed effects coefficient, and beta0 is the obtained estimate for beta, then the 
  # confidence and prediction intervals are centered at X*beta0.
  #
  # The data can be either supplied as a Daru::DataFrame object +newdata+, or as raw design 
  # matrix +x+ for the fixed effects. If both, +newdata+ and +x+ are passed, then an error 
  # message is thrown. If neither is passed, then the results are computed for the data that 
  # was used to fit the model. If prediction rather than confidence intervals are desired, and 
  # +x+ is used rather than +newdata+, then a random effects model matrix +z+ needs to be passed
  # as well.
  #
  # Returned is a Hash containing an Array of predictions, an Array of lower interval bounds, 
  # and an Array of upper interval bounds.
  #
  # === Arguments
  #
  # * +newdata+ - a Daru::DataFrame object containing the data for which the predictions
  #               will be evaluated
  # * +x+       - fixed effects model matrix, a NMatrix object
  # * +z+       - random effects model matrix, a NMatrix object
  # * +level+   - confidence level, a number between 0 and 1
  # * +type+    - +:confidence+ or +:prediction+ for confidence and prediction intervals
  #               respectively; see above for explanation of their difference
  #
  def predict_with_intervals(newdata: nil, x: nil, z: nil, level: 0.95, type: :confidence)
    raise(ArgumentError, "EITHER pass newdata OR x OR nothing") if newdata && x
    raise(ArgumentError, "If you pass z you need to pass x as well") if z && x.nil?
    raise(ArgumentError, "type should be :confidence or :prediction") unless (type == :confidence || type == :prediction)

    input_type = if x then
                   :from_raw
                 elsif newdata then
                   :from_daru
                 end

    ######################################################################
    # Obtain the design matrix +x+ and the predictions as point estimates
    ######################################################################

    # predict from raw model matrices
    case input_type
    when :from_raw
      y = x.dot(@model_data.beta)
    when :from_daru
      raise(ArgumentError, "LMM#predict does not work with a Daru::DataFrame," +
            "if the model was not fit using a Daru::DataFrame") if @from_daru_args.nil?
      # to prevent side effects on these parameters:
      fe = Marshal.load(Marshal.dump(@from_daru_args[:fixed_effects]))
      re = Marshal.load(Marshal.dump(@from_daru_args[:random_effects]))
      gr = Marshal.load(Marshal.dump(@from_daru_args[:grouping]))
    
      adjusted = MixedModels::adjust_lmm_from_daru_inputs(fixed_effects: fe, random_effects: re,
                                                          grouping: gr, data: newdata)
      newdata    = adjusted[:data]
      fe, re, gr = adjusted[:fixed_effects], adjusted[:random_effects], adjusted[:grouping]
      x_frame    = newdata[*@fix_ef_names]
      x          = x_frame.to_nm
      y          = x.dot(@model_data.beta)
    else
      # predict on the data that was used to fit the model
      x = @model_data.x
      y = x.dot(@model_data.beta)
    end

    # add the offset
    y += @model_data.offset 

    ###########################################################
    # Obtain the random effects model matrix +z+, if necessary
    ###########################################################
    
    if type == :prediction then
      case input_type
      when :from_raw
        raise(ArgumentError, "EITHER pass z OR set type to be :confidence") if z.nil?
      when :from_daru
        num_groups     = gr.length
        ran_ef_raw_mat = Array.new
        ran_ef_grp     = Array.new
        num_ran_ef     = Array.new
        num_grp_levels = Array.new
        0.upto(num_groups-1) do |i|
          xi_frame = newdata[*re[i]]
          ran_ef_raw_mat[i] = xi_frame.to_nm
          ran_ef_grp[i] = newdata[gr[i]].to_a
          num_ran_ef[i] = ran_ef_raw_mat[i].shape[1]
          num_grp_levels[i] = ran_ef_grp[i].uniq.length
        end
        z = MixedModels::mk_ran_ef_model_matrix(ran_ef_raw_mat, ran_ef_grp, re)[:z]
      else
        # use the data that was used to fit the model
        z = @model_data.zt.transpose
      end
    end

    #######################################
    # Compute the intervals
    #######################################
    
    y = y.to_flat_a

    # Array of variances for the confidence intervals
    y_var   = Array.new
    cov_mat = (x.dot self.fix_ef_cov_mat).dot x.transpose
    y.each_index { |i| y_var[i] = cov_mat[i,i] }
    # Adjust the variances for the prediction intervals
    if type == :prediction then
      unless (@model_data.weights.nil? || @model_data.weights.all? { |w| w == 1 }) then
        raise(ArgumentError, "Cannot construct prediction intervals" +
                             "if the model was fit with prior weights (other than all ones)")
      end
      z_sigma_zt = (z.dot @sigma_mat).dot z.transpose
      y.each_index { |i| y_var[i] += z_sigma_zt[i,i] + @sigma2 }
    end
    # Array of standard deviations
    y_sd = Array.new
    y.each_index { |i| y_sd[i] = Math::sqrt(y_var[i]) }

    # Use normal approximations to compute intervals
    alpha = 1.0 - level
    z = Distribution::Normal.p_value(alpha/2.0).abs
    y_lower, y_upper = Array.new, Array.new
    y.each_index do |i| 
      y_lower[i] = y[i] - z * y_sd[i]
      y_upper[i] = y[i] + z * y_sd[i]
    end

    return {pred: y, 
            "lower#{(level*100).to_int}".to_sym => y_lower,
            "upper#{(level*100).to_int}".to_sym => y_upper}
  end
end
