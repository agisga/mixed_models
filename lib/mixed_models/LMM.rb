# Copyright (c) 2015 Alexej Gossmann 
 
# Linear mixed effects models.
# The implementation of the model fitting algorithm is based on Bates et al. (2014)
#
# === References
# 
# * Douglas Bates, Martin Maechler, Ben Bolker, Steve Walker, 
#   "Fitting Linear Mixed - Effects Models using lme4". arXiv:1406.5823v1 [stat.CO]. 2014.
#
class LMM

  # indicator whether the REML criterion or the deviance function was used
  attr_reader :reml 
  # formula used to fit the model
  attr_reader :formula 
  # deviance function or REML criterion as a Proc
  attr_reader :dev_fun 
  # object returned by the optimization routine
  attr_reader :optimization_result
  # object of class LMMData containing all model matrices etc
  attr_reader :model_data
  # covariance scaling factor (residual variance if no weights were used)
  attr_reader :sigma2
  # covariance matrix of the random effects vector
  attr_reader :sigma_mat
  # fixed effect coefficients estimates
  attr_reader :fix_ef
  # random effects coefficients estimates
  attr_reader :ran_ef
  # names of the fixed effects coefficients
  attr_reader :fix_ef_names
  # names of the random effects coefficients
  attr_reader :ran_ef_names
  # Hash storing some model specification information supplied to #from_daru
  attr_reader :from_daru_args

  # Fit and store a linear mixed effects model according to the input from the user.
  # Parameter estimates are obtained by the method described in Bates et. al. (2014).
  #
  # === Arguments
  #
  # * +x+              - fixed effects model matrix as a dense NMatrix
  # * +y+              - response vector as a nx1 dense NMatrix
  # * +zt+             - transpose of the random effects model matrix as a dense NMatrix
  # * +x_col_names+    - (Optional) column names for the matrix +x+, i.e. the names of the fixed
  #   effects terms
  # * +z_col_names+    - (Optional) column names for the matrix z, i.e. row names for the matrix
  #   +zt+, i.e. the names of the random effects terms
  # * +weights+        - (Optional) Array of prior weights
  # * +offset+         - an optional vector of offset terms which are known a priori; a nx1 NMatrix
  # * +reml+           - if true than the profiled REML criterion will be used as the objective
  #   function for the minimization; if false then the profiled deviance will be used; defaults to true
  # * +start_point+    - an Array specifying the initial parameter estimates for the minimization
  # * +lower_bound+    - an optional Array of lower bounds for each coordinate of the optimal solution 
  # * +upper_bound+    - an optional Array of upper bounds for each coordinate of the optimal solution 
  # * +epsilon+        - a small number specifying the thresholds for the convergence check 
  #   of the optimization algorithm; see the respective documentation for more detail
  # * +max_iterations+ - the maximum number of iterations for the optimization algorithm
  # * +from_daru_args+ - (! Never used in a direct call of #initialize) a Hash, storinig some 
  #   arguments supplied to #from_daru (except the data set and the arguments that #from_daru shares 
  #   with #initialize), if #initilize was originally called from within the #from_daru method
  # * +formula+        - (! Never used in a direct call of #initialize) a String containing the 
  #   formula used to fit the model, if the model was fit by #from_formula
  # * +thfun+          - a block or +Proc+ object that takes in an Array +theta+ and produces
  #   the non-zero elements of the dense NMatrix +lambdat+, which is the upper triangular Cholesky 
  #   factor of the relative covariance matrix of the random effects. The structure of +lambdat+ cannot 
  #   change, only the numerical values.
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
                      (0...@model_data.beta.shape[0]).map { |i| "x#{i}".to_sym } 
                    else
                      x_col_names
                    end
    # Hash containing the estimated fixed effects coefficiants (these estimates are 
    # conditional on the estimated covariance parameters).
    @fix_ef = Hash.new
    @fix_ef_names.each_with_index { |name, i| @fix_ef[name] = @model_data.beta[i] }
    
    # Array containing the names of the random effects terms
    @ran_ef_names = if z_col_names.nil? then 
                      (0...@model_data.b.shape[0]).map { |i| "z#{i}".to_sym }
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
  #   fixed effects and random effects of the model, with the response on 
  #   the left of a ~ operator and the terms, separated by + operators, 
  #   on the right hand side. Random effects specifications are in 
  #   parentheses () and contain a vertical bar |. Expressions for design 
  #   matrices are on the left of the vertical bar |, and grouping factors 
  #   are on the right. 
  # * +data+           - a Daru::DataFrame object, containing the response, fixed and random 
  #   effects, as well as the grouping variables
  # * +weights+        - optional Array of prior weights
  # * +offset+         - an optional vector of offset terms which are known 
  #   a priori; a nx1 NMatrix
  # * +reml+           - if true than the profiled REML criterion will be used as the objective
  #   function for the minimization; if false then the profiled deviance 
  #   will be used; defaults to true
  # * +start_point+    - an optional Array specifying the initial parameter estimates for the 
  #   minimization
  # * +epsilon+        - an optional  small number specifying the thresholds for the 
  #   convergence check of the optimization algorithm; see the respective 
  #   documentation for more detail
  # * +max_iterations+ - optional, the maximum number of iterations for the optimization 
  #   algorithm
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

    # to prevent side effects on the passed formula
    original_formula = formula
    formula = original_formula.clone

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
  #   interaction effect can be specified as Array of length two.
  #   An intercept term can be denoted as +:intercept+; and 
  #   +:no_intercept+ denotes the exclusion of an intercept term, even 
  #   if +:intercept+ is given.
  # * +random_effects+ - names of the random effects in +data+, given as an Array of Arrays;
  #   where the variables in each (inner) Array share a common grouping 
  #   structure, and the corresponding random effects are modeled as 
  #   correlated. An interaction effect can be specified as Array of 
  #   length two. An intercept term can be denoted as +:intercept+; and 
  #   +:no_intercept+ denotes the exclusion of an intercept term, even 
  #   if +:intercept+ is given.
  # * +grouping+       - an Array of the names of the variables in +data+, which determine the
  #   grouping structures for +random_effects+
  # * +data+           - a Daru::DataFrame object, containing the response, fixed and random 
  #   effects, as well as the grouping variables
  # * +weights+        - optional Array of prior weights
  # * +offset+         - an optional vector of offset terms which are known 
  #   a priori; a nx1 NMatrix
  # * +reml+           - if true than the profiled REML criterion will be used as the objective
  #   function for the minimization; if false then the profiled deviance 
  #   will be used; defaults to true
  # * +start_point+    - an optional Array specifying the initial parameter estimates for the 
  #   minimization
  # * +epsilon+        - an optional  small number specifying the thresholds for the 
  #   convergence check of the optimization algorithm; see the respective 
  #   documentation for more detail
  # * +max_iterations+ - optional, the maximum number of iterations for the optimization 
  #   algorithm
  # * +formula+        - (! Never used in a direct call of #from_daru) a String containing the 
  #   formula used to fit the model, if the model was fit by #from_formula
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
                                            random_effects: random_effects, grouping: grouping, 
                                            data: data}))

    # to prevent side effects on these parameters
    fixed_effects_copy = Marshal.load(Marshal.dump(fixed_effects))
    random_effects_copy = Marshal.load(Marshal.dump(random_effects))
    grouping_copy = Marshal.load(Marshal.dump(grouping))
    data_copy = Marshal.load(Marshal.dump(data)) 

    n = data_copy.size

    ################################################################
    # Adjust +data_copy+, +fixed_effects_copy+, +random_effects_copy+ and 
    # +grouping_copy+ for inclusion or exclusion of an intercept term, 
    # categorical variables, interaction effects and nested 
    # grouping factors
    ################################################################

    adjusted = MixedModels::adjust_lmm_from_daru_inputs(fixed_effects: fixed_effects_copy, 
                                                        random_effects: random_effects_copy, 
                                                        grouping: grouping_copy, data: data_copy)
    fixed_effects_copy  = adjusted[:fixed_effects]
    random_effects_copy = adjusted[:random_effects]
    grouping_copy       = adjusted[:grouping]
    data_copy           = adjusted[:data]

    ################################################################
    # Construct model matrices and vectors, covariance function,
    # and optimization parameters 
    ################################################################

    # construct the response vector
    y = NMatrix.new([n,1], data_copy[response].to_a, dtype: :float64)

    # construct the fixed effects design matrix
    x_frame     = data_copy[*fixed_effects_copy]
    x           = x_frame.to_nm
    x_col_names = fixed_effects_copy.clone # column names of the x matrix

    # construct the random effects model matrix and covariance function 
    num_groups = grouping_copy.length
    raise(ArgumentError, "Length of +random_effects+ (#{random_effects_copy.length}) " +
          "mismatches length of +grouping+ (#{num_groups})") unless random_effects_copy.length == num_groups
    ran_ef_raw_mat = Array.new
    ran_ef_grp     = Array.new
    num_ran_ef     = Array.new
    num_grp_levels = Array.new
    0.upto(num_groups-1) do |i|
      xi_frame = data_copy[*random_effects_copy[i]]
      ran_ef_raw_mat[i] = xi_frame.to_nm
      ran_ef_grp[i] = data_copy[grouping_copy[i]].to_a
      num_ran_ef[i] = ran_ef_raw_mat[i].shape[1]
      num_grp_levels[i] = ran_ef_grp[i].uniq.length
    end
    z_result_hash = MixedModels::mk_ran_ef_model_matrix(ran_ef_raw_mat, ran_ef_grp, random_effects_copy)
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
  #   returned value is X*beta or X*beta+Z*b, where beta are the fixed
  #   b the random effects coefficients; default is true 
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

  # Computes the Akaike Information Criterion (AIC) using the formula given in 2.4.1 in
  # Pinheiro & Bates (2000).
  #
  # === References
  #
  # * J C Pinheiro and D M Bates, "Mixed Effects Models in S and S-PLUS". Springer. 2000.
  #
  def aic
    num_param = @fix_ef.length + self.theta.length + 1
    aic = self.deviance + 2.0 * num_param
    return aic
  end

  # Computes the Bayesian Information Criterion (BIC) using the formula given in 2.4.1 in
  # Pinheiro & Bates (2000).
  #
  # === References
  #
  # * J C Pinheiro and D M Bates, "Mixed Effects Models in S and S-PLUS". Springer. 2000.
  #
  def bic
    num_param = @fix_ef.length + self.theta.length + 1
    bic = self.deviance + num_param * Math::log(@model_data.n)
    return bic
  end

  # Returns a Daru::DataFrame object containing in it's columns the fixed effect coefficient 
  # estimates, the standard deviations of the fixed effects coefficient estimates, the
  # corresponding z statistics (or z-score, or equivalently t statistics), and the corresponding 
  # Wald-Z p-values testing for each fixed effects term the null hypothesis that the true 
  # coefficient is equal to zero. The rows of the data frame correspond the fixed effects
  # coefficients and are named accordingly.
  # See also #fix_ef, #fix_ef_sd, #fix_ef_z, #fix_ef_p, #fix_ef_test, #likelihood_ratio_test.
  #
  # === Usage
  #
  #  > df = Daru::DataFrame.from_csv "spec/data/alien_species.csv"
  #  > mod = LMM.from_formula(formula: "Aggression ~ Age + Species + (Age | Location)", data: df)
  #  > puts mod.fix_ef_summary.inspect(24)
  # 
  #  #<Daru::DataFrame:69819740843180 @name = 6a13d7f0-d16a-4da9-a199-3a320a8ffc59 @size = 5>
  #                                               coef                       sd                  z_score            WaldZ_p_value 
  #                 intercept       1016.2867207696772        60.19727495932258       16.882603431075875                      0.0 
  #                       Age     -0.06531615343467667      0.08988486367253856      -0.7266646548258817       0.4674314106158888 
  #         Species_lvl_Human        -499.693695290209       0.2682523406941927      -1862.7747813759402                      0.0 
  #           Species_lvl_Ood       -899.5693213535769       0.2814470814004366      -3196.2289922406044                      0.0 
  #  Species_lvl_WeepingAngel      -199.58895804200762       0.2757835779525997       -723.7158917283754                      0.0
  #
  def fix_ef_summary
    coef_vec = Daru::Vector.new @fix_ef
    sd_vec = Daru::Vector.new self.fix_ef_sd
    z_vec = Daru::Vector.new self.fix_ef_z
    p_vec = Daru::Vector.new self.fix_ef_p(method: :wald)
    
    return Daru::DataFrame.new([coef_vec, sd_vec, z_vec, p_vec], 
                               order: [:coef, :sd, :z_score, :WaldZ_p_value])
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

  # Tests each fixed effects coefficient against the null hypothesis that
  # it is equal to zero, i.e. whether it is a significant predictor of the response.
  # Available methods are Wald Z test, likelihood ratio test via the Chi squared approximation,
  # and a bootstrap based likelihood ratio test (see also LMM#likelihood_ratio_test).
  # Both types of likelihood ratio tests are available only if the model was fit 
  # via LMM#from_formula or LMM#from_daru with +reml+ set to false.
  #
  # == Returns
  #
  # * a Hash containing the p-values for all fixed effects coefficients, 
  #   if +method+ is +:wald+
  # * a p-value corresponding to the fixed effect coefficient +variable+, 
  #   if +method+ is +:lrt+ or +:bootstrap+
  #
  # === Arguments
  #
  # * +method+ - determines the method used to compute the p-values;
  #   possibilities are: +:wald+ which denotes the Wald z test; +:lrt+ which performs a
  #   likelihood ratio test based on the Chi square distribution, as delineated in section 2.4.1 in 
  #   Pinheiro & Bates (2000); +:bootstrap+ performs a simulation based likelihood ratio test,
  #   as illustrated in 4.2.3 in Davison & Hinkley (1997);
  #   see also LMM#likelihood_ratio_test
  # * +variable+ - denotes the fixed effects coefficient to be tested; required if and only if 
  #   +method+ is +:lrt+ or +:bootstrap+; ignored if +method+ is +:wald+
  # * +nsim+ - (only relevant if method is +:bootstrap+) number of simulations for 
  #   the bootstrapping; default is 1000
  # * +parallel+ - (only relevant if method is +:bootstrap+) if true than the bootstrap
  #   is performed in parallel using all available CPUs; default is true.
  #   
  # === References
  #
  # * J C Pinheiro and D M Bates, "Mixed Effects Models in S and S-PLUS". Springer. 2000.
  # * A C Davison and D V Hinkley, "Bootstrap Methods and their Application". 
  #   Cambridge Series in Statistical and Probabilistic Mathematics. 1997.
  #
  def fix_ef_p(method: :wald, variable: nil, nsim: 1000, parallel: true)
    case method
    when :wald
      z = self.fix_ef_z
      p = Hash.new
      z.each_key { |k| p[k] = 2.0*(1.0 - Distribution::Normal.cdf(z[k].abs)) }
    when :lrt
      reduced_model = self.drop_fix_ef(variable) # this will also check if variable is valid
      p = LMM.likelihood_ratio_test(reduced_model, self, method: :chi2)
    when :bootstrap
      reduced_model = self.drop_fix_ef(variable) # this will also check if variable is valid
      p = LMM.likelihood_ratio_test(reduced_model, self, method: :bootstrap,
                                    nsim: nsim, parallel: parallel)
    else
      raise(NotImplementedError, "Method #{method} is currently not implemented")
    end

    return p
  end

  alias fix_ef_test fix_ef_p

  # Returns a Hash containing the confidence intervals of the fixed effects coefficients.
  #
  # === Arguments
  #
  # * +level+  - confidence level, a number between 0 and 1
  # * +method+ - determines the method used to compute the confidence intervals;
  #   possible values are +:wald+, +:bootstrap+ and +:all+; 
  #   +:wald+ approximates the confidence intervals based on the Wald z test statistic; 
  #   +:bootstrap+ constructs the confidence intervals from the bootstrap distribution of the fixed 
  #   effects terms, where several sub-types are available (see +boottype+ below);
  #   +:all+ designates the use of all available methods (including all bootstrap types), and returns a 
  #   Daru::DataFrame containing confidence bounds obtained from each method.
  #   Default is +:wald+.
  # * +boottype+ - (only relevant if method is +:bootstrap+) determines how bootstrap confidence
  #   intervals are computed:
  #   +:basic+ computes the basic bootstrap intervals according to (5.6) in Chapter 5 of Davison & Hinkley;
  #   +:normal+ computes confidence intervals based on the normal distribution using resampling estimates
  #   for bias correction and variance estimation, as in (5.5) in Chapter 5 of Davison & Hinkley;
  #   +:studentized+ computes studentized bootstrap confidence intervals, also known as 
  #   bootstrap-t, as given in (5.7) in Chapter 5 of Davison & Hinkley;
  #   +:percentile+ computes basic percentile bootstrap confidence intervals as in (5.18) in Davison & Hinkley.
  #   Default is +:studentized+.
  # * +nsim+   - (only relevant if method is +:bootstrap+) number of simulations for 
  #   the bootstrapping
  # * +parallel+ - (only relevant if method is +:bootstrap+) if true than the bootstrap resampling is performed
  #   in parallel using all available CPUs; default is true.
  #
  # === References
  #
  # * A C Davison and D V Hinkley, Bootstrap Methods and their Application. 
  #   Cambridge Series in Statistical and Probabilistic Mathematics. 1997.
  #
  def fix_ef_conf_int(level: 0.95, method: :wald, boottype: :studentized, nsim: 1000, parallel: true)
    alpha = 1.0 - level

    #####################################################
    # Define some auxialliary Proc objects
    #####################################################

    # Proc computing basic bootstrap confidence intervals
    bootstrap_basic = Proc.new do |bootstrap_sample|
      bootstrap_df = Daru::DataFrame.rows(bootstrap_sample, order: @fix_ef.keys)
      conf_int = Hash.new 
      @fix_ef.each_key do |key|
        z1 = bootstrap_df[key].percentile((alpha/2.0)*100.0)
        z2 = bootstrap_df[key].percentile((1.0 - alpha/2.0)*100.0)
        conf_int[key] = Array.new
        conf_int[key][0] = 2.0 * @fix_ef[key] - z2
        conf_int[key][1] = 2.0 * @fix_ef[key] - z1
      end
      conf_int
    end

    # Proc computing bootstrap normal confidence intervals
    bootstrap_normal = Proc.new do |bootstrap_sample|
      bootstrap_df = Daru::DataFrame.rows(bootstrap_sample, order: @fix_ef.keys)
      conf_int = Hash.new 
      @fix_ef.each_key do |key|
        sd = bootstrap_df[key].sd
        bias = bootstrap_df[key].mean - @fix_ef[key]
        z = Distribution::Normal.p_value(alpha/2.0).abs
        conf_int[key] = Array.new
        conf_int[key][0] = @fix_ef[key] - bias - z * sd
        conf_int[key][1] = @fix_ef[key] - bias + z * sd
      end
      conf_int
    end

    # Proc computing studentized bootstrap confidence intervals
    bootstrap_t = Proc.new do |bootstrap_sample|
      bootstrap_df = Daru::DataFrame.rows(bootstrap_sample, 
                                          order: bootstrap_sample[0].keys)
      conf_int = Hash.new 
      @fix_ef.each_key do |key|
        key_z = "#{key}_z".to_sym
        bootstrap_df[key_z] = (bootstrap_df[key] - @fix_ef[key]) / bootstrap_df["#{key}_sd".to_sym] 
        z1 = bootstrap_df[key_z].percentile((alpha/2.0)*100.0)
        z2 = bootstrap_df[key_z].percentile((1.0 - alpha/2.0)*100.0)
        conf_int[key] = Array.new
        conf_int[key][0] = @fix_ef[key] - z2 * self.fix_ef_sd[key] 
        conf_int[key][1] = @fix_ef[key] - z1 * self.fix_ef_sd[key] 
      end
      conf_int
    end
    
    # Proc computing bootstrap percentile confidence intervals
    bootstrap_percentile = Proc.new do |bootstrap_sample|
      bootstrap_df = Daru::DataFrame.rows(bootstrap_sample, order: @fix_ef.keys)
      conf_int = Hash.new 
      @fix_ef.each_key do |key|
        conf_int[key] = Array.new
        conf_int[key][0] = bootstrap_df[key].percentile((alpha/2.0)*100.0)
        conf_int[key][1] = bootstrap_df[key].percentile((1.0 - alpha/2.0)*100.0)
      end
      conf_int
    end

    # Proc to supply to #bootstrap as argument what_to_collect
    fix_ef_and_sd = Proc.new do |model| 
      result = model.fix_ef
      cov_mat = self.fix_ef_cov_mat
      model.fix_ef_names.each_with_index do |name, i| 
        result["#{name}_sd".to_sym] = Math::sqrt(cov_mat[i,i])
      end
      result
    end

    # Proc to compute Wald Z intervals
    wald = Proc.new do
      conf_int = Hash.new 
      z = Distribution::Normal.p_value(alpha/2.0).abs
      sd = self.fix_ef_sd
      @fix_ef.each_key do |key|
        conf_int[key] = Array.new
        conf_int[key][0] = @fix_ef[key] - z * sd[key]
        conf_int[key][1] = @fix_ef[key] + z * sd[key]
      end
      conf_int
    end

    ##########################################################
    # Compute the intervals specified by method and boottype
    ##########################################################
    
    case [method, boottype]
    when [:wald, boottype]
      conf_int = wald.call
    when [:bootstrap, :basic]
      bootstrap_sample = self.bootstrap(nsim: nsim, parallel: parallel)
      conf_int = bootstrap_basic.call(bootstrap_sample)
    when [:bootstrap, :normal]
      bootstrap_sample = self.bootstrap(nsim: nsim, parallel: parallel)
      conf_int = bootstrap_normal.call(bootstrap_sample)
    when [:bootstrap, :studentized]
      bootstrap_sample = self.bootstrap(nsim: nsim, parallel: parallel, 
                                        what_to_collect: fix_ef_and_sd)
      conf_int = bootstrap_t.call(bootstrap_sample)
    when [:bootstrap, :percentile]
      bootstrap_sample = self.bootstrap(nsim: nsim, parallel: parallel)
      conf_int = bootstrap_percentile.call(bootstrap_sample)
    when [:all, boottype]
      conf_int_wald = wald.call
      bootstrap_sample = self.bootstrap(nsim: nsim, parallel: parallel, 
                                        what_to_collect: fix_ef_and_sd)
      conf_int_basic = bootstrap_basic.call(bootstrap_sample)
      conf_int_normal = bootstrap_normal.call(bootstrap_sample)
      conf_int_studentized = bootstrap_t.call(bootstrap_sample)
      conf_int_percentile = bootstrap_percentile.call(bootstrap_sample)
      conf_int = Daru::DataFrame.new([conf_int_wald, conf_int_basic, conf_int_normal,
                                      conf_int_studentized, conf_int_percentile],
                                     index: [:wald_z, :boot_basic, :boot_norm, 
                                             :boot_t, :boot_perc])
    else
      raise(NotImplementedError, "Method #{method} is currently not implemented")
    end

    return conf_int
  end

  # A convenience method, which summarizes the estimates of the variances and covariances of 
  # the random effects.
  # If the model was fit via #from_formula or #from_daru, then a Daru::DatFrame with rows 
  # and columns named according to the random effects, containing all random effects variances
  # and covariances is returned.
  # If the model was fit from raw model matrices via #initialize, then the covariance matrix
  # of the random effects vector is returned as a NMatrix object.
  # This is mainly an auxilliary function for #ran_ef_summary.
  #
  # See also LMM#sigma_mat.
  #
  def ran_ef_cov

    ########################################################################
    # when the model was fit from raw matrices with a custom argument thfun
    ########################################################################

    return @sigma_mat if @from_daru_args.nil?

    ################################################
    # when the model was fit from a Daru::DataFrame
    ################################################

    data = @from_daru_args[:data]
    
    # get the random effects terms and grouping structures used in the model
    grp = Marshal.load(Marshal.dump(@from_daru_args[:grouping]))
    re  = Marshal.load(Marshal.dump(@from_daru_args[:random_effects]))
    num_grp_levels = Array.new
    # take care of nested effects (see MixedModels.adjust_lmm_from_daru_inputs)
    # and determine the number of distinct elements in each grouping variable
    grp.each_index do |ind| 
      if grp[ind].is_a? Array then
        var1, var2 = data[grp[ind][0]].to_a, data[grp[ind][1]].to_a
        num_grp_levels[ind] = var1.zip(var2).map { |p| p.join("_and_") }.uniq.size
        grp[ind] = "#{grp[ind][0]}_and_#{grp[ind][1]}"
      else
        num_grp_levels[ind] = data[grp[ind]].uniq.size
      end
    end
    # take care of :no_intercept as random effect (see MixedModels.adjust_lmm_from_daru_inputs)
    re.each do |ef|
      if ef.include? :no_intercept then
        ef.delete(:intercept)
        ef.delete(:no_intercept)
      end
    end

    #FIXME: this:
    re.each do |ef_array|
      if ef_array.any? { |ef| ef.is_a? Array }  then
        raise(NotImplementedError, "LMM#ran_ef_cov does not work correctly in the presence of random interaction effects. Please use LMM#sigma_mat in those cases.")
      else
        ef_array.each do |ef|
          unless ef == :intercept then
            unless data[ef].type == :numeric then
              raise(NotImplementedError, "LMM#ran_ef_cov does not work correctly in the presence of categorical variables as random effects. Please use LMM#sigma_mat in those cases.")
            end
          end
        end
      end
    end

    # get names for the rows and columns of the returned data frame
    get_name = Proc.new do |i,j|
      if re[i][j] == :intercept then
        "#{grp[i]}".to_sym
      else
        "#{grp[i]}_#{re[i][j]}".to_sym
      end
    end
    names = []
    grp.each_index do |i|
      re[i].each_index do |j|
        names << get_name.call(i,j)
      end
    end

    # generate the data frame to be returned, yet filled with nil's 
    nils = Array.new(names.length) { Array.new(names.length) {nil} }
    varcov = Daru::DataFrame.new(nils, order: names, index: names)

    # fill the data frame
    block_position = 0 # position of the analyzed block in the block-diagonal sigma_mat
    grp.each_index do |i|
      re[i].each_index do |j|
        name1 = get_name.call(i,j)
        re[i].each_index do |k|
          name2 = get_name.call(i,k)
          varcov[name1][name2] = @sigma_mat[block_position + j, block_position + k]
        end
      end
      block_position += num_grp_levels[i] * re[i].length
    end

    return varcov
  end

  # A convenience method, which summarizes the estimates of the standard deviations and 
  # correlations of the random effects.
  # If the model was fit via #from_formula or #from_daru, then a Daru::DatFrame with rows 
  # and columns named according to the random effects, containing all random effects variances
  # and covariances is returned.
  # If the model was fit from raw model matrices via #initialize, then the correlation matrix
  # of the random effects vector is returned as a NMatrix object.
  #
  # See also LMM#sigma_mat.
  #
  def ran_ef_summary

    varcov = self.ran_ef_cov

    if @from_daru_args then
      # when the model was fit from a Daru::DataFrame
      # turn data frame of covariances into data frame of correlations
      names = varcov.vectors.relation_hash.keys
      names.each { |x| varcov[x][x] = Math::sqrt(varcov[x][x]) }
      names.each do |x|
        names.each do |y|
          # do for x != y if varcov is not nil
          varcov[x][y] = varcov[x][y] / (varcov[x][x] * varcov[y][y]) if x != y && varcov[x][y]
        end
      end
    else
      # when the model was fit from raw matrices with a custom argument thfun
      # turn covariance matrix varcov into a correlation matrix
      q = varcov.shape[0]
      q.times { |i| varcov[i,i] = Math::sqrt(varcov[i,i]) }
      q.times do |i|
        q.times do |j|
          varcov[i,j] = varcov[i,j] / (varcov[i,i] * varcov[j,j]) if i != j
        end
      end
    end

    return varcov 
  end
      
  alias ran_ef_cor ran_ef_summary
  alias ran_ef_corr ran_ef_summary

  # Significance test for random effects variables.
  # Available methods are a likelihood ratio test via the Chi squared approximation,
  # and a bootstrap based likelihood ratio test. Both types of likelihood ratio tests are
  # available only if the model was fit via LMM#from_formula or LMM#from_daru 
  # with +reml+ set to false (see also LMM#likelihood_ratio_test).
  # Returned is a p-value corresponding to the random effect term +variable+.
  #
  # === Arguments
  #
  # * +method+ - determines the method used to compute the p-value;
  #   possibilities are: +:lrt+ (default) which performs a likelihood ratio test based on the Chi square 
  #   distribution, as delineated in section 2.4.1 in Pinheiro & Bates (2000); 
  #   +:bootstrap+ performs a simulation based likelihood ratio test, as illustrated in 4.2.3 
  #   in Davison & Hinkley (1997); see also LMM#likelihood_ratio_test
  # * +variable+ - denotes the random effects coefficient to be tested
  # * +grouping+ - the grouping variable corresponding to +variable+
  # * +nsim+ - (only relevant if method is +:bootstrap+) number of simulations for 
  #   the bootstrapping; default is 1000
  # * +parallel+ - (only relevant if method is +:bootstrap+) if true than the bootstrap
  #   is performed in parallel using all available CPUs; default is true.
  #   
  # === References
  #
  # * J C Pinheiro and D M Bates, "Mixed Effects Models in S and S-PLUS". Springer. 2000.
  # * A C Davison and D V Hinkley, "Bootstrap Methods and their Application". 
  #   Cambridge Series in Statistical and Probabilistic Mathematics. 1997.
  #
  # === Usage
  #
  #   df = Daru::DataFrame.from_csv "alien_species.csv"
  #   model = LMM.from_formula(formula: "Aggression ~ Age + Species + (Age | Location)", 
  #                            reml: false, data: df)
  #   p = model.ran_ef_p(variable: :Age, grouping: :Location) 
  #            
  def ran_ef_p(method: :lrt, variable:, grouping:, nsim: 1000, parallel: true)
    # this will also check if variable and grouping are valid
    reduced_model = self.drop_ran_ef(variable, grouping) 

    p = case method
        when :lrt
          LMM.likelihood_ratio_test(reduced_model, self, method: :chi2)
        when :bootstrap
          LMM.likelihood_ratio_test(reduced_model, self, method: :bootstrap,
                                    nsim: nsim, parallel: parallel)
        else
          raise(NotImplementedError, "Method #{method} is currently not implemented")
        end

    return p
  end

  alias ran_ef_test ran_ef_p

  # Conditional variance-covariance matrix of the random effects *estimates*, based on
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
  # equal to one
  #
  def sigma
    Math::sqrt(@sigma2)
  end

  # Refit the same model on new data. That is, apart from the model matrices X, Z and the response y,
  # the resulting model is fit using exactly the same parameters as were used to fit +self. 
  # New data can be supplied either in form of a +Daru::DataFrame+ or in form of model matrices.
  #
  # === Arguments
  #
  # * +newdata+ - a Daru::DataFrame object containing the data.
  # * +x+       - fixed effects model matrix, a dense NMatrix object
  # * +y+       - response vector as a nx1 dense NMatrix
  # * +zt+      - random effects model matrix, a dense NMatrix object
  #
  def refit(newdata: nil, x: nil, y: nil, zt: nil)
    raise(ArgumentError, "EITHER pass newdata OR x, y and zt") unless newdata || (x && y && zt)

    # refit from Daru::DataFrame
    if newdata then
      raise(ArgumentError, "newdata and x or y or zt cannot be passed simultaneously") if newdata && (x || y || zt)
      raise(ArgumentError, "LMM#refit does not work with a Daru::DataFrame," +
            "if the model was not fit using a Daru::DataFrame") if @from_daru_args.nil?
      return LMM.from_daru(response: @from_daru_args[:response], 
                           fixed_effects: @from_daru_args[:fixed_effects], 
                           random_effects: @from_daru_args[:random_effects], 
                           grouping: @from_daru_args[:grouping], 
                           data: newdata,
                           weights: @model_data.weights, 
                           offset: @model_data.offset, 
                           reml: @reml, 
                           start_point: @optimization_result.start_point,
                           epsilon: @optimization_result.epsilon, 
                           max_iterations: @optimization_result.max_iterations, 
                           formula: @formula)
    end

    # refit from raw model matrices
    raise(ArgumentError, "x and y and zt need to be passed together") unless x && y && zt
    return LMM.new(x: x, y: y, zt: zt, 
                   x_col_names: @fix_ef_names, 
                   z_col_names: @ran_ef_names, 
                   weights: @model_data.weights, 
                   offset: @model_data.offset, 
                   reml: @reml, 
                   start_point: @optimization_result.start_point,
                   lower_bound: @optimization_result.lower_bound,
                   upper_bound: @optimization_result.upper_bound,
                   epsilon: @optimization_result.epsilon, 
                   max_iterations: @optimization_result.max_iterations, 
                   formula: @formula,
                   &@model_data.thfun)
  end

  # Assuming that the estimated fixed effects and covariances are the true model parameters, 
  # simulate a new response vector for the data that was used to fit the model.
  #
  # === Arguments
  #
  # * +type+ - determines how exactly the new response is to be simulated; currently the only
  #   possible option is +:parameteric+, which defines the new response as 
  #   y = X*beta + Z*new_b + new_epsilon,
  #   where new_b is simulated from the multivariate normal distribution with the covariance
  #   matrix +@sigma_mat+, and new_epsilon are i.i.d. random errors with variance +@sigma2+
  #   (for more detail see under references).
  #
  # === References
  #
  # * Joseph E., Cavanaugh ; Junfeng, Shang. (2008) An assumption for the development of 
  #   bootstrap variants of the Akaike information criterion in mixed models. 
  #   In: Statistics & Probability Letters. Accessible at http://personal.bgsu.edu/~jshang/AICb_assumption.pdf.
  #
  def simulate_new_response(type: :parametric)
    normal_rng = Distribution::Normal.rng
    n = @model_data.n
    q = @model_data.q

    if type == :parametric then
      # generate the random effects vector from its estimated multivariate normal distribution
      std_norm_vals = Array.new(q) { normal_rng.call }
      std_norm_vec = NMatrix.new([q, 1], std_norm_vals, dtype: :float64)
      cholesky_factor = @sigma_mat.factorize_cholesky[1]
      new_ran_ef = cholesky_factor.dot(std_norm_vec)

      # generate new random residuals 
      sigma = self.sigma
      new_epsilon_a = n.times.map { |i| (sigma / @model_data.sqrtw[i,i]) * normal_rng.call }
      new_epsilon = NMatrix.new([n, 1], new_epsilon_a, dtype: :float64)

      # generate new response vector
      new_response =  (@model_data.x.dot(@model_data.beta) + @model_data.zt.transpose.dot(new_ran_ef) + new_epsilon)
      
      # add the offset
      new_response += @model_data.offset 
    else
      raise(ArgumentError, "Not a valid simulation type")
    end

    return new_response.to_flat_a
  end

  # Perform bootstrapping for linear mixed models to generate bootstrap samples of the parameters.
  #
  # === Arguments
  #
  # * +nsim+ - number of simulations
  # * +what_to_collect+ - (optional) a Proc taking a LMM object as input, and generating a 
  #   statistic of interest; if unspecified, then an Array containing the estimates of the fixed 
  #   effects terms for each simulated model is generated 
  # * +how_to_simulate+ - (optional) a Proc taking a LMM object as input, and returning a new 
  #   simulated response as an Array; if unspecified, then LMM#simulate_new_response is used instead
  # * +type+ - (optional) the argument +type+ for LMM#simulate_new_response; only used if
  #   +how_to_simulate+ is unspecified
  # * +parallel+ - if true than the resampling is done in parallel using all available CPUs; 
  #   default is true
  #
  # === References
  #
  # * Joseph E., Cavanaugh ; Junfeng, Shang. (2008) An assumption for the development of bootstrap 
  #   variants of the Akaike information criterion in mixed models. In: Statistics & Probability Letters. 
  #   Accessible at http://personal.bgsu.edu/~jshang/AICb_assumption.pdf.
  #
  def bootstrap(nsim:, how_to_simulate: nil, type: :parametric, what_to_collect: nil, parallel: true)
    require 'parallel'
    num_proc = (parallel ? Parallel.processor_count : 0)

    results = Parallel.map((0...nsim).to_a, :in_processes => num_proc) do |i|
      new_y = if how_to_simulate then
                how_to_simulate.call(self)
              else
                self.simulate_new_response(type: type)
              end

      new_model = self.refit(x: @model_data.x, 
                             y: NMatrix.new([@model_data.n, 1], new_y, dtype: :float64), 
                             zt: @model_data.zt)

      if what_to_collect then
        what_to_collect.call(new_model)
      else
        new_model.fix_ef
      end
    end

    return results
  end

  # Drop one fixed effect predictor from the model; i.e. refit the model without one predictor variable.
  # Works only if the model was fit via #from_daru or #from_formula.
  #
  # === Arguments
  #
  # * +variable+ - name of the fixed effect to be dropped. An interaction effect can be specified as 
  #   an Array of length two. An intercept term can be denoted as +:intercept+.
  #
  def drop_fix_ef(variable)
    raise(NotImplementedError, "LMM#drop_fix_ef does not work if the model was not fit using a Daru::DataFrame") if @from_daru_args.nil?
    raise(ArgumentError, "variable is not one of the fixed effects of the linear mixed model") unless @from_daru_args[:fixed_effects].include? variable

    fe = Marshal.load(Marshal.dump(@from_daru_args[:fixed_effects]))
    variable_ind = fe.find_index variable
    fe.delete_at variable_ind

    return LMM.from_daru(response: @from_daru_args[:response], 
                         fixed_effects: fe,
                         random_effects: @from_daru_args[:random_effects], 
                         grouping: @from_daru_args[:grouping], 
                         data: @from_daru_args[:data],
                         weights: @model_data.weights, 
                         offset: @model_data.offset, 
                         reml: @reml, 
                         start_point: @optimization_result.start_point,
                         epsilon: @optimization_result.epsilon, 
                         max_iterations: @optimization_result.max_iterations, 
                         formula: @formula)
  end

  # Drop one random effects term from the model; i.e. refit the model without one random effect variable.
  # Works only if the model was fit via #from_daru or #from_formula.
  #
  # === Arguments
  #
  # * +variable+ - name of the random effect to be dropped. An interaction effect can be specified as 
  #   an Array of length two. An intercept term can be denoted as +:intercept+.
  # * +grouping+ - the grouping variable corresponding to +variable+
  # * +start_point+ - (optional) since the same starting point can not be used in the optimization algorithm 
  #   to fit a model with fewer random effects, a new starting point can be provided with this argument
  #
  # === Usage
  #
  #   df = Daru::DataFrame.from_csv "alien_species.csv"
  #   model = LMM.from_formula(formula: "Aggression ~ Age + Species + (Age | Location)", 
  #                            reml: false, data: df)
  #   reduced_model = model.drop_ran_ef(:Age, :Location)
  #            
  def drop_ran_ef(variable, grouping, start_point: nil)
    raise(NotImplementedError, "LMM#drop_ran_ef does not work if the model was not fit using a Daru::DataFrame") if @from_daru_args.nil?
    raise(ArgumentError, "grouping is not one of grouping variables in the linear mixed model") unless @from_daru_args[:grouping].include? grouping

    # get the indices of groups of random effects with grouping structure 
    # determined by the variable +grouping+
    possible_ind = @from_daru_args[:grouping].each_with_index.select { |var, i| var == grouping }.map { |pair| pair[1] }
    # get the index of +variable+ and the index of the corresponding group
    variable_ind= nil
    group_ind = nil
    possible_ind.each do |i|
      unless variable_ind then
        variable_ind = @from_daru_args[:random_effects][i].find_index variable 
        group_ind = i
      end
    end
    raise(ArgumentError, "variable does not match grouping") unless variable_ind

    # delete the variable from the Array of random effects names
    re = Marshal.load(Marshal.dump(@from_daru_args[:random_effects]))
    re[group_ind].delete_at variable_ind
    # delete group of random effects if no more variables fall under it;
    # also delete group of random effects if nothing but :intercept and :no_intercept fall under it
    gr = Marshal.load(Marshal.dump(@from_daru_args[:grouping]))
    if (re[group_ind].empty? || re[group_ind].uniq == [:no_intercept] ||
        re[group_ind].uniq == [:intercept, :no_intercept] || 
        re[group_ind].uniq == [:no_intercept, :intercept]) then
      gr.delete_at group_ind
      re.delete_at group_ind
    end

    return LMM.from_daru(response: @from_daru_args[:response], 
                         fixed_effects: @from_daru_args[:fixed_effects], 
                         random_effects: re,
                         grouping: gr,
                         data: @from_daru_args[:data],
                         weights: @model_data.weights, 
                         offset: @model_data.offset, 
                         reml: @reml, 
                         start_point: start_point,
                         epsilon: @optimization_result.epsilon, 
                         max_iterations: @optimization_result.max_iterations, 
                         formula: @formula)
  end

  # Computes the likelihood ratio statistic of two linear mixed models as
  # 2 * log(L2 / L1), where L1 and L2 denote the likelihood of +model1+ and +model2+ respectively. 
  # 
  # == Arguments
  #
  # * +model1+ - a LMM object
  # * +model2+ - a LMM object
  #
  # === References
  # 
  # * J C Pinheiro and D M Bates, "Mixed Effects Models in S and S-PLUS", Springer, 2000.
  #
  def LMM.likelihood_ratio(model1, model2)
    # compute the likelihood ratio as in 2.4.1 in Pinheiro & Bates (2000)
    model1.deviance - model2.deviance
  end

  # Performs a likelihood ratio test for two nested models. Nested means that all predictors
  # used in +model1+ must also be predictors in +model2+ (i.e. +model1+ is a reduced version of +model2+). 
  # The null hypothesis is that the restricted model (+model1+) is adequate. 
  # This method works only if both models were fit using the deviance (as opposed to REML criterion) as 
  # the objective function for the minimization (i.e. fit with reml: false).
  # Returned is the p-value of the test.
  #
  # == Arguments
  #
  # * +model1+ - a restricted model, nested with respect to +model2+, a LMM object
  # * +model2+ - a more general model than +model1+, a LMM object
  # * +method+ - the method used to perform the test; possibilities are:
  #   +:chi2+ approximates distribution of the likelihood ratio test statistic with a Chi squared distribution
  #   as delineated in 2.4.1 in Pinheiro & Bates (2000);
  #   +:bootstrap+ simulates a sample of +nsim+ LRT statistics under the null hypothesis, and estimates
  #   the p-value as the proportion of LRT statistics greater than or equal to the oberved value, as delineated
  #   in 4.2.3 in Davison & Hinkley (1997)
  # * +nsim+   - (only relevant if method is +:bootstrap+) number of simulations for 
  #   the bootstrapping; default is 1000
  # * +parallel+ - (only relevant if method is +:bootstrap+) if true than the bootstrap
  #   is performed in parallel using all available CPUs; default is true.
  #
  # === References
  # 
  # * J C Pinheiro and D M Bates, "Mixed Effects Models in S and S-PLUS". Springer. 2000.
  # * A C Davison and D V Hinkley, "Bootstrap Methods and their Application". 
  #   Cambridge Series in Statistical and Probabilistic Mathematics. 1997.
  #
  def LMM.likelihood_ratio_test(model1, model2, method: :chi2, nsim: 1000, parallel: true)
    raise(NotImplementedError, "does not work if linear mixed model was not " +
          "fit using a Daru::DataFrame") if model1.from_daru_args.nil? || model2.from_daru_args.nil?
    raise(ArgumentError, "models were not fit to the same data") unless model1.from_daru_args[:data] == model2.from_daru_args[:data]
    raise(ArgumentError, "both model should be based on the deviance function " +
          "instead of the REML criterion (i.e. reml: false)") if model1.reml || model2.reml

    ##############################
    # check if models are nested
    ##############################
    
    # check for fixed effects that model1 has but model2 does not
    fe = model1.from_daru_args[:fixed_effects] - model2.from_daru_args[:fixed_effects]
    raise(ArgumentError, "fixed effects of model1 must be a subset of those of model2") unless fe.empty?
    # check for random effects that model1 has but model2 does not
    len = model1.from_daru_args[:random_effects].length
    if len == model2.from_daru_args[:random_effects].length then
      # one group of random effects of model1 must have fewer variables in this case
      len.times do |i|
        re = model1.from_daru_args[:random_effects][i] - model2.from_daru_args[:random_effects][i]
        raise(ArgumentError, "random effects of model1 must be a subset of those of model2") unless re.empty?
      end
    else
      # model1 must have fewer groups of random effects in this case
      re = model1.from_daru_args[:random_effects] - model2.from_daru_args[:random_effects]
      raise(ArgumentError, "random effects of model1 must be a subset of those of model2") unless re.empty?
    end
    # check for grouping variables that model1 has but model2 does not
    gr = model1.from_daru_args[:grouping] - model2.from_daru_args[:grouping]
    raise(ArgumentError, "grouping variables of model1 must be a subset of those of model2") unless gr.empty?

    ##############################
    # Compute the test statistic
    ##############################

    likelihood_ratio = LMM.likelihood_ratio(model1, model2)

    ##############################
    # Perform the test
    ##############################

    case method
    when :chi2
      num_param_model1 = model1.fix_ef.length + model1.theta.length
      num_param_model2 = model2.fix_ef.length + model2.theta.length
      df = num_param_model2 - num_param_model1
      p_value = Distribution::ChiSquare.q_chi2(df, likelihood_ratio)
    when :bootstrap
      require 'parallel'
      num_proc = (parallel ? Parallel.processor_count : 0)

      bootstrap_sample = Parallel.map((0...nsim).to_a, :in_processes => num_proc) do |i|
        # simulate data according to the null model 
        new_y = NMatrix.new([model1.model_data.n, 1], 
                            model1.simulate_new_response(type: :parametric), 
                            dtype: :float64)
        # refit both models to new data
        new_model1 = model1.refit(x: model1.model_data.x, y: new_y, zt: model1.model_data.zt)
        new_model2 = model2.refit(x: model2.model_data.x, y: new_y, zt: model2.model_data.zt)
        # compute LRT statistic
        LMM.likelihood_ratio(new_model1, new_model2)
      end

      num_greater_or_equal = bootstrap_sample.each.select { |lr| lr >= likelihood_ratio }.length 
      p_value = (num_greater_or_equal + 1.0) / (nsim + 1.0) 
    else
      raise(ArgumentError, "#{method} is not an available method")
    end

    return p_value
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
  #   will be evaluated
  # * +x+           - fixed effects model matrix, a NMatrix object
  # * +z+           - random effects model matrix, a NMatrix object
  # * +with_ran_ef+ - indicator whether the random effects should be considered in the
  #   predictions; i.e. whether the predictions are computed as x*beta
  #   or as x*beta+z*b; default is true
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
  #   will be evaluated
  # * +x+       - fixed effects model matrix, a NMatrix object
  # * +z+       - random effects model matrix, a NMatrix object
  # * +level+   - confidence level, a number between 0 and 1
  # * +type+    - +:confidence+ or +:prediction+ for confidence and prediction intervals
  #   respectively; see above for explanation of their difference
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
