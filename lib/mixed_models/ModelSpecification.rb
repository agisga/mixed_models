# Copyright (c) 2015 Alexej Gossmann 
#
# Methods for internal use.

module MixedModels 

  # Generate the random effects model matrix Z in the linear mixed effects model 
  # y = X*beta + Z*b + epsilon,
  # from the matrices +x[i]+, which contain the the covariates for random effects
  # with a common grouping structure given as +grp[i]+.
  # Optionally, an Array of column names for the matrix Z is generated as well.
  #
  # === Arguments
  #
  # * +x+     - Array of NMatrix objects. Each matrix +x[i]+ contains the covariates
  #   for a group of random effects with a common grouping structure.
  #   Typically, a given +x[i]+ contains correlated random effects, and the
  #   random effects from different matrices +x[i]+ are modeled as uncorrelated.  
  # * +grp+   - Array of Arrays. Each +grp[i]+ has length equal to the number of rows in +x[i]+,
  #   and specifies which observations in +x[i]+ correspond to which group.
  # * +names+ - (Optional) Array of Arrays, where the i'th Array contains the column names
  #   of the +x[i]+ matrix, i.e. the names of the corresponding random effects terms.
  #
  # === Returns
  #
  # A Hash containing the random effects model matrix Z under the key +:z+, and if the argument 
  # +names+ was provided then also an Array of column names under key +:names+. If no argument
  # was provided for +names+, then the returned Hash contains +nil+ under key +:names+. 
  #
  # === References
  # 
  # * Douglas Bates, Martin Maechler, Ben Bolker, Steve Walker, 
  #   "Fitting Linear Mixed - Effects Models using lme4". arXiv:1406.5823v1 [stat.CO]. 2014.
  #
  # === Usage
  #
  #   grp = Array[["grp1", "grp1", 2, 2, "grp3", "grp3"]]
  #   x1  = NMatrix.new([6,2], [1,-1,1,1,1,-1,1,1,1,-1,1,1], dtype: :float64)
  #   x   = Array[x1]
  #   z   = MixedModels::mk_ran_ef_model_matrix(x, grp)[:z]
  #           # => 
  #             [
  #               [1.0, -1.0, 0.0,  0.0, 0.0,  0.0]
  #               [1.0,  1.0, 0.0,  0.0, 0.0,  0.0]
  #               [0.0,  0.0, 1.0, -1.0, 0.0,  0.0]
  #               [0.0,  0.0, 1.0,  1.0, 0.0,  0.0]
  #               [0.0,  0.0, 0.0,  0.0, 1.0, -1.0]
  #               [0.0,  0.0, 0.0,  0.0, 1.0,  1.0]
  #             ]
  #
  def MixedModels.mk_ran_ef_model_matrix(x, grp, names=nil)
    num_x = x.length  # number of raw random effects matrices
    raise(ArgumentError, "Number of X matrices different than the number of grouping structures") unless num_x == grp.length
    unless names.nil? || num_x == names.length
      raise(ArgumentError, "Number of X matrices different than the number of Arrays of column names")
    end
    n = x[0].shape[0] # number of observations in each matrix x[i]

    z           = Array.new # Array of matrices where z[i] correponds to x[i] and grp[i]
    z_col_names = Array.new # Array whose i'th entry is an Array of column names of z[i]
    z_ncol      = 0 # Number of columns in the ran ef model matrix
    (0...num_x).each do |i|
      # sort the levels if possible
      begin
        grp_levels = grp[i].uniq.sort
      rescue
        grp_levels = grp[i].uniq
      end

      m = grp_levels.length 
      # generate a 0-1-valued matrix specifying the group memberships for each subject
      grp_mat = NMatrix.zeros([n,m], dtype: :float64)
      (0...m).each do |j|
        (0...n).each do |k|
          grp_mat[k,j] = 1.0 if grp[i][k] == grp_levels[j]
        end
      end

      # the random effects model matrix for the i'th grouping structure
      z[i] = grp_mat.khatri_rao_rows x[i]
      z_ncol += z[i].shape[1] 

      # create the column names for z[i]
      # Specific names are produced from Numeric, String or Symbol only, otherwise
      # codes based on indices are used. The names are saved as Symbols for consistency,
      # as the names of the fixed effect terms are Symbols as well. 
      is_NSS = Proc.new { |ind| ind.is_a?(Numeric) || ind.is_a?(String) || ind.is_a?(Symbol) }
      z_col_names[i] = Array.new
      if names then 
        grp_levels.each_with_index do |lvl, j|
          names[i].each_with_index do |term, k|
            case [is_NSS.call(term), is_NSS.call(lvl)]
            when [true, true]
              z_col_names[i].push "#{term}_#{lvl}".to_sym
            when [true, false]
              z_col_names[i].push "#{term}_grp#{i}_#{j}".to_sym
            when [false, true]
              z_col_names[i].push "x#{i}_#{k}_#{lvl}".to_sym
            else
              z_col_names[i].push "x#{i}_#{k}_grp#{i}_#{j}".to_sym
            end
          end
        end
      end
    end

    # concatenate the matrices z[i]
    z_model_mat = NMatrix.new([n, z_ncol], dtype: :float64)
    start_index = 0
    end_index   = 0
    z.each do |zi|
      end_index += zi.shape[1]
      z_model_mat[0...n, start_index...end_index] = zi
      start_index += zi.shape[1]
    end
    # concatenate the Arrays z_col_names[i]
    z_model_mat_names = (z_col_names.empty? ? nil : z_col_names.flatten)
    
    return {z: z_model_mat, names: z_model_mat_names}
  end

  # Generate a Proc object which parametrizes the transpose of the random effects covariance 
  # Cholesky factor Lambda as a function of theta. Lambda is defined as 
  # Lambda * Lambda^T = sigma^2*Sigma, where b ~ N(0,Sigma) is the distribution of the random 
  # effects vector, and the scaling factor sigma^2 comes from 
  # (y|b=b_0) ~ N(X*beta+Z*b_0, sigma^2*I).
  # Lambda^T is a upper triangular matrix of block-diagonal shape. The first +num_ran_ef.sum+ 
  # elements of theta determine the diagonal of Lambda^T, and the remaining entries of theta 
  # specify the off-diagonal entries of Lambda^T.
  #
  # === Arguments
  #
  # * +num_ran_ef+     - Array, where +num_ran_ef[i]+ is the number of random effects terms 
  #   associated with the i'th grouping structure. 
  # * +num_grp_levels+ - Array, where +num_grp_levels[i]+ is the number of levels of the i'th 
  #   grouping structure.
  #
  # === Usage
  #
  #   mapping = MixedModels::mk_ran_ef_cov_fun([2], [3])
  #   mapping.call([1,2,3]) # => [ [1.0, 3.0, 0.0, 0.0, 0.0, 0.0,
  #                                 0.0, 2.0, 0.0, 0.0, 0.0, 0.0,
  #                                 0.0, 0.0, 1.0, 3.0, 0.0, 0.0,
  #                                 0.0, 0.0, 0.0, 2.0, 0.0, 0.0,
  #                                 0.0, 0.0, 0.0, 0.0, 1.0, 3.0,
  #                                 0.0, 0.0, 0.0, 0.0, 0.0, 2.0] ]
  #
  def MixedModels.mk_ran_ef_cov_fun(num_ran_ef, num_grp_levels)
    raise(ArgumentError, "Supplied number of random effects does not match the supplied number of grouping structures") unless num_ran_ef.length == num_grp_levels.length
    cov_fun = Proc.new do |theta|
      lambdat_array = Array.new # Array of component matrices of the block-diagonal lambdat
      # the first num_ran_ef.sum elements of theta parametrize the diagonal of 
      # the covariance matrix
      th_count_diag = 0 
      # the remaining theta parametrize the off-diagonal entries of the covariance matrix
      th_count = num_ran_ef.sum 
      num_grp_levels.each_index do |i|
        k = num_ran_ef[i]
        m = num_grp_levels[i]
        lambdat_component = NMatrix.diagonal(theta[th_count_diag...(th_count_diag+k)], 
                                             dtype: :float64)
        th_count_diag += k
        (0...(k-1)).each do |j|
          ((j+1)...k).each do |l|
            lambdat_component[j,l] = theta[th_count]
            th_count += 1
          end
        end
        lambdat_array.concat(Array.new(m) {lambdat_component})
      end
      lambdat = NMatrix.block_diagonal(*lambdat_array, dtype: :float64)
    end
  end

  # For internal use in LMM#from_daru and LMM#predict (and maybe other LMM methods).
  # Adjusts +data+, +fixed_effects+, +random_effects+ and +grouping+ for the inclusion 
  # or exclusion of an intercept term, categorical variables, interaction effects and 
  # nested grouping factors. The categorical vectors in the data frame are replaced with 
  # sets of 0-1-valued indicator vectors. New vectors are added to the data frame for 
  # pair-wise interaction effects and for pair-wise nestings. The names of the fixed 
  # and random effects as well as grouping factors are adjusted accordingly.
  # Returned is a Hash containing the updated +data+, +fixed_effects+, +random_effects+ 
  # and +grouping+
  #
  # === Arguments
  # 
  # * +fixed_effects+  - Array of names of the fixed effects, see LMM#from_daru for details
  # * +random_effects+ - Array of Arrays of names of random effects, see LMM#from_daru for details
  # * +grouping+       - Array of names which determine the grouping structure underlying the
  #   random effects, see LMM#from_daru for details 
  # * +data+           - Daru::DataFrame object, containing the response, fixed and random 
  #   effects, as well as the grouping variables
  #
  def MixedModels.adjust_lmm_from_daru_inputs(fixed_effects:, random_effects:, grouping:, data:)
    n = data.size

    ##############################################
    # Does the model include intercept terms?
    ##############################################
    
    [fixed_effects, *random_effects].each do |ef|
      if ef.include? :no_intercept then
        ef.delete(:intercept)
        ef.delete(:no_intercept)
      end
    end

    # add an intercept column to the data frame, 
    # so the intercept will be used whenever specified
    data[:intercept] = Array.new(n) {1.0}

    #################################################################################
    # Transform categorical (non-numeric) variables to sets of indicator vectors,
    # and update the fixed and random effects names accordingly
    #################################################################################

    new_names = data.create_indicator_vectors_for_categorical_vectors!
    categorical_names = new_names.keys

    # Replace the categorical variable names in non-interaction terms with the
    # names of the corresponding indicator vectors 
    [fixed_effects, *random_effects].each do |effects_array|
      reduced = effects_array.include?(:intercept)
      categorical_names.each do |name|
        ind = effects_array.find_index(name)
        if ind then
          effects_array[ind..ind] = reduced ? new_names[name][1..-1] : new_names[name]
          reduced = true
        end
      end
    end

    ################################################################
    # Deal with interaction effects and nested grouping factors
    ################################################################

    # this Array will collect the names of all interaction effects, which have a correponding
    # vector in the data frame 
    interaction_names = Array.new
    
    # Proc that adds a new vector to the data frame for an interaction of two numeric vectors,
    # and returns the name of the newly created data frame column
    num_x_num = Proc.new do |ef0, ef1| 
      inter_name = "#{ef0}_interaction_with_#{ef1}".to_sym
      unless interaction_names.include? inter_name
        data[inter_name] = data[ef0] * data[ef1] 
        interaction_names.push(inter_name)
      end
      inter_name
    end

    # Proc that adds new vectors to the data frame for an interaction of a numeric (ef0) and 
    # a categorical (ef1) vector, and returns the names of the newly created data frame columns
    num_x_cat = Proc.new do |ef0, ef1, has_noninteraction_ef0| 
      # if ef0 is present as a fixed/random (whichever applicable) effect already,
      # then first level of the interaction factor should be removed
      indicator_column_names = if has_noninteraction_ef0 then 
                                 new_names[ef1][1..-1]
                               else
                                 new_names[ef1]
                               end
      num_x_cat_interactions = Array.new
      indicator_column_names.each do |name|
        inter_name = "#{ef0}_interaction_with_#{name}".to_sym
        unless interaction_names.include? inter_name
          data[inter_name] = data[ef0] * data[name] 
          interaction_names.push(inter_name)
        end
        num_x_cat_interactions.push(inter_name)
      end
      num_x_cat_interactions
    end

    # Proc that adds new vectors to the data frame for an interaction of two categorical 
    # vectors, and returns the names of the newly created data frame columns
    cat_x_cat = Proc.new do |ef0, ef1, has_noninteraction_ef0, has_noninteraction_ef1, has_intercept| 
      # if noninteraction effects are present, then some levels of the interaction variable need to 
      # be removed, in order to preserve full column rank of the model matrix
      case [has_noninteraction_ef0, has_noninteraction_ef1]
      when [true, true]
        names_ef0 = new_names[ef0][1..-1]
        names_ef1 = new_names[ef1][1..-1]
      when [true, false]
        names_ef0 = new_names[ef0]
        names_ef1 = new_names[ef1][1..-1]
      when [false, true]
        names_ef0 = new_names[ef0][1..-1]
        names_ef1 = new_names[ef1]
      else
        names_ef0 = new_names[ef0]
        names_ef1 = new_names[ef1]
      end

      cat_x_cat_interactions = Array.new
      names_ef0.each do |name0|
        names_ef1.each do |name1|
          inter_name = "#{name0}_interaction_with_#{name1}".to_sym
          unless interaction_names.include? inter_name
            data[inter_name] = data[name0] * data[name1] 
            interaction_names.push(inter_name)
          end
          cat_x_cat_interactions.push(inter_name)
        end
      end
      # remove the last level of the interaction variable if an intercept is present;
      # if noninteraction effects are present, then this is already accounted for
      if !has_noninteraction_ef0 & !has_noninteraction_ef1 & has_intercept then
        cat_x_cat_interactions.pop
      end

      cat_x_cat_interactions
    end

    # Deal with interaction effects among the fixed and random effects terms
    [fixed_effects, *random_effects].each do |effects_array|
      effects_array.each_with_index do |ef, ind|
        if ef.is_a? Array then
          raise(NotImplementedError, "interaction effects can only be bi-variate") unless ef.length == 2

          str0, str1 = "^#{ef[0]}_lvl_", "^#{ef[1]}_lvl_"
          has_noninteraction_ef0 = (effects_array.include?(ef[0]) || effects_array.any? { |e| e.to_s =~ /#{str0}/ })
          has_noninteraction_ef1 = (effects_array.include?(ef[1]) || effects_array.any? { |e| e.to_s =~ /#{str1}/ })
          has_intercept = effects_array.include?(:intercept)

          case [categorical_names.include?(ef[0]), categorical_names.include?(ef[1])]
          when [true, true]
            effects_array[ind..ind] = cat_x_cat.call(ef[0], ef[1], has_noninteraction_ef0, 
                                                     has_noninteraction_ef1, has_intercept)
          when [true, false]
            effects_array[ind..ind] = num_x_cat.call(ef[1], ef[0], has_noninteraction_ef1)
          when [false, true]
            effects_array[ind..ind] = num_x_cat.call(ef[0], ef[1], has_noninteraction_ef0)
          else
            effects_array[ind] = num_x_num.call(ef[0], ef[1])
          end
        end
      end
    end

    # Deal with nestings in the random effects
    grouping.each_with_index do |grp, ind|
      if grp.is_a? Array then
        raise(NotImplementedError, "nested effects can only be bi-variate") unless grp.length == 2
        var1, var2 = data[grp[0]].to_a, data[grp[1]].to_a
        combination_var = var1.zip(var2).map { |p| p.join("_and_") }
        combination_var_name = "#{grp[0]}_and_#{grp[1]}"
        data[combination_var_name] = combination_var
        grouping[ind] = combination_var_name
      end
    end

    return {fixed_effects: fixed_effects,
            random_effects: random_effects,
            grouping: grouping,
            data: data}
  end
end
