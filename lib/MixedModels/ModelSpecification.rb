module MixedModels 

  # Generate the random effects model matrix Z in the linear mixed effects model 
  # y = X*beta + Z*b + epsilon,
  # from the matrices +x[i]+, which contain the the covariates for random effects
  # with a common grouping structure given as +grp[i]+.
  #
  # === Arguments
  #
  # * +x+   - Array of NMatrix objects. Each matrix +x[i]+ contains the covariates
  #           for a group of random effects with a common grouping structure.
  #           Typically, a given +x[i]+ contains correlated random effects, and the
  #           random effects from different matrices +x[i]+ are modeled as uncorrelated.  
  # * +grp+ - Array of Arrays. Each +grp[i]+ has length equal to the number of rows in +x[i]+,
  #           and specifies which observations in +x[i]+ correspond to which group.
  #
  # === References
  # 
  # * Douglas Bates, Martin Maechler, Ben Bolker, Steve Walker, 
  #   "Fitting Linear Mixed - Effects Models using lme4". arXiv:1406.5823v1 [stat.CO]. 2014.
  #
  # === Usage
  #
  # grp = Array[["grp1", "grp1", 2, 2, "grp3", "grp3"]]
  # x1  = NMatrix.new([6,2], [1,-1,1,1,1,-1,1,1,1,-1,1,1], dtype: :float64)
  # x   = Array[x1]
  # z   = MixedModels::mk_ran_ef_model_matrix(x, grp)
  #         # => 
  #           [
  #             [1.0, -1.0, 0.0,  0.0, 0.0,  0.0]
  #             [1.0,  1.0, 0.0,  0.0, 0.0,  0.0]
  #             [0.0,  0.0, 1.0, -1.0, 0.0,  0.0]
  #             [0.0,  0.0, 1.0,  1.0, 0.0,  0.0]
  #             [0.0,  0.0, 0.0,  0.0, 1.0, -1.0]
  #             [0.0,  0.0, 0.0,  0.0, 1.0,  1.0]
  #           ]
  #
  def MixedModels.mk_ran_ef_model_matrix(x, grp)
    num_x = x.length  # number of raw random effects matrices
    raise(ArgumentError, "Number of X matrices different than the number of grouping structures") unless num_x = grp.length
    n = x[0].shape[0] # number of observations in each matrix x[i]

    z = Array.new
    z_ncol  = 0
    (0...num_x).each do |i|
      grp_levels = grp[i].uniq
      m          = grp_levels.length 
      grp_mat    = NMatrix.zeros([n,m], dtype: :float64)
      (0...m).each do |j|
        (0...n).each do |k|
          grp_mat[k,j] = (grp[i][k] == grp_levels[j] ? 1.0 : 0.0)
        end
      end
      z[i] = grp_mat.khatri_rao_rows x[i]
      z_ncol += z[i].shape[1] 
    end

    z_model_mat = NMatrix.new([n, z_ncol], dtype: :float64)
    start_index = 0
    end_index   = 0
    z.each do |zi|
      end_index += zi.shape[1]
      z_model_mat[0...n, start_index...end_index] = zi
      start_index += zi.shape[1]
    end
    
    return z_model_mat
  end
end
