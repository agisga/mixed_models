require 'MixedModels'

RSpec.describe MixedModels do
  context "with mk_ran_ef_model_matrix" do
    it "generates a random effects model matrix using mk_ran_ef_model_matrix with one group of random effects" do
      x1  = NMatrix.new([6,2], [1,-1,1,1,1,-1,1,1,1,-1,1,1], dtype: :float64)
      x   = Array[x1]
      grp = Array[["grp1", "grp1", 2, 2, "grp3", "grp3"]]
      z   = MixedModels::mk_ran_ef_model_matrix(x, grp)
      expect(z).to eq(NMatrix.new([6,6], [1.0, -1.0, 0.0,  0.0, 0.0,  0.0,
                                          1.0,  1.0, 0.0,  0.0, 0.0,  0.0,
                                          0.0,  0.0, 1.0, -1.0, 0.0,  0.0,
                                          0.0,  0.0, 1.0,  1.0, 0.0,  0.0,
                                          0.0,  0.0, 0.0,  0.0, 1.0, -1.0,
                                          0.0,  0.0, 0.0,  0.0, 1.0,  1.0], dtype: :float64))
    end

    it "generates a random effects model matrix with mk_ran_ef_model_matrix with two groups of random effects" do
      x1  = NMatrix.new([6,2], [1,-1,1,1,1,-1,1,1,1,-1,1,1], dtype: :float64)
      x2  = NMatrix.new([6,1], [1,1,1,1,1,1], dtype: :float64)
      x   = Array[x1, x2]
      grp = Array[["grp1", "grp1", 2, 2, "grp3", "grp3"], [1,1,1,2,2,2]]
      z   = MixedModels::mk_ran_ef_model_matrix(x, grp)
      expect(z).to eq(NMatrix.new([6,8], [1.0, -1.0, 0.0,  0.0, 0.0,  0.0, 1.0, 0.0,
                                          1.0,  1.0, 0.0,  0.0, 0.0,  0.0, 1.0, 0.0,
                                          0.0,  0.0, 1.0, -1.0, 0.0,  0.0, 1.0, 0.0,
                                          0.0,  0.0, 1.0,  1.0, 0.0,  0.0, 0.0, 1.0,
                                          0.0,  0.0, 0.0,  0.0, 1.0, -1.0, 0.0, 1.0,
                                          0.0,  0.0, 0.0,  0.0, 1.0,  1.0, 0.0, 1.0], dtype: :float64))
    end

    it "generates a random effects model matrix with mk_ran_ef_model_matrix with three groups of random effects" do
      x1  = NMatrix.new([6,2], [1,-1,1,1,1,-1,1,1,1,-1,1,1], dtype: :float64)
      x2  = NMatrix.new([6,1], [1,1,1,1,1,1], dtype: :float64)
      x3  = NMatrix.new([6,2], [1,1,1,2,1,3,1,4,1,5,1,6], dtype: :float64)
      x   = Array[x1, x2, x3]
      grp = Array[["grp1", "grp1", 2, 2, "grp3", "grp3"], [1,1,1,2,2,2], ['a','b','c','a','b','c']]
      z   = MixedModels::mk_ran_ef_model_matrix(x, grp)
      expect(z).to eq(NMatrix.new([6,14], [1.0, -1.0, 0.0,  0.0, 0.0,  0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                           1.0,  1.0, 0.0,  0.0, 0.0,  0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0,
                                           0.0,  0.0, 1.0, -1.0, 0.0,  0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0,
                                           0.0,  0.0, 1.0,  1.0, 0.0,  0.0, 0.0, 1.0, 1.0, 4.0, 0.0, 0.0, 0.0, 0.0,
                                           0.0,  0.0, 0.0,  0.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0,
                                           0.0,  0.0, 0.0,  0.0, 1.0,  1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 6.0], dtype: :float64))
    end
  end

  context "with mk_ran_ef_cov_fun" do
    it "generates a Proc which parametrizes the random effects covariance matrix as a function of theta" do
      mapping = MixedModels::mk_ran_ef_cov_fun([2], [3])
      expect(mapping.call([1,2,3])).to eq(NMatrix.new([6,6], [1, 3, 0, 0, 0, 0,
                                                              0, 2, 0, 0, 0, 0,
                                                              0, 0, 1, 3, 0, 0,
                                                              0, 0, 0, 2, 0, 0,
                                                              0, 0, 0, 0, 1, 3,
                                                              0, 0, 0, 0, 0, 2], dtype: :float64))
    end

    it "generates a Proc which parametrizes the random effects covariance matrix as a function of theta" do
      mapping = MixedModels::mk_ran_ef_cov_fun([2,1], [3,2])
      expect(mapping.call([1,2,3,4])).to eq(NMatrix.new([8,8], [1, 4, 0, 0, 0, 0, 0, 0,
                                                                0, 2, 0, 0, 0, 0, 0, 0,
                                                                0, 0, 1, 4, 0, 0, 0, 0,
                                                                0, 0, 0, 2, 0, 0, 0, 0,
                                                                0, 0, 0, 0, 1, 4, 0, 0,
                                                                0, 0, 0, 0, 0, 2, 0, 0,
                                                                0, 0, 0, 0, 0, 0, 3, 0,
                                                                0, 0, 0, 0, 0, 0, 0, 3], dtype: :float64))
    end

    it "generates a Proc which parametrizes the random effects covariance matrix as a function of theta" do
      mapping = MixedModels::mk_ran_ef_cov_fun([2,1,2], [3,2,3])
      expect(mapping.call([1,2,3,4,5,6,7])).to eq(NMatrix.new([14,14], [1, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                        0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                        0, 0, 1, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                        0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                        0, 0, 0, 0, 1, 6, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                        0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                        0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
                                                                        0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0,
                                                                        0, 0, 0, 0, 0, 0, 0, 0, 4, 7, 0, 0, 0, 0,
                                                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0,
                                                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 7, 0, 0,
                                                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0,
                                                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 7,
                                                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5], dtype: :float64))
    end
  end

end
