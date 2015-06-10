require 'MixedModels'

RSpec.describe MixedModels do
  before :each do
    grp = Array[["grp1", "grp1", 2, 2, "grp3", "grp3"]]
    x1  = NMatrix.new([6,2], [1,-1,1,1,1,-1,1,1,1,-1,1,1], dtype: :float64)
    x   = Array[x1]

  end

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
