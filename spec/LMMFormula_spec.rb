require 'mixed_models'

RSpec.describe MixedModels::LMMFormula do
  it "produces LMM#from_daru input" do
    intercept = MixedModels::lmm_variable(:intercept)
    x         = MixedModels::lmm_variable(:x)
    y         = MixedModels::lmm_variable(:y)
    u         = MixedModels::lmm_variable(:u)
    z         = intercept + x + y + (intercept + x| u) 
    input     = z.to_input_for_lmm_from_daru
    expect(input[:fixed_effects]).to eq([:intercept, :x, :y])
    expect(input[:random_effects]).to eq([[:intercept, :x]])
    expect(input[:grouping]).to eq([:u])
  end

  it "handles interaction fixed effects" do
    intercept = MixedModels::lmm_variable(:intercept)
    x         = MixedModels::lmm_variable(:x)
    y         = MixedModels::lmm_variable(:y)
    u         = MixedModels::lmm_variable(:u)
    z         = intercept + x + y + x*y + (intercept| u)
    input     = z.to_input_for_lmm_from_daru
    expect(input[:fixed_effects]).to eq([:intercept, :x, :y, [:x, :y]])
    expect(input[:random_effects]).to eq([[:intercept]])
    expect(input[:grouping]).to eq([:u])
  end

  it "handles interaction random effects" do
    intercept = MixedModels::lmm_variable(:intercept)
    x         = MixedModels::lmm_variable(:x)
    y         = MixedModels::lmm_variable(:y)
    u         = MixedModels::lmm_variable(:u)
    z         = intercept+ (intercept + x + y + x*y| u)
    input     = z.to_input_for_lmm_from_daru
    expect(input[:fixed_effects]).to eq([:intercept])
    expect(input[:random_effects]).to eq([[:intercept, :x, :y, [:x, :y]]])
    expect(input[:grouping]).to eq([:u])
  end

  it "handles crossed random effects" do
    intercept = MixedModels::lmm_variable(:intercept)
    x         = MixedModels::lmm_variable(:x)
    y         = MixedModels::lmm_variable(:y)
    u         = MixedModels::lmm_variable(:u)
    w         = MixedModels::lmm_variable(:w)
    z         = intercept+ (intercept + x + y + x*y| u) + (intercept + x*y | w)
    input     = z.to_input_for_lmm_from_daru
    expect(input[:fixed_effects]).to eq([:intercept])
    expect(input[:random_effects]).to eq([[:intercept, :x, :y, [:x, :y]], [:intercept, [:x,:y]]])
    expect(input[:grouping]).to eq([:u, :w])
  end

  it "handles nested random effects" do
    intercept = MixedModels::lmm_variable(:intercept)
    x         = MixedModels::lmm_variable(:x)
    y         = MixedModels::lmm_variable(:y)
    u         = MixedModels::lmm_variable(:u)
    w         = MixedModels::lmm_variable(:w)
    z         = intercept + x + y + (intercept + x| u*w) 
    input     = z.to_input_for_lmm_from_daru
    expect(input[:fixed_effects]).to eq([:intercept, :x, :y])
    expect(input[:random_effects]).to eq([[:intercept, :x]])
    expect(input[:grouping]).to eq([[:u,:w]])
  end
end
