require 'MixedModels'

describe Daru::DataFrame do
  context "#interaction_df_with" do
    it "creates a data frame of interaction effects in the sense of linear models" do
      df1 = Daru::DataFrame.new([[1,2],[3,4]], order: ['a','b'])
      df2 = Daru::DataFrame.new([[1,1],[2,2]], order: ['x','y'])
      df  = Daru::DataFrame.new([[1,2],[2,4],[3,4],[6,8]], order: ['a_and_x','a_and_y','b_and_x','b_and_y'])
      expect(df1.interaction_df_with(df2).to_a).to eq(df.to_a)
    end
  end
end

describe Daru::Vector do
  context "#to_indicator_cols_df" do
    it "creates a data frame of indicator variables" do
      a  = Daru::Vector.new([1,2,3,2,1,1])
      df = Daru::DataFrame.new([[1.0,0.0,0.0,0.0,1.0,1.0],
                                [0.0,1.0,0.0,1.0,0.0,0.0],
                                [0.0,0.0,1.0,0.0,0.0,0.0]], order: ['v1','v2','v3'])
      expect(a.to_indicator_cols_df(name: 'v', for_model_without_intercept: true).to_a).to eq(df.to_a)
    end
  end
end
