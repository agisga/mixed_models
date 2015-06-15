require 'MixedModels'

describe Daru::DataFrame do
  context "#interaction_df_with" do
    it "creates a data frame of interaction effects in the sense of linear models" do
      df1 = Daru::DataFrame.new([[1,2],[3,4]], order: ['a','b'])
      df2 = Daru::DataFrame.new([[1,1],[2,2]], order: ['x','y'])
      df  = Daru::DataFrame.new([[1,2],[2,4],[3,4],[6,8]], order: ['ax','ay','bx','by'])
      expect(df1.interaction_df_with(df2).to_a).to eq(df.to_a)
    end
  end

  context "#replace_categorical_vectors_with_indicators!" do
    before do
      @df = Daru::DataFrame.new([(1..7).to_a, 
                                 ['a','b','b','a','c','d','c'],
                                 [:q, :p, :q, :p, :p, :q, :p]],
                                order: ['num','nonum','sym']) 
    end

    it "replaces all non-numeric vectors with 0-1 indicator columns" do
      @df.replace_categorical_vectors_with_indicators!
      df2 = Daru::DataFrame.new([(1..7).to_a, 
                                 [0.0,1.0,1.0,0.0,0.0,0.0,0.0],
                                 [0.0,0.0,0.0,0.0,1.0,0.0,1.0],
                                 [0.0,0.0,0.0,0.0,0.0,1.0,0.0],
                                 [0.0,1.0,0.0,1.0,1.0,0.0,1.0]],
                                order: ['num','nonum_lvl_b',
                                        'nonum_lvl_c','nonum_lvl_d','sym_lvl_p']) 
      expect(@df.to_a).to eq(df2.to_a)
    end

    it "returns the names of the deleted vectors" do
      names = @df.replace_categorical_vectors_with_indicators!
      expect(names.keys).to eq([:nonum, :sym])
    end

    it "returns the names of the newly created vectors" do
      names = @df.replace_categorical_vectors_with_indicators!
      expect(names[:nonum] + names[:sym]).to eq([:nonum_lvl_b, :nonum_lvl_c, 
                                                 :nonum_lvl_d, :sym_lvl_p])
    end
  end
end

describe Daru::Vector do
  context "#to_indicator_cols_df" do
    it "creates a data frame of indicator variables" do
      a  = Daru::Vector.new([1,2,3,2,1,1])
      df = Daru::DataFrame.new([[1.0,0.0,0.0,0.0,1.0,1.0],
                                [0.0,1.0,0.0,1.0,0.0,0.0],
                                [0.0,0.0,1.0,0.0,0.0,0.0]], 
                               order: ['v_lvl_1','v_lvl_2','v_lvl_3'])
      expect(a.to_indicator_cols_df(name: 'v', for_model_without_intercept: true).to_a).to eq(df.to_a)
    end
  end
end
