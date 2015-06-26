require 'mixed_models'

describe Daru::DataFrame do
  context "#interaction_df_with" do
    it "creates a data frame of interaction effects in the sense of linear models" do
      df1 = Daru::DataFrame.new([[1,2],[3,4]], order: ['a','b'])
      df2 = Daru::DataFrame.new([[1,1],[2,2]], order: ['x','y'])
      df  = Daru::DataFrame.new([[1,2],[2,4],[3,4],[6,8]], order: ['a_and_x','a_and_y','b_and_x','b_and_y'])
      expect(df1.interaction_df_with(df2).to_a).to eq(df.to_a)
    end
  end

  context "#create_indicator_vectors_for_categorical_vectors!" do
    before do
      @df = Daru::DataFrame.new([(1..7).to_a, 
                                 ['a','b','b','a','c','d','c'],
                                 [:q, :p, :q, :p, :p, :q, :p]],
                                order: ['num','char','sym']) 
    end

    it "creates 0-1 indicator columns for all non-numeric vectors" do
      @df.create_indicator_vectors_for_categorical_vectors!
      df2 = Daru::DataFrame.new([(1..7).to_a, 
                                 ['a','b','b','a','c','d','c'],
                                 [:q, :p, :q, :p, :p, :q, :p],
                                 [1.0,0.0,0.0,1.0,0.0,0.0,0.0],
                                 [0.0,1.0,1.0,0.0,0.0,0.0,0.0],
                                 [0.0,0.0,0.0,0.0,1.0,0.0,1.0],
                                 [0.0,0.0,0.0,0.0,0.0,1.0,0.0],
                                 [0.0,1.0,0.0,1.0,1.0,0.0,1.0],
                                 [1.0,0.0,1.0,0.0,0.0,1.0,0.0]],
                                order: ['num', 'char', 'sym', 'char_lvl_a', 'char_lvl_b',
                                        'char_lvl_c', 'char_lvl_d', 'sym_lvl_p', 'sym_lvl_q']) 
      expect(@df.to_a).to eq(df2.to_a)
    end

    it "returns the names of all non-numeric vectors" do
      names = @df.create_indicator_vectors_for_categorical_vectors!
      expect(names.keys).to eq([:char, :sym])
    end

    it "returns the names of the newly created 0-1 indicator vectors" do
      names = @df.create_indicator_vectors_for_categorical_vectors!
      expect(names.values.flatten).to eq([:char_lvl_a, :char_lvl_b, :char_lvl_c, 
                                          :char_lvl_d, :sym_lvl_p, :sym_lvl_q])
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
