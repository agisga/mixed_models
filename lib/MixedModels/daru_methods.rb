require 'nmatrix'
require 'daru'

module Daru
  class DataFrame
    # Transform a Daru::DataFrame into a NMatrix
    #
    # === Arguments
    #
    # * +dtype+ - the +dtype+ of the returned NMatrix; defaults to +float64+
    # * +stype+ - the +stype+ of the returned NMatrix; defaults to +dense+
    # 
    def to_nm(dtype: :float64, stype: :dense)
      n, m = self.nrows, self.ncols
      data_array = Array.new 
      0.upto(n-1) { |i| data_array.concat(self.row[i].to_a) }
      return NMatrix.new([n,m], data_array, dtype: dtype, stype: stype)
    end

    # Create a data frame of interaction effects in the sense of linear models.
    # That is, it returns a Daru::DataFrame containing all pair-wise products of
    # vectors in +self+ and +other+
    #
    # === Arguments
    #
    # * +other+ - A Daru::DataFrame
    #
    # === Usage
    # > df1 = Daru::DataFrame.new([[1,2],[3,4]], order: ['a','b'])
    #          # => <Daru::DataFrame:69920382023980 @name = 900f7fbe-ad31-4ab3-8a16-232e22b17d53 @size = 2>
    #                a          b 
    #     0          1          3 
    #     1          2          4 
    #
    # > df2 = Daru::DataFrame.new([[1,1],[2,2]], order: ['x','y'])
    #           # => <Daru::DataFrame:69920381762120 @name = 3cb2690b-52a2-44b9-9a7e-686c8d92c38e @size = 2>
    #                    x          y 
    #         0          1          2 
    #         1          1          2 
    #
    # > df1.interaction_df_with df2
    #            # => <Daru::DataFrame:69920381579240 @name = d6773420-31ce-45dc-8639-b94b276b63a6 @size = 2>
    #                   ax         ay         bx         by 
    #         0          1          2          3          6 
    #         1          2          4          4          8 
    #
    def interaction_df_with other
      df = Daru::DataFrame.new([], order: [])
      self.each_vector_with_index do |vec, name|
        other.each_vector_with_index do |other_vec, other_name|
          product_name = (name.to_s + other_name.to_s).to_sym
          df[product_name] = vec * other_vec
        end
      end
      return df
    end

    # Replaces all non-numeric vectors with 0-1-indicator columns.
    # Returns the names/indices of the deleted vectors and the created new vectors.
    #
    # === Returns
    #
    # A Hash where the keys are the names of the deleted vectors, and their values
    # are Arrays containing the names of the vectors which have replaced the deleted vector
    #
    # === Arguments
    #
    # * +for_model_without_intercept+ - if false (which is the default), then 
    #                                   the indicator vector for the first level of every 
    #                                   categorical variable will be excluded. Otherwise,
    #                                   all levels for the first categorical variable are
    #                                   considered, but the first level for all others are
    #                                   excuded.
    #
    # === Usage
    #
    # > df = Daru::DataFrame.new([(1..7).to_a, ['a','b','b','a','c','d','c']],
    #                            order: ['num','nonum']) 
    # > df.replace_categorical_vectors_with_indicators!
    #   # => #<Daru::DataFrame:70062395762160 @name = c647bf98-142f-4a0e-a4da-42ca8efdd20b @size = 7>
    #                    num  nonum_lvl_b  nonum_lvl_c  nonum_lvl_d 
    #           0          1          0.0          0.0          0.0 
    #           1          2          1.0          0.0          0.0 
    #           2          3          1.0          0.0          0.0 
    #           3          4          0.0          0.0          0.0 
    #           4          5          0.0          1.0          0.0 
    #           5          6          0.0          0.0          1.0 
    #           6          7          0.0          1.0          0.0
    #
    def replace_categorical_vectors_with_indicators!(for_model_without_intercept: false)
      indices = Hash.new

      self.each_vector_with_index do |vec, name|
        # all non-numeric vectors are considered categorical data
        unless vec.type == :numeric
          levels = vec.to_a.uniq
          # sort the levels if possible
          begin
            levels.sort!
          rescue
            levels = vec.to_a.uniq
          end
          level_indices = Array.new
          unless for_model_without_intercept 
            levels.shift
            for_model_without_intercept = false
          end
          levels.each do |l|
            ind = name.to_s + "_lvl_" + l.to_s
            col = Array.new
            vec.each { |e| e==l ? col.push(1.0) : col.push(0.0) }
            vec_for_level_l = Daru::Vector.new(col)
            self.add_vector(ind, vec_for_level_l)
            level_indices.push(ind.to_sym)
          end
          # delete the categorical vector, which just got replaced
          self.delete_vector(name)
          indices[name] = level_indices
        end
      end
       
      return indices
    end
  end

  class Vector
    # Transform a Daru::Vector into a NMatrix
    #
    # === Arguments
    #
    # * +dtype+ - the +dtype+ of the returned NMatrix; defaults to +float64+
    # * +stype+ - the +stype+ of the returned NMatrix; defaults to +dense+
    # 
    def to_nm(dtype: :float64, stype: :dense)
      n = self.size
      return NMatrix.new([n,1], self.to_a, dtype: dtype, stype: stype)
    end

    # Auxiliary function which is useful for fitting of linear models.
    # Transforms a Daru::Vector, whose entries are assumed to represent levels of
    # a categorical variable, into a Daru::DataFrame with a column of zeros and ones
    # for each category. If +full+ is set to false, then the first category is discarded,
    # which is useful to generate a design matrix for linear regression, when a intercept term
    # is present in the model.
    #
    # === Arguments
    #
    # * +name+ - used for the naming of the columns of the returned data frame
    # * +for_model_without_intercept+ - if false (which is the default), then 
    #                                   the first column of the produced data 
    #                                   frame will be excluded
    #
    # === Usage
    #
    # a # => <Daru::Vector:70083983735480 @name = nil @size = 5 >
    #           nil
    #         0 1.0
    #         1 2.0
    #         2 3.0
    #         3 1.0
    #         4 1.0
    #
    # a.to_indicator_cols_df(name: 'MyVar', for_model_without_intercept: true) #   => 
    #   #<Daru::DataFrame:70083988870200 @name = 08de5ef9-5c59-4acf-9853-04289d1a4ba5 @size = 5>
    #               MyVar.1.0  MyVar.2.0  MyVar.3.0 
    #            0        1.0        0.0        0.0 
    #            1        0.0        1.0        0.0 
    #            2        0.0        0.0        1.0 
    #            3        1.0        0.0        0.0 
    #            4        1.0        0.0        0.0 
    #
    def to_indicator_cols_df(name:, for_model_without_intercept: false)
      levels = self.to_a.uniq
      names = levels.map { |e| name.to_s + "_lvl_" + e.to_s }
      unless for_model_without_intercept 
        levels.shift
        names.shift
      end

      cols_array = Array.new
      levels.each do |l|
        col = Array.new
        self.each { |e| e==l ? col.push(1.0) : col.push(0.0) }
        cols_array.push(col)
      end

      return Daru::DataFrame.new(cols_array, order: names)
    end
  end
end
