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

    # Create for all non-numeric vectors 0-1-indicator columns.
    # Returns the names/indices of the non-numeric vectors and the corresponding created new vectors.
    #
    # === Returns
    #
    # A Hash where the keys are the names of all non-numeric vectors, and their values
    # are Arrays containing the names of the corresponding 0-1 valued vectors
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
    # > df = Daru::DataFrame.new([(1..7).to_a, ['a','b','b','a','c','d','c']], order: ['int','char']) 
    # > df.create_indicator_vectors_for_categorical_vectors!
    #   # => {:char=>[:char_lvl_a, :char_lvl_b, :char_lvl_c, :char_lvl_d]}
    # > df
    #   # => #<Daru::DataFrame:70180517472080 @name = 75ddbda4-d4df-41b2-a41e-2f600764061b @size = 7>
    #              int       char char_lvl_a char_lvl_b char_lvl_c char_lvl_d 
    #     0          1          a        1.0        0.0        0.0        0.0 
    #     1          2          b        0.0        1.0        0.0        0.0 
    #     2          3          b        0.0        1.0        0.0        0.0 
    #     3          4          a        1.0        0.0        0.0        0.0 
    #     4          5          c        0.0        0.0        1.0        0.0 
    #     5          6          d        0.0        0.0        0.0        1.0 
    #     6          7          c        0.0        0.0        1.0        0.0 
    #
    def create_indicator_vectors_for_categorical_vectors!
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
          levels.each do |l|
            ind = "#{name}_lvl_#{l}".to_sym
            col = Array.new
            vec.each { |e| e==l ? col.push(1.0) : col.push(0.0) }
            vec_for_level_l = Daru::Vector.new(col)
            self.add_vector(ind, vec_for_level_l)
            level_indices.push(ind)
          end
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
      names = levels.map { |e| "#{name}_lvl_#{e}".to_sym }
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
