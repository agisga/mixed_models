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
  end
end
