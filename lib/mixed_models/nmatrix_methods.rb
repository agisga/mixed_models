# Copyright (c) 2015 Alexej Gossmann 

class NMatrix

  # Compute the the Kronecker product of two row vectors (NMatrix of shape [1,n])
  #
  # === Arguments
  #
  # * +v+ - A NMatrix of shape [1,n] (i.e. a row vector)
  #
  # === Usage 
  #
  #    a = NMatrix.new([1,3], [0,1,0])
  #    b = NMatrix.new([1,2], [3,2])
  #    a.kron_prod_1D b #  =>  [ [0, 0, 3, 2, 0, 0] ]
  #  
  def kron_prod_1D(v)
    unless self.dimensions==2 && v.dimensions==2 && self.shape[0]==1 && v.shape[0]==1
      raise ArgumentError, "Implemented for NMatrix of shape [1,n] (i.e. one row) only."
    end
    #TODO: maybe some outer product function from LAPACK would be more efficient to compute for m
    m = self.transpose.dot v
    l = self.shape[1]*v.shape[1]
    return m.reshape([1,l])
  end

  # Compute a simplified version of the Khatri-Rao product of +self+ and other NMatrix +mat+.
  # The i'th row of the resulting matrix is the Kronecker product of the i'th row of +self+
  # and the i'th row of +mat+.
  #
  # === Arguments
  #
  # * +mat+ - A 2D NMatrix object
  #
  # === Usage
  #
  #   a = NMatrix.new([3,2], [1,2,1,2,1,2], dtype: dtype, stype: stype)
  #   b = NMatrix.new([3,2], (1..6).to_a, dtype: dtype, stype: stype)
  #   m = a.khatri_rao_rows b # =>  [ [1.0, 2.0,  2.0,  4.0]
  #                                   [3.0, 4.0,  6.0,  8.0]
  #                                   [5.0, 6.0, 10.0, 12.0] ]
  #
  def khatri_rao_rows(mat)
    raise NotImplementedError, "Implemented for 2D matrices only" unless self.dimensions==2 and mat.dimensions==2
    n = self.shape[0]
    raise NotImplementedError, "Both matrices must have the same number of rows" unless n==mat.shape[0]
    m = self.shape[1]*mat.shape[1]
    prod_dtype = NMatrix.upcast(self.dtype, mat.dtype)
    khrao_prod = NMatrix.new([n,m], dtype: prod_dtype)
    (0...n).each do |i|
      kronecker_prod = self.row(i).kron_prod_1D mat.row(i)
      khrao_prod[i,0...m] = kronecker_prod
    end
    return khrao_prod
  end

  # Solve a linear system A * X = B, where A is a lower triangular matrix, 
  # X and B are vectors or matrices. 
  #
  # === Arguments
  #
  # * +uplo+ - flag indicating whether the matrix is lower or upper triangular;
  #   possible values are :lower and :upper
  # * +rhs+  - the right hand side, an NMatrix object
  #
  # === Usage
  #
  #   a = NMatrix.new(3, [4, 0, 0, -2, 2, 0, -4, -2, -0.5], dtype: :float64)
  #   b = NMatrix.new([3,1], [-1, 17, -9], dtype: :float64)
  #   x = a.triangular_solve(:lower, b)
  #   a.dot x # => [ [-1.0]   [17.0]   [-9.0] ]
  #
  def triangular_solve(uplo, rhs)
    raise(ArgumentError, "uplo should be :lower or :upper") unless uplo == :lower or uplo == :upper
    b = rhs.clone
    # this is the correct function call; it came up in during
    # discussion in https://github.com/SciRuby/nmatrix/issues/374
    NMatrix::BLAS::cblas_trsm(:row, :left, uplo, false, :nounit, 
                              b.shape[0], b.shape[1], 1.0, self, self.shape[0],
                              b, b.shape[1])
    return b
  end
end
