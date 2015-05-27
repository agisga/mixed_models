require 'nmatrix'

class NMatrix

  # Compute the the Kronecker product of two row vectors (NMatrix of shape [1,n])
  #
  # === Arguments
  #
  #   * +v+ - A NMatrix of shape [1,n] (i.e. a row vector)
  #
  # === Usage 
  #
  #  a = NMatrix.new([1,3], [0,1,0])
  #  b = NMatrix.new([1,2], [3,2])
  #  a.kron_prod_1D b #  =>  [ [0, 0, 3, 2, 0, 0] ]
  #  
  def kron_prod_1D(v)
    unless self.dimensions==2 and v.dimensions==2 and self.shape[0]==1 and v.shape[0]==1
      raise ArgumentError, "Implemented for NMatrix of shape [1,n] (i.e. one row) only."
    end
    #TODO: maybe some outer product function from LAPACK would be more efficient to compute for m
    m = self.transpose.dot v
    l = self.shape[1]*v.shape[1]
    m.reshape!([1,l])
    return m
  end

  class << self

    # Generate a block-diagonal NMatrix from the supplied 2D square matrices.
    #
    # ==== Arguments
    #
    #  * +params+ - An array that collects all arguments passed to the method. The method
    #                can receive any number of arguments. The last entry of +params+ is a 
    #                hash of options from NMatrix#initialize. All other entries of +params+ are 
    #                the blocks of the desired block-diagonal matrix, which are supplied
    #                as square 2D NMatrix objects.
    #
    # ==== Usage
    #
    #  a = NMatrix.new([2,2],[1,2,3,4])
    #  b = NMatrix.new([1,1],[123],dtype: :int32)
    #  c = NMatrix.new([3,3],[1,2,3,1,2,3,1,2,3], dtype: :float64)
    #  m = NMatrix.block_diagonal(a,b,c,dtype: :int64, stype: :yale)
    #      => 
    #      [
    #        [1, 2,   0, 0, 0, 0]
    #        [3, 4,   0, 0, 0, 0]
    #        [0, 0, 123, 0, 0, 0]
    #        [0, 0,   0, 1, 2, 3]
    #        [0, 0,   0, 1, 2, 3]
    #        [0, 0,   0, 1, 2, 3]
    #      ]
    #
    def block_diagonal(*params)
      options = params.last.is_a?(Hash) ? params.pop : {}

      block_sizes = [] #holds the size of each matrix block
      params.each do |b|
        raise ArgumentError, "Only NMatrix objects allowed" unless b.is_a?(NMatrix)
        raise ArgumentError, "Only 2D matrices allowed" unless b.shape.size == 2
        raise ArgumentError, "Only square matrices allowed" unless b.shape[0] == b.shape[1]
        block_sizes << b.shape[0]
      end

      bdiag = NMatrix.zeros(block_sizes.sum, options)
      (0...params.length).each do |n|
        # First determine the size and position of the n'th block in the block-diagonal matrix
        k = block_sizes[n]
        block_pos = block_sizes[0...n].sum
        # populate the n'th block in the block-diagonal matrix
        (0...k).each do |i|
          (0...k).each do |j|
            bdiag[block_pos+i,block_pos+j] = params[n][i,j]
          end
        end
      end

      return bdiag
    end

  end
end
