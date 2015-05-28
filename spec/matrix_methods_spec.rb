require 'MixedModels'

ALL_DTYPES = [:byte,:int8,:int16,:int32,:int64, :float32,:float64, :object,
  :rational32,:rational64,:rational128, :complex64, :complex128]

describe NMatrix do

  ALL_DTYPES.each do |dtype|
    [:dense, :yale].each do |stype|
      context "#kron_prod_1D #{dtype} #{stype}" do
        before do 
          a = NMatrix.new([1,3], [0,1,0], dtype: dtype, stype: stype)
          b = NMatrix.new([1,2], [3,2], dtype: dtype, stype: stype)
          m = NMatrix.new([1,6],[0, 0, 3, 2, 0, 0], dtype: dtype, stype: stype)
        end
        it "Compute the Kronecker product of two [1,n] NMatrix objects" do
          expect(a.kron_prod_1D b).to eq(m)
        end
      end
    end
  end

  ALL_DTYPES.each do |dtype|
    [:dense, :yale].each do |stype|
      context "#khatri_rao_rows #{dtype} #{stype}" do
        before do 
          a = NMatrix.new([3,2], [1,2,1,2,1,2], dtype: dtype, stype: stype)
          b = NMatrix.new([3,2], (1..6).to_a, dtype: dtype, stype: stype)
          m = NMatrix.new([3,4], [1, 2,  2,  4,
                                  3, 4,  6,  8,
                                  5, 6, 10, 12], dtype: dtype, stype: stype)
        end
        it "Compute the simplified Khatri-Rao product of two NMatrix objects" do
          expect(a.khatri_rao_rows b).to eq(m)
        end
      end
    end
  end

  ALL_DTYPES.each do |dtype|
    [:dense, :yale, :list].each do |stype|
      context "#block_diagonal #{dtype} #{stype}" do
        it "block_diagonal() creates a block-diagonal NMatrix" do
          a = NMatrix.new([2,2],[1,2,
                                 3,4])
          b = NMatrix.new([1,1],[123.0])
          c = NMatrix.new([3,3],[1,2,3,
                                 1,2,3,
                                 1,2,3])
          m = NMatrix.block_diagonal(a,b,c, dtype: dtype, stype: stype)
          expect(m).to eq(NMatrix.new([6,6], [1, 2,   0, 0, 0, 0,
                                              3, 4,   0, 0, 0, 0,
                                              0, 0, 123, 0, 0, 0,
                                              0, 0,   0, 1, 2, 3,
                                              0, 0,   0, 1, 2, 3,
                                              0, 0,   0, 1, 2, 3], dtype: dtype, stype: stype))
        end
      end
    end
  end

end
