require 'MixedModels'

ALL_DTYPES = [:byte,:int8,:int16,:int32,:int64, :float32,:float64, :object,
  :rational32,:rational64,:rational128, :complex64, :complex128]

describe NMatrix do

  ALL_DTYPES.each do |dtype|
    [:dense, :yale].each do |stype|
      context "#kron_prod_1D #{dtype} #{stype}" do
        it "computes the Kronecker product of two [1,n] NMatrix objects" do
          a = NMatrix.new([1,3], [0,1,0], dtype: dtype, stype: stype)
          b = NMatrix.new([1,2], [3,2], dtype: dtype, stype: stype)
          m = NMatrix.new([1,6], [0, 0, 3, 2, 0, 0], dtype: dtype, stype: stype)
          expect(a.kron_prod_1D b).to eq(m)
        end
      end
    end
  end

  ALL_DTYPES.each do |dtype|
    [:dense, :yale].each do |stype|
      context "#khatri_rao_rows #{dtype} #{stype}" do
        it "computes the simplified Khatri-Rao product of two NMatrix objects" do
          a = NMatrix.new([3,2], [1,2,1,2,1,2], dtype: dtype, stype: stype)
          b = NMatrix.new([3,2], (1..6).to_a, dtype: dtype, stype: stype)
          m = NMatrix.new([3,4], [1, 2,  2,  4,
                                  3, 4,  6,  8,
                                  5, 6, 10, 12], dtype: dtype, stype: stype)
          expect(a.khatri_rao_rows b).to eq(m)
        end
      end
    end
  end

  [:float32, :float64].each do |dtype|
    context "#matrix_valued_solve #{dtype} dense" do
      it "solves a matrix-valued linear system A * X = B" do
        a = NMatrix.random([10,10], dtype: dtype)
        m = rand(1..10)
        b = NMatrix.random([10,m], dtype: dtype)
        x = a.matrix_valued_solve(b)
        r = a.dot(x) - b
        expect(r.max(1).max).to be_within(1e-6).of(0.0)
      end
    end
  end

  [:float32, :float64].each do |dtype|
    context "#triangular_solve #{dtype} dense" do
      it "solves a linear system A * x = b with a lower triangular A" do
        a = NMatrix.random([10,10], dtype: dtype).tril
        b = NMatrix.random([10,1], dtype: dtype)
        x = a.triangular_solve(:lower, b)
        r = a.dot(x) - b
        expect(r.max).to be_within(1e-6).of(0.0)
      end

      it "solves a linear system A * x = b with an upper triangular A" do
        a = NMatrix.random([10,10], dtype: dtype).triu
        b = NMatrix.random([10,1], dtype: dtype)
        x = a.triangular_solve(:upper, b)
        r = a.dot(x) - b
        expect(r.max).to be_within(1e-6).of(0.0)
      end
    end
  end

  ALL_DTYPES.each do |dtype|
    [:dense, :yale, :list].each do |stype|
      context "#block_diagonal #{dtype} #{stype}" do
        it "creates a block-diagonal NMatrix" do
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
