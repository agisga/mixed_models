require 'mixed_models'

ALL_DTYPES = [:byte,:int8,:int16,:int32,:int64, :float32,:float64, :object, :complex64, :complex128]

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
    context "#triangular_solve #{dtype} dense" do
      context "when lower triangular" do
        let(:a) { NMatrix.new([3,3], [1, 0, 0, 2, 0.5, 0, 3, 3, 9], dtype: dtype) }

        it "solves a linear system A * x = b with vector b" do
          b = NMatrix.new([3,1], [1,2,3], dtype: dtype)
          x = a.triangular_solve(:lower, b)
          r = a.dot(x) - b
          expect(r.max).to be_within(1e-6).of(0.0)
        end

        it "solves a linear system A * X = B with narrow B" do
          b = NMatrix.new([3,2], [1,2,3,4,5,6], dtype: dtype)
          x = a.triangular_solve(:lower, b)
          r = (a.dot(x) - b).to_flat_a
          expect(r.max).to be_within(1e-6).of(0.0)
        end

        it "solves a linear system A * X = B with wide B" do
          b = NMatrix.new([3,5], (1..15).to_a, dtype: dtype)
          x = a.triangular_solve(:lower, b)
          r = (a.dot(x) - b).to_flat_a
          expect(r.max).to be_within(1e-6).of(0.0)
        end
      end

      context "when upper triangular" do
        let(:a) { NMatrix.new([3,3], [3, 2, 1, 0, 2, 0.5, 0, 0, 9], dtype: dtype) }

        it "solves a linear system A * x = b with vector b" do
          b = NMatrix.new([3,1], [1,2,3], dtype: dtype)
          x = a.triangular_solve(:upper, b)
          r = a.dot(x) - b
          expect(r.max).to be_within(1e-6).of(0.0)
        end

        it "solves a linear system A * X = B with narrow B" do
          b = NMatrix.new([3,2], [1,2,3,4,5,6], dtype: dtype)
          x = a.triangular_solve(:upper, b)
          r = (a.dot(x) - b).to_flat_a
          expect(r.max).to be_within(1e-6).of(0.0)
        end

        it "solves a linear system A * X = B with a wide B" do
          b = NMatrix.new([3,5], (1..15).to_a, dtype: dtype)
          x = a.triangular_solve(:upper, b)
          r = (a.dot(x) - b).to_flat_a
          expect(r.max).to be_within(1e-6).of(0.0)
        end
      end
    end
  end
end
