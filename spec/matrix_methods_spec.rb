require 'MixedModels'

describe NMatrix do

  it "Generate a block-diagonal NMatrix" do
    a = NMatrix.new([2,2],[1,2,3,4])
    b = NMatrix.new([1,1],[123.0])
    c = NMatrix.new([3,3],[1,2,3,1,2,3,1,2,3])
    m = NMatrix.block_diagonal(a,b,c, dtype: :int32, stype: :yale)
    expect(m).to eq(NMatrix.new([6,6], [1, 2,   0, 0, 0, 0,
                                        3, 4,   0, 0, 0, 0,
                                        0, 0, 123, 0, 0, 0,
                                        0, 0,   0, 1, 2, 3,
                                        0, 0,   0, 1, 2, 3,
                                        0, 0,   0, 1, 2, 3], dtype: :int32, stype: :yale))
  end

end
