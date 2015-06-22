require 'mixed_models'

RSpec.describe MixedModels::NelderMead do
  context "when unconstrained" do
    let(:min) { MixedModels::NelderMead.minimize(start_point: [0,0,0]) { |x| x[0]**2 + (x[1]-4)**2 + (x[2]+7)**2 } }

    it "finds an #x_minimum close to the true solution" do 
      q = [0,4,-7]
      min.x_minimum.each_with_index do |p, i|
        expect(p).to be_within(1e-6).of(q[i])
      end
    end

    it "yields an #f_minimum close to the true solution" do 
      expect(min.f_minimum).to be_within(1e-6).of(0)
    end
  end

  context "when constrained from below" do
    let(:min) do 
      MixedModels::NelderMead.minimize(start_point: [0,0,0], lower_bound: [0,0,-5], epsilon: 1e-12) do |x| 
        (x[0]-1)**2 + (x[1]-4)/2 + (x[2]+7)**3 
      end
    end

    it "finds an #x_minimum close to the true solution" do 
      q = [1,0,-5]
      min.x_minimum.each_with_index do |p, i|
        expect(p).to be_within(1e-6).of(q[i])
      end
    end

    it "yields an #f_minimum close to the true solution" do 
      expect(min.f_minimum).to be_within(1e-6).of(6)
    end
  end

  context "when constrained from above" do
    let(:min) do 
      MixedModels::NelderMead.minimize(start_point: [0,0,0], upper_bound: [5,5,5], epsilon: 1e-12) do |x| 
        (x[0]-1)**2 + (4-x[1])**3 + (x[2]+7)**2
      end
    end

    it "finds an #x_minimum close to the true solution" do 
      q = [1,5,-7]
      min.x_minimum.each_with_index do |p, i|
        expect(p).to be_within(1e-6).of(q[i])
      end
    end

    it "yields an #f_minimum close to the true solution" do 
      expect(min.f_minimum).to be_within(1e-6).of(-1)
    end
  end

  context "when constrained from below and above" do
    let(:min) do
      MixedModels::NelderMead.minimize(start_point: [0,0,0], upper_bound: [5,5,5], 
                                       lower_bound: [-5,-5,-5], epsilon: 1e-12) do |x| 
        (x[0]-1)**2 + (4-x[1])**3 + (x[2]+7)**2
      end
    end

    it "finds an #x_minimum close to the true solution" do 
      q = [1,5,-5]
      min.x_minimum.each_with_index do |p, i|
        expect(p).to be_within(1e-6).of(q[i])
      end
    end

    it "yields an #f_minimum close to the true solution" do 
      expect(min.f_minimum).to be_within(1e-6).of(3)
    end
  end
end
