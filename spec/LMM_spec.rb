require 'mixed_models'

RSpec.describe "LMM#from_formula" do
  context "with numeric and categorical fixed effects and numeric random effects" do
    let(:model_fit) do
      LMM.from_formula(formula: "Aggression ~ Age + Species + (Age | Location)", 
                       data: Daru::DataFrame.from_csv("spec/data/alien_species.csv"))
    end

    # compare the obtained estimates to the ones obtained by the function lmer
    # from the package lme4 in R:
    #
    #  > mod <- lmer("Aggression~Age+Species+(Age|Location)", data=alien.species)
    #  > REMLcrit(mod)
    #  [1] 333.7155
    #  > fixef(mod)
    #          (Intercept)                 Age        SpeciesHuman          SpeciesOod SpeciesWeepingAngel 
    #        1016.28672021         -0.06531615       -499.69369521       -899.56932076       -199.58895813 
    #  > ranef(mod)
    #  $Location
    #            (Intercept)         Age
    #  Asylum     -116.68080 -0.03353392
    #  Earth        83.86571 -0.13613995
    #  OodSphere    32.81509  0.16967387
    #  > sigma(mod)
    #  [1] 0.9745324

    it "finds the minimal REML deviance correctly" do
      expect(model_fit.dev_optimal).to be_within(1e-4).of(333.7155)
    end

    it "estimates the residual standard deviation correctly" do
      expect(model_fit.sigma).to be_within(1e-4).of(0.9745)
    end

    it "estimates the fixed effects terms correctly" do
      fix_ef_from_R = [1016.2867, -0.06532, -499.6937, -899.5693, -199.5890] 
      model_fit.fix_ef.values.each_with_index do |e, i|
        expect(e).to be_within(1e-4).of(fix_ef_from_R[i])
      end
    end

    it "estimates the random effects terms correctly" do
      ran_ef_from_R = [-116.6808, -0.0335, 83.8657, -0.1361, 32.8151, 0.1697] 
      model_fit.ran_ef.values.each_with_index do |e, i|
        expect(e).to be_within(1e-4).of(ran_ef_from_R[i])
      end
    end
  end
end
