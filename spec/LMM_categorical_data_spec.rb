require 'mixed_models'

describe LMM do
  describe "Performance on categorical data" do
    describe "#from_formula" do
      context "with categorical fixed and random effects" do
        subject(:model_fit) do
          LMM.from_formula(formula: "y ~ x + (x | g)", epsilon: 1e-8, 
                           data: Daru::DataFrame.from_csv("spec/data/categorical_data.csv"))
        end

        # compare the obtained estimates to the ones obtained for the same data 
        # by the function lmer from the package lme4 in R:
        #
        #  > mod <- lmer(y~x+(x|g), df)
        #  > REMLcrit(mod)
        #  [1] 285.3409
        #  > sigma(mod)
        #  [1] 0.9814615
        #  > fixef(mod)
        #  (Intercept)          xB          xC 
        #     2.441537    1.207339   -1.805640 
        #  > ranef(mod)
        #  $g
        #     (Intercept)        xB         xC
        #  g1   0.3285087 -1.026759 -0.4829453
        #  g2  -0.3285087  1.026759  0.4829453
        #

        it "finds the minimal REML deviance correctly" do
          expect(model_fit.deviance).to be_within(1e-4).of(285.3409)
        end

        it "estimates the residual standard deviation correctly" do
          expect(model_fit.sigma).to be_within(1e-4).of(0.9815)
        end

        it "estimates the fixed effects terms correctly" do
          fix_ef_from_R = [2.4415, 1.2073, -1.8056] 
          model_fit.fix_ef.values.each_with_index do |e, i|
            expect(e).to be_within(1e-4).of(fix_ef_from_R[i])
          end
        end

        it "estimates the random effects terms correctly" do
          ran_ef_from_R = [0.3285, -1.0268, -0.4829, -0.3285, 1.0268, 0.4829] 
          model_fit.ran_ef.values.each_with_index do |e, i|
            expect(e).to be_within(1e-4).of(ran_ef_from_R[i])
          end
        end

        it "names the fixed effects correctly" do
          fix_ef_names = [:intercept, :x_lvl_B, :x_lvl_C]
          expect(model_fit.fix_ef.keys).to eq(fix_ef_names)
        end

        it "names the random effects correctly" do
          ran_ef_names = [:intercept_g1, :x_lvl_B_g1, :x_lvl_C_g1, 
                          :intercept_g2, :x_lvl_B_g2, :x_lvl_C_g2]
          expect(model_fit.ran_ef.keys).to eq(ran_ef_names)
        end
      end

      context "with categorical fixed and random effects and exclusion of random intercept" do
        subject(:model_fit) do
          LMM.from_formula(formula: "y ~ x + (0 + x | g)", epsilon: 1e-8, 
                           data: Daru::DataFrame.from_csv("spec/data/categorical_data.csv"))
        end

        # compare the obtained estimates to the ones obtained for the same data 
        # by the function lmer from the package lme4 in R:
        #
        #  > mod <- lmer(y~x+(0+x|g), df)
        #  > ranef(mod)
        #  $g
        #             xA       xB         xC
        #  g1  0.3285088 -0.69825 -0.1544366
        #  g2 -0.3285088  0.69825  0.1544366
        #  > fixef(mod)
        #  (Intercept)          xB          xC 
        #     2.441537    1.207339   -1.805640 
        #  > REMLcrit(mod)
        #  [1] 285.3409
        #  > sigma(mod)
        #  [1] 0.9814615
        #

        it "finds the minimal REML deviance correctly" do
          expect(model_fit.deviance).to be_within(1e-4).of(285.3409)
        end

        it "estimates the residual standard deviation correctly" do
          expect(model_fit.sigma).to be_within(1e-4).of(0.9815)
        end

        it "estimates the fixed effects terms correctly" do
          fix_ef_from_R = [2.4415, 1.2073, -1.8056] 
          model_fit.fix_ef.values.each_with_index do |e, i|
            expect(e).to be_within(1e-4).of(fix_ef_from_R[i])
          end
        end

        it "estimates the random effects terms correctly" do
          ran_ef_from_R = [0.3285, -0.6983, -0.1544,-0.3285, 0.6983, 0.1544] 
          model_fit.ran_ef.values.each_with_index do |e, i|
            expect(e).to be_within(1e-4).of(ran_ef_from_R[i])
          end
        end

        it "names the fixed effects correctly" do
          fix_ef_names = [:intercept, :x_lvl_B, :x_lvl_C]
          expect(model_fit.fix_ef.keys).to eq(fix_ef_names)
          expect(model_fit.fix_ef_names).to eq(fix_ef_names)
        end

        it "names the random effects correctly" do
          ran_ef_names = [:x_lvl_A_g1, :x_lvl_B_g1, :x_lvl_C_g1, 
                          :x_lvl_A_g2, :x_lvl_B_g2, :x_lvl_C_g2]
          expect(model_fit.ran_ef.keys).to eq(ran_ef_names)
          expect(model_fit.ran_ef_names).to eq(ran_ef_names)
        end
      end

      context "with categorical fixed and random effects and exclusion of fixed intercept" do
        subject(:model_fit) do
          LMM.from_formula(formula: "y ~ 0 + x + (x | g)", epsilon: 1e-8, 
                           data: Daru::DataFrame.from_csv("spec/data/categorical_data.csv"))
        end

        # compare the obtained estimates to the ones obtained for the same data 
        # by the function lmer from the package lme4 in R:
        #
        #  > mod <- lmer(y~0+x+(x|g), df)
        #  > REMLcrit(mod)
        #  [1] 285.3409
        #  > sigma(mod)
        #  [1] 0.9814615
        #  > fixef(mod)
        #         xA        xB        xC 
        #  2.4415370 3.6488761 0.6358974 
        #  > ranef(mod)
        #  $g
        #     (Intercept)        xB         xC
        #  g1   0.3285088 -1.026759 -0.4829455
        #  g2  -0.3285088  1.026759  0.4829455

        it "finds the minimal REML deviance correctly" do
          expect(model_fit.deviance).to be_within(1e-4).of(285.3409)
        end

        it "estimates the residual standard deviation correctly" do
          expect(model_fit.sigma).to be_within(1e-4).of(0.9815)
        end

        it "estimates the fixed effects terms correctly" do
          fix_ef_from_R = [2.4415, 3.6489, 0.6359] 
          model_fit.fix_ef.values.each_with_index do |e, i|
            expect(e).to be_within(1e-4).of(fix_ef_from_R[i])
          end
        end

        it "estimates the random effects terms correctly" do
          ran_ef_from_R = [0.3285, -1.0268, -0.4829, -0.3285, 1.0268, 0.4829] 
          model_fit.ran_ef.values.each_with_index do |e, i|
            expect(e).to be_within(1e-4).of(ran_ef_from_R[i])
          end
        end

        it "names the fixed effects correctly" do
          fix_ef_names = [:x_lvl_A, :x_lvl_B, :x_lvl_C]
          expect(model_fit.fix_ef.keys).to eq(fix_ef_names)
          expect(model_fit.fix_ef_names).to eq(fix_ef_names)
        end

        it "names the random effects correctly" do
          ran_ef_names = [:intercept_g1, :x_lvl_B_g1, :x_lvl_C_g1, 
                          :intercept_g2, :x_lvl_B_g2, :x_lvl_C_g2]
          expect(model_fit.ran_ef.keys).to eq(ran_ef_names)
          expect(model_fit.ran_ef_names).to eq(ran_ef_names)
        end
      end
    end

    describe "#from_daru" do
      context "with categorical fixed and random effects" do
        subject(:model_fit) do
          LMM.from_daru(response: :y, fixed_effects: [:intercept, :x],
                        random_effects: [[:intercept, :x]], grouping: [:g], epsilon: 1e-8, 
                        data: Daru::DataFrame.from_csv("spec/data/categorical_data.csv"))
        end

        # compare the obtained estimates to the ones obtained for the same data 
        # Same as using #from_formula

        it "finds the minimal REML deviance correctly" do
          expect(model_fit.deviance).to be_within(1e-4).of(285.3409)
        end

        it "estimates the residual standard deviation correctly" do
          expect(model_fit.sigma).to be_within(1e-4).of(0.9815)
        end

        it "estimates the fixed effects terms correctly" do
          fix_ef_from_R = [2.4415, 1.2073, -1.8056] 
          model_fit.fix_ef.values.each_with_index do |e, i|
            expect(e).to be_within(1e-4).of(fix_ef_from_R[i])
          end
        end

        it "estimates the random effects terms correctly" do
          ran_ef_from_R = [0.3285, -1.0268, -0.4829, -0.3285, 1.0268, 0.4829] 
          model_fit.ran_ef.values.each_with_index do |e, i|
            expect(e).to be_within(1e-4).of(ran_ef_from_R[i])
          end
        end

        it "names the fixed effects correctly" do
          fix_ef_names = [:intercept, :x_lvl_B, :x_lvl_C]
          expect(model_fit.fix_ef.keys).to eq(fix_ef_names)
        end

        it "names the random effects correctly" do
          ran_ef_names = [:intercept_g1, :x_lvl_B_g1, :x_lvl_C_g1, 
                          :intercept_g2, :x_lvl_B_g2, :x_lvl_C_g2]
          expect(model_fit.ran_ef.keys).to eq(ran_ef_names)
        end
      end

      context "with categorical fixed and random effects and exclusion of random intercept" do
        subject(:model_fit) do
          LMM.from_daru(response: :y, fixed_effects: [:intercept, :x],
                        random_effects: [[:no_intercept, :x]], grouping: [:g], epsilon: 1e-8, 
                        data: Daru::DataFrame.from_csv("spec/data/categorical_data.csv"))
        end

        # compare the obtained estimates to the ones obtained for the same data 
        # Same as using #from_formula

        it "finds the minimal REML deviance correctly" do
          expect(model_fit.deviance).to be_within(1e-4).of(285.3409)
        end

        it "estimates the residual standard deviation correctly" do
          expect(model_fit.sigma).to be_within(1e-4).of(0.9815)
        end

        it "estimates the fixed effects terms correctly" do
          fix_ef_from_R = [2.4415, 1.2073, -1.8056] 
          model_fit.fix_ef.values.each_with_index do |e, i|
            expect(e).to be_within(1e-4).of(fix_ef_from_R[i])
          end
        end

        it "estimates the random effects terms correctly" do
          ran_ef_from_R = [0.3285, -0.6983, -0.1544,-0.3285, 0.6983, 0.1544] 
          model_fit.ran_ef.values.each_with_index do |e, i|
            expect(e).to be_within(1e-4).of(ran_ef_from_R[i])
          end
        end

        it "names the fixed effects correctly" do
          fix_ef_names = [:intercept, :x_lvl_B, :x_lvl_C]
          expect(model_fit.fix_ef.keys).to eq(fix_ef_names)
          expect(model_fit.fix_ef_names).to eq(fix_ef_names)
        end

        it "names the random effects correctly" do
          ran_ef_names = [:x_lvl_A_g1, :x_lvl_B_g1, :x_lvl_C_g1, 
                          :x_lvl_A_g2, :x_lvl_B_g2, :x_lvl_C_g2]
          expect(model_fit.ran_ef.keys).to eq(ran_ef_names)
          expect(model_fit.ran_ef_names).to eq(ran_ef_names)
        end
      end

      context "with categorical fixed and random effects and exclusion of fixed intercept" do
        subject(:model_fit) do
          LMM.from_daru(response: :y, fixed_effects: [:no_intercept, :x],
                        random_effects: [[:intercept, :x]], grouping: [:g], epsilon: 1e-8, 
                        data: Daru::DataFrame.from_csv("spec/data/categorical_data.csv"))
        end

        # compare the obtained estimates to the ones obtained for the same data 
        # Same as using #from_formula

        it "finds the minimal REML deviance correctly" do
          expect(model_fit.deviance).to be_within(1e-4).of(285.3409)
        end

        it "estimates the residual standard deviation correctly" do
          expect(model_fit.sigma).to be_within(1e-4).of(0.9815)
        end

        it "estimates the fixed effects terms correctly" do
          fix_ef_from_R = [2.4415, 3.6489, 0.6359] 
          model_fit.fix_ef.values.each_with_index do |e, i|
            expect(e).to be_within(1e-4).of(fix_ef_from_R[i])
          end
        end

        it "estimates the random effects terms correctly" do
          ran_ef_from_R = [0.3285, -1.0268, -0.4829, -0.3285, 1.0268, 0.4829] 
          model_fit.ran_ef.values.each_with_index do |e, i|
            expect(e).to be_within(1e-4).of(ran_ef_from_R[i])
          end
        end

        it "names the fixed effects correctly" do
          fix_ef_names = [:x_lvl_A, :x_lvl_B, :x_lvl_C]
          expect(model_fit.fix_ef.keys).to eq(fix_ef_names)
          expect(model_fit.fix_ef_names).to eq(fix_ef_names)
        end

        it "names the random effects correctly" do
          ran_ef_names = [:intercept_g1, :x_lvl_B_g1, :x_lvl_C_g1, 
                          :intercept_g2, :x_lvl_B_g2, :x_lvl_C_g2]
          expect(model_fit.ran_ef.keys).to eq(ran_ef_names)
          expect(model_fit.ran_ef_names).to eq(ran_ef_names)
        end
      end
    end
  end
end
