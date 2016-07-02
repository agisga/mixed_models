require 'mixed_models'

describe LMM do
  ["#from_formula", "#from_daru"].each do |constructor_method|
    describe constructor_method do
      describe "on categorical data" do
        context "with categorical fixed and random effects" do

          let(:df) { Daru::DataFrame.from_csv("spec/data/categorical_data.csv", headers: true) }

          subject(:model_fit) do
            case constructor_method
            when "#from_formula"
              LMM.from_formula(formula: "y ~ x + (x | g)", epsilon: 1e-8, data: df)
            when "#from_daru"
              LMM.from_daru(response: :y, fixed_effects: [:intercept, :x], 
                            random_effects: [[:intercept, :x]], grouping: [:g], 
                            epsilon: 1e-8, data: df)
            end
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
          #  > mod <- lmer(y~x+(x|g), df, REML=FALSE)
          #  > mod1 <- lmer(y~x+(1|g), df, REML=FALSE)
          #  > anova(mod, mod1, test="Chisq")
          #  Data: df
          #  Models:
          #  mod1: y ~ x + (1 | g)
          #  mod: y ~ x + (x | g)
          #       Df    AIC    BIC  logLik deviance  Chisq Chi Df Pr(>Chisq)  
          #  mod1  5 308.03 321.06 -149.01   298.03                           
          #  mod  10 303.05 329.10 -141.52   283.05 14.982      5    0.01044 *
          #  ---
          #  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
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

          describe "#ran_ef_p" do
            context "with method: :lrt" do
              it "returns the correct p-value" do
                no_reml_model = case constructor_method
                                when "#from_formula"
                                  LMM.from_formula(formula: "y ~ x + (x | g)", epsilon: 1e-8, 
                                                   reml: false, data: df)
                                when "#from_daru"
                                  LMM.from_daru(response: :y, fixed_effects: [:intercept, :x], 
                                                random_effects: [[:intercept, :x]], grouping: [:g], 
                                                epsilon: 1e-8, reml: false, data: df)
                                end
                p = no_reml_model.ran_ef_p(variable: :x, grouping: :g, method: :lrt)
                expect(p).to be_within(0.005).of(0.01044)
              end
            end
          end
        end

        context "with categorical fixed and random effects and exclusion of random intercept" do

          let(:df) { Daru::DataFrame.from_csv("spec/data/categorical_data.csv", headers: true) }

          subject(:model_fit) do
            case constructor_method
            when "#from_formula"
              LMM.from_formula(formula: "y ~ x + (0 + x | g)", epsilon: 1e-8, data: df)
            when "#from_daru"
              LMM.from_daru(response: :y, fixed_effects: [:intercept, :x],
                            random_effects: [[:no_intercept, :x]], grouping: [:g], 
                            epsilon: 1e-8, data: df)
            end
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
          #  > mod = lmer(y~x + (0+x|g), df, REML=FALSE)
          #  > mod1 = lmer(y~1 + (0+x|g), df, REML=FALSE)
          #  > anova(mod, mod1, test="Chisq")
          #  Data: df
          #  Models:
          #  mod1: y ~ 1 + (0 + x | g)
          #  mod: y ~ x + (0 + x | g)
          #       Df    AIC    BIC  logLik deviance Chisq Chi Df Pr(>Chisq)   
          #  mod1  8 309.32 330.16 -146.66   293.32                           
          #  mod  10 303.05 329.10 -141.52   283.05 10.27      2   0.005888 **
          #  ---
          #  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
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

          let(:no_reml_model) do
            case constructor_method
            when "#from_formula"
              LMM.from_formula(formula: "y ~ x + (0+x | g)", epsilon: 1e-8, 
                               reml: false, data: df)
            when "#from_daru"
              LMM.from_daru(response: :y, fixed_effects: [:intercept, :x], 
                            random_effects: [[:no_intercept, :x]], grouping: [:g], 
                            epsilon: 1e-8, reml: false, data: df)
            end
          end

          describe "#fix_ef_p" do
            context "with method: :lrt" do
              it "returns the correct p-value" do
                p = no_reml_model.fix_ef_p(variable: :x, method: :lrt)
                expect(p).to be_within(1e-4).of(0.005888)
              end
            end
          end

          describe "#ran_ef_p" do
            context "with method: :lrt" do
              it "raises" do
                expect{no_reml_model.ran_ef_p(variable: :x, grouping: :g, method: :lrt)}.to raise_error(NoMethodError)
              end
            end
          end
        end

        context "with categorical fixed and random effects and exclusion of fixed intercept" do

          let(:df) { Daru::DataFrame.from_csv("spec/data/categorical_data.csv", headers: true) }

          subject(:model_fit) do
            case constructor_method
            when "#from_formula"
              LMM.from_formula(formula: "y ~ 0 + x + (x | g)", epsilon: 1e-8, data: df)
            when "#from_daru"
              LMM.from_daru(response: :y, fixed_effects: [:no_intercept, :x],
                            random_effects: [[:intercept, :x]], grouping: [:g], 
                            epsilon: 1e-8, data: df)
            end
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
          #
          #  > mod = lmer(y~0+x + (x|g), df, REML=FALSE)
          #  > mod1 = lmer(y~0+x + (1|g), df, REML=FALSE)
          #  > anova(mod, mod1, test="Chisq")
          #  Data: df
          #  Models:
          #  mod1: y ~ 0 + x + (1 | g)
          #  mod: y ~ 0 + x + (x | g)
          #       Df    AIC    BIC  logLik deviance  Chisq Chi Df Pr(>Chisq)  
          #  mod1  5 308.03 321.06 -149.01   298.03                           
          #  mod  10 303.05 329.10 -141.52   283.05 14.982      5    0.01044 *
          #  ---
          #  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

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

          let(:no_reml_model) do
            case constructor_method
            when "#from_formula"
              LMM.from_formula(formula: "y ~ 0 + x + (x | g)", epsilon: 1e-8, 
                               reml: false, data: df)
            when "#from_daru"
              LMM.from_daru(response: :y, fixed_effects: [:no_intercept, :x], 
                            random_effects: [[:intercept, :x]], grouping: [:g], 
                            epsilon: 1e-8, reml: false, data: df)
            end
          end

          describe "#ran_ef_p" do
            context "with method: :lrt" do
              it "returns the correct p-value" do
                p = no_reml_model.ran_ef_p(variable: :x, grouping: :g, method: :lrt)
                expect(p).to be_within(0.005).of(0.01044)
              end
            end
          end

          describe "#fix_ef_p" do
            context "with method: :lrt" do
              it "raises" do
                expect{no_reml_model.fix_ef_p(variable: :x, method: :lrt)}.to raise_error(IndexError)
              end
            end
          end
        end
      end
    end
  end
end
