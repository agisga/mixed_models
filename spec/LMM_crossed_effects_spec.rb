require 'mixed_models'

describe LMM do
  ["#from_formula", "#from_daru"].each do |constructor_method|
    describe constructor_method do
      context "with multiple crossed random effects" do
        
        let(:df) { Daru::DataFrame.from_csv("spec/data/crossed_effects_data.csv") }

        subject(:model_fit) do
          case constructor_method
          when "#from_formula"
            LMM.from_formula(formula: "y ~ x + (x|g) + (x|h)", data: df, reml: false)
          when "#from_daru"
            LMM.from_daru(response: :y, fixed_effects: [:intercept, :x], 
                          random_effects: [[:intercept, :x], [:intercept, :x]], 
                          grouping: [:g, :h], data: df, reml: false)
          end
        end

        # Results from R:
        #
        #  > mod = lmer(y ~ x + (x|g) + (x|h), data = df, REML=FALSE)
        #  > summary(mod)
        #  Linear mixed model fit by maximum likelihood  ['lmerMod']
        #  Formula: y ~ x + (x | g) + (x | h)
        #     Data: df
        #
        #       AIC      BIC   logLik deviance df.resid 
        #     168.6    192.0    -75.3    150.6       91 
        #
        #  Scaled residuals: 
        #      Min      1Q  Median      3Q     Max 
        #  -2.1486 -0.7006  0.1248  0.6801  1.9042 
        #
        #  Random effects:
        #   Groups   Name        Variance Std.Dev. Corr
        #   g        (Intercept) 0.5680   0.7537       
        #            x           0.5606   0.7488   1.00
        #   h        (Intercept) 0.3179   0.5638       
        #            x           0.1485   0.3854   1.00
        #   Residual             0.2047   0.4525       
        #  Number of obs: 100, groups:  g, 3; h, 3
        #
        #  Fixed effects:
        #              Estimate Std. Error t value
        #  (Intercept)   1.2562     0.5453   2.304
        #  x             1.2742     0.4889   2.606
        #
        #  Correlation of Fixed Effects:
        #    (Intr)
        #  x 0.977 
        #  > mod1 = lmer(y ~ 1 + (x|g) + (x|h), data = df, REML=FALSE)
        #  > mod2 = lmer(y ~ x + (1|g) + (x|h), data = df, REML=FALSE)
        #  > anova(mod1, mod, test = "Chisq")
        #  Data: df
        #  Models:
        #  mod1: y ~ 1 + (x | g) + (x | h)
        #  mod: y ~ x + (x | g) + (x | h)
        #       Df    AIC    BIC  logLik deviance  Chisq Chi Df Pr(>Chisq)  
        #  mod1  8 170.46 191.30 -77.228   154.46                           
        #  mod   9 168.57 192.02 -75.285   150.57 3.8852      1    0.04871 *
        #  ---
        #  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
        #  > anova(mod2, mod, test = "Chisq")
        #  Data: df
        #  Models:
        #  mod2: y ~ x + (1 | g) + (x | h)
        #  mod: y ~ x + (x | g) + (x | h)
        #       Df    AIC    BIC   logLik deviance  Chisq Chi Df Pr(>Chisq)    
        #  mod2  7 277.94 296.18 -131.972   263.94                             
        #  mod   9 168.57 192.02  -75.285   150.57 113.37      2  < 2.2e-16 ***
        #  ---
        #  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

        it "computes the deviance correctly" do
          expect(model_fit.deviance).to be_within(1e-1).of(150.6)
        end

        it "computes the fixed effect estimates correctly" do
          expect(model_fit.fix_ef[:intercept]).to be_within(1e-4).of(1.2562)
          expect(model_fit.fix_ef[:x]).to be_within(1e-4).of(1.2742)
        end

        it "estimates the random effects correlation structures correctly" do
          result = model_fit.ran_ef_summary
          expect(result[:g][:g]).to be_within(1e-3).of(0.7537)
          expect(result[:g_x][:g_x]).to be_within(1e-3).of(0.7488)
          expect(result[:g][:g_x]).to be_within(1e-3).of(1.00)
          expect(result[:g_x][:g]).to be_within(1e-3).of(1.00)
          expect(result[:h][:h]).to be_within(1e-3).of(0.5638)  
          expect(result[:h_x][:h_x]).to be_within(1e-3).of(0.3854)
          expect(result[:h][:h_x]).to be_within(1e-3).of(1.00)       
          expect(result[:h_x][:h]).to be_within(1e-3).of(1.00)
        end

        it "estimates the residual standard deviantion correctly" do
          expect(model_fit.sigma).to be_within(1e-4).of(0.4525)
        end

        it "names the fixed effects correctly" do
          expect(model_fit.fix_ef_names).to eq([:intercept, :x])
        end

        it "names the random effects correctly" do
          names = [:intercept_1, :x_1,
                   :intercept_2, :x_2,
                   :intercept_3, :x_3,
                   :intercept_1, :x_1,
                   :intercept_2, :x_2,
                   :intercept_3, :x_3]
          expect(model_fit.ran_ef_names).to eq(names)
        end

        describe "#fix_ef_p" do
          context "with method: :lrt" do
            it "returns the correct p-value" do
              p = model_fit.fix_ef_p(variable: :x, method: :lrt)
              expect(p).to be_within(1e-4).of(0.04871)
            end
          end
        end

        describe "#ran_ef_p" do
          context "with method: :lrt" do
            it "returns the correct p-value" do
              p = model_fit.ran_ef_p(variable: :x, grouping: :g, method: :lrt)
              expect(p).to be_within(1e-15).of(0.0)
            end
          end
        end
      end
    end
  end
end

