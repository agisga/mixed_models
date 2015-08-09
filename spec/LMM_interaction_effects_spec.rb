require 'mixed_models'

describe LMM do
  ["#from_formula", "#from_daru"].each do |constructor_method|
    describe constructor_method do
      context "with interaction effects" do
        context "between two numeric variables" do
          
          let(:df) { Daru::DataFrame.from_csv("spec/data/numeric_x_numeric_interaction.csv") }

          case constructor_method
          when "#from_formula"
            subject(:model_fit) { LMM.from_formula(formula: "y ~ a + b + a:b + (0 + a:b | gr)", data: df) }
          when "#from_daru"
            subject(:model_fit) do
              LMM.from_daru(response: :y, fixed_effects: [:intercept, :a, :b, [:a, :b]], 
                            random_effects: [[:no_intercept, [:a, :b]]], grouping: [:gr], data: df)
            end
          end

          # result from R for comparison:
          #  > mod <- lmer(y~a+b+a:b+(0+a:b|gr), data=df)
          #  > summary(mod)
          #  Linear mixed model fit by REML ['lmerMod']
          #  Formula: y ~ a + b + a:b + (0 + a:b | gr)
          #     Data: df
          #
          #  REML criterion at convergence: 312.3
          #
          #  Scaled residuals: 
          #       Min       1Q   Median       3Q      Max 
          #  -2.76624 -0.68003 -0.07408  0.62803  2.06279 
          #
          #  Random effects:
          #   Groups   Name Variance Std.Dev.
          #   gr       a:b  0.5451   0.7383  
          #   Residual      1.1298   1.0629  
          #  Number of obs: 100, groups:  gr, 5
          #
          #  Fixed effects:
          #              Estimate Std. Error t value
          #  (Intercept)  0.02967    0.10830   0.274
          #  a            1.08225    0.10691  10.123
          #  b            0.96928    0.10242   9.464
          #  a:b          1.25433    0.34734   3.611
          #
          #  Correlation of Fixed Effects:
          #      (Intr) a      b     
          #  a    0.043              
          #  b    0.030 -0.050       
          #  a:b -0.013 -0.022 -0.027
          
          it "finds the minimal REML deviance correctly" do
            expect(model_fit.deviance).to be_within(1e-1).of(312.3)
          end

          it "estimates the residual standard deviation correctly" do
            expect(model_fit.sigma).to be_within(1e-4).of(1.0629)
          end

          it "estimates the fixed effects terms correctly" do
            fix_ef_from_R = [0.02967, 1.08225, 0.96928, 1.25433] 
            model_fit.fix_ef.values.each_with_index do |e, i|
              expect(e).to be_within(1e-4).of(fix_ef_from_R[i])
            end
          end

          it "names the fixed effects correctly" do
            fix_ef_names = [:intercept, :a, :b, :a_interaction_with_b]
            expect(model_fit.fix_ef.keys).to eq(fix_ef_names)
            expect(model_fit.fix_ef_names).to eq(fix_ef_names)
          end

          it "names the random effects correctly" do
            ran_ef_names = [:a_interaction_with_b_1, :a_interaction_with_b_2, 
                            :a_interaction_with_b_3, :a_interaction_with_b_4, 
                            :a_interaction_with_b_5]
            expect(model_fit.ran_ef.keys).to eq(ran_ef_names)
            expect(model_fit.ran_ef_names).to eq(ran_ef_names)
          end

          it "has no side effects on the input parameters" do
            reml = true
            df_unaltered = Daru::DataFrame.from_csv("spec/data/numeric_x_numeric_interaction.csv")
            weights = Array.new(df_unaltered.nrows) { 1.0 }
            offset = 0.0

            case constructor_method
            when "#from_formula"
              form = "y ~ a + b + a:b + (0 + a:b | gr)"
              LMM.from_formula(formula: form, reml: true, weights: weights, offset: offset, data: df_unaltered)
              expect(form).to eq("y ~ a + b + a:b + (0 + a:b | gr)")
            when "#from_daru"
              resp = :y
              fe = [:intercept, :a, :b, [:a, :b]]
              re = [[:no_intercept, [:a, :b]]]
              gr = [:gr]
              LMM.from_daru(response: resp, fixed_effects: fe, random_effects: re, grouping: gr,
                            reml: reml, weights: weights, offset: offset, data: df_unaltered) 
              expect(resp).to eq(:y)
              expect(fe).to eq([:intercept, :a, :b, [:a, :b]])
              expect(re).to eq([[:no_intercept, [:a, :b]]])
              expect(gr).to eq([:gr])
            end

            expect(reml).to eq(true)
            expect(Daru::DataFrame.from_csv("spec/data/numeric_x_numeric_interaction.csv")).to eq(df_unaltered)
            expect(Array.new(df_unaltered.nrows) { 1.0 }).to eq(weights)
            expect(offset).to eq(0.0)
          end
        end

        context "between a numeric and a categorical variable" do

          let(:df) { Daru::DataFrame.from_csv("spec/data/numeric_x_categorical_interaction.csv") }

          case constructor_method
          when "#from_formula"
            subject(:model_fit) { LMM.from_formula(formula: "y ~ num + cat + num:cat + (0 + num:cat | gr)", data: df) }
          when "#from_daru"
            subject(:model_fit) do
              LMM.from_daru(response: :y, fixed_effects: [:intercept, :num, :cat, [:num, :cat]],
                            random_effects: [[:no_intercept, [:num, :cat]]], grouping: [:gr], data: df)
            end
          end

          # Result from R for comparison:
          #  > mod <- lmer(y~num*cat+(0+num:cat|gr), data=df)
          #
          #  > fixef(mod)
          #  (Intercept)         num        catB        catC    num:catB    num:catC 
          #    2.1121836   2.5502758   0.8093798   2.0581310  -0.8488252  -0.7940961 
          #  > ranef(mod)
          #  $gr
          #            num:catA   num:catB    num:catC
          #  case     0.3051041 -0.3758435 -0.04775093
          #  control -0.3051041  0.3758435  0.04775093
          #  > REMLcrit(mod)
          #  [1] 286.3773
          #  > sigma(mod)
          #  [1] 0.9814441
          
          it "finds the minimal REML deviance correctly" do
            expect(model_fit.deviance).to be_within(1e-2).of(286.3773)
          end

          it "estimates the residual standard deviation correctly" do
            expect(model_fit.sigma).to be_within(1e-2).of(0.9814441)
          end

          it "estimates the fixed effects terms correctly" do
            fix_ef_from_R = [2.1121836, 2.5502758, 0.8093798, 2.0581310, -0.8488252, -0.7940961] 
            model_fit.fix_ef.values.each_with_index do |e, i|
              expect(e).to be_within(1e-2).of(fix_ef_from_R[i])
            end
          end
          
          it "estimates the random effects terms correctly" do
            ran_ef_from_R = [0.3051041,-0.3758435,-0.04775093,-0.3051041, 0.3758435, 0.04775093]
            model_fit.ran_ef.values.each_with_index do |e, i|
              expect(e).to be_within(1e-2).of(ran_ef_from_R[i])
            end
          end

          it "names the fixed effects correctly" do
            fix_ef_names = [:intercept, :num, :cat_lvl_B, :cat_lvl_C,
                            :num_interaction_with_cat_lvl_B, :num_interaction_with_cat_lvl_C]
            expect(model_fit.fix_ef.keys).to eq(fix_ef_names)
            expect(model_fit.fix_ef_names).to eq(fix_ef_names)
          end

          it "names the random effects correctly" do
            ran_ef_names = [:num_interaction_with_cat_lvl_A_case, :num_interaction_with_cat_lvl_B_case, 
                            :num_interaction_with_cat_lvl_C_case, :num_interaction_with_cat_lvl_A_control, 
                            :num_interaction_with_cat_lvl_B_control, :num_interaction_with_cat_lvl_C_control]
            expect(model_fit.ran_ef.keys).to eq(ran_ef_names)
            expect(model_fit.ran_ef_names).to eq(ran_ef_names)
          end
        end

        context "between a categorical and a numeric variable" do

          let(:df) { Daru::DataFrame.from_csv("spec/data/numeric_x_categorical_interaction.csv") }

          case constructor_method
          when "#from_formula"
            subject(:model_fit) { LMM.from_formula(formula: "y ~ cat + num + cat:num + (0 + cat:num | gr)", data: df) }
          when "#from_daru"
            subject(:model_fit) do
              LMM.from_daru(response: :y, fixed_effects: [:intercept, :cat, :num, [:cat, :num]],
                            random_effects: [[:no_intercept, [:cat, :num]]], grouping: [:gr], data: df)
            end
          end

          # Result from R for comparison:
          # same as previous example
          
          it "finds the minimal REML deviance correctly" do
            expect(model_fit.deviance).to be_within(1e-2).of(286.3773)
          end

          it "estimates the residual standard deviation correctly" do
            expect(model_fit.sigma).to be_within(1e-2).of(0.9814441)
          end

          it "estimates the fixed effects terms correctly" do
            fix_ef_from_R = [2.1121836, 0.8093798, 2.0581310, 2.5502758, -0.8488252, -0.7940961] 
            model_fit.fix_ef.values.each_with_index do |e, i|
              expect(e).to be_within(1e-2).of(fix_ef_from_R[i])
            end
          end
          
          it "estimates the random effects terms correctly" do
            ran_ef_from_R = [0.3051041,-0.3758435,-0.04775093,-0.3051041, 0.3758435, 0.04775093]
            model_fit.ran_ef.values.each_with_index do |e, i|
              expect(e).to be_within(1e-2).of(ran_ef_from_R[i])
            end
          end

          it "names the fixed effects correctly" do
            fix_ef_names = [:intercept, :cat_lvl_B, :cat_lvl_C, :num,
                            :num_interaction_with_cat_lvl_B, :num_interaction_with_cat_lvl_C]
            expect(model_fit.fix_ef.keys).to eq(fix_ef_names)
            expect(model_fit.fix_ef_names).to eq(fix_ef_names)
          end

          it "names the random effects correctly" do
            ran_ef_names = [:num_interaction_with_cat_lvl_A_case, :num_interaction_with_cat_lvl_B_case, 
                            :num_interaction_with_cat_lvl_C_case, :num_interaction_with_cat_lvl_A_control, 
                            :num_interaction_with_cat_lvl_B_control, :num_interaction_with_cat_lvl_C_control]
            expect(model_fit.ran_ef.keys).to eq(ran_ef_names)
            expect(model_fit.ran_ef_names).to eq(ran_ef_names)
          end
        end

        context "between two categorical variables" do

          let(:df) { Daru::DataFrame.from_csv("spec/data/categorical_x_categorical_interaction.csv") }

          context "with both non-interaction fixed effects and fixed intercept, but no random non-interaction effects and without random intercept" do

            case constructor_method
            when "#from_formula"
              subject(:model_fit) { LMM.from_formula(formula: "y ~ f1 + f2 + f1:f2 + (0 + f1:f2 | gr)", data: df) }
            when "#from_daru"
              subject(:model_fit) do
                LMM.from_daru(response: :y, fixed_effects: [:intercept, :f1, :f2, [:f1, :f2]],
                              random_effects: [[:no_intercept, [:f1, :f2]]], grouping: [:gr], data: df)
              end
            end

            # Result from R for comparison:
            #
            #  > mod <- lmer(y ~ f1 + f2 + f1:f2 + (0 + f1:f2 | gr), data=df)
            #  > summary(mod)
            #  Linear mixed model fit by REML ['lmerMod']
            #  Formula: y ~ f1 + f2 + f1:f2 + (0 + f1:f2 | gr)
            #     Data: df
            #
            #  REML criterion at convergence: 273.3
            #
            #  Scaled residuals: 
            #       Min       1Q   Median       3Q      Max 
            #  -2.05784 -0.77037 -0.06954  0.53708  2.28541 
            #
            #  Random effects:
            #   Groups   Name    Variance Std.Dev. Corr                         
            #   gr       f1a:f2u 0.09120  0.3020                                
            #            f1b:f2u 0.44541  0.6674    1.00                        
            #            f1a:f2v 1.74446  1.3208    1.00  1.00                  
            #            f1b:f2v 1.32788  1.1523   -1.00 -1.00 -1.00            
            #            f1a:f2w 0.02028  0.1424   -1.00 -1.00 -1.00  1.00      
            #            f1b:f2w 0.13479  0.3671    1.00  1.00  1.00 -1.00 -1.00
            #   Residual         0.86300  0.9290                                
            #  Number of obs: 100, groups:  gr, 2
            #  
            #  Fixed effects:
            #              Estimate Std. Error t value
            #  (Intercept)   7.7788     0.3347  23.239
            #  f1b          -1.2910     0.4235  -3.048
            #  f2v           1.0533     0.8049   1.309
            #  f2w          -2.3771     0.4616  -5.149
            #  f1b:f2v      -4.1637     2.0603  -2.021
            #  f1b:f2w      -0.4784     0.4780  -1.001
            #
            #  Correlation of Fixed Effects:
            #          (Intr) f1b    f2v    f2w    f1b:f2v
            #  f1b     -0.081                             
            #  f2v      0.324  0.743                      
            #  f2w     -0.864 -0.075 -0.430               
            #  f1b:f2v -0.525 -0.724 -0.950  0.593        
            #  f1b:f2w  0.552 -0.426  0.016 -0.664 -0.093 
            
            it "finds the minimal REML deviance correctly" do
              expect(model_fit.deviance).to be_within(1e-1).of(273.3)
            end

            it "estimates the residual standard deviation correctly" do
              expect(model_fit.sigma).to be_within(1e-2).of(0.9290)
            end

            it "estimates the fixed effects terms correctly" do
              fix_ef_from_R = [7.7788, -1.2910, 1.0533, -2.3771, -4.1637, -0.4784]
              model_fit.fix_ef.values.each_with_index do |e, i|
                expect(e).to be_within(1e-2).of(fix_ef_from_R[i])
              end
            end
            
            it "names the fixed effects correctly" do
              fix_ef_names = [:intercept, :f1_lvl_b, :f2_lvl_v, :f2_lvl_w,
                              :f1_lvl_b_interaction_with_f2_lvl_v, :f1_lvl_b_interaction_with_f2_lvl_w]
              expect(model_fit.fix_ef.keys).to eq(fix_ef_names)
              expect(model_fit.fix_ef_names).to eq(fix_ef_names)
            end

            it "names the random effects correctly" do
              ran_ef_names = [:f1_lvl_a_interaction_with_f2_lvl_u_g1,
                              :f1_lvl_a_interaction_with_f2_lvl_v_g1,
                              :f1_lvl_a_interaction_with_f2_lvl_w_g1,
                              :f1_lvl_b_interaction_with_f2_lvl_u_g1,
                              :f1_lvl_b_interaction_with_f2_lvl_v_g1,
                              :f1_lvl_b_interaction_with_f2_lvl_w_g1,
                              :f1_lvl_a_interaction_with_f2_lvl_u_g2,
                              :f1_lvl_a_interaction_with_f2_lvl_v_g2,
                              :f1_lvl_a_interaction_with_f2_lvl_w_g2,
                              :f1_lvl_b_interaction_with_f2_lvl_u_g2,
                              :f1_lvl_b_interaction_with_f2_lvl_v_g2,
                              :f1_lvl_b_interaction_with_f2_lvl_w_g2]
              expect(model_fit.ran_ef.keys).to eq(ran_ef_names)
              expect(model_fit.ran_ef_names).to eq(ran_ef_names)
            end
          end


          context "without non-interaction fixed effects, with fixed intercept, with one random non-interaction effect and without random intercept" do

            case constructor_method
            when "#from_formula"
              subject(:model_fit) { LMM.from_formula(formula: "y ~ f1:f2 + (0 + f1 + f1:f2 | gr)", data: df) }
            when "#from_daru"
              subject(:model_fit) do
                LMM.from_daru(response: :y, fixed_effects: [:intercept, [:f1, :f2]],
                              random_effects: [[:no_intercept, :f1, [:f1, :f2]]], grouping: [:gr], data: df)
              end
            end

            # Result from R for comparison:
            #
            #  > mod <- lmer(y ~ f1:f2 + (0 + f1 + f1:f2 | gr), data=df)
            #  > summary(mod)
            #    Linear mixed model fit by REML ['lmerMod']
            #    Formula: y ~ f1:f2 + (0 + f1 + f1:f2 | gr)
            #       Data: df
            #
            #    REML criterion at convergence: 273.3
            #
            #    Scaled residuals: 
            #         Min       1Q   Median       3Q      Max 
            #    -2.05784 -0.77037 -0.06954  0.53708  2.28541 
            #
            #    Random effects:
            #     Groups   Name    Variance Std.Dev. Corr                         
            #     gr       f1a     0.09120  0.3020                                
            #              f1b     0.44540  0.6674    1.00                        
            #              f1a:f2v 1.03793  1.0188    1.00  1.00                  
            #              f1b:f2v 3.31139  1.8197   -1.00 -1.00 -1.00            
            #              f1a:f2w 0.19749  0.4444   -1.00 -1.00 -1.00  1.00      
            #              f1b:f2w 0.09015  0.3003   -1.00 -1.00 -1.00  1.00  1.00
            #     Residual         0.86300  0.9290                                
            #    Number of obs: 100, groups:  gr, 2
            #
            #    Fixed effects:
            #                Estimate Std. Error t value
            #    (Intercept)   3.6323     0.3535  10.276
            #    f1a:f2u       4.1465     0.3551  11.678
            #    f1b:f2u       2.8555     0.3854   7.408
            #    f1a:f2v       5.1997     0.7580   6.860
            #    f1b:f2v      -0.2549     1.1194  -0.228
            #    f1a:f2w       1.7693     0.4851   3.648
            #
            #    Correlation of Fixed Effects:
            #            (Intr) f1a:f2 f1b:f2u f1:f2v f1b:f2v
            #    f1a:f2u -0.554                              
            #    f1b:f2u -0.018  0.348                       
            #    f1a:f2v  0.438  0.098  0.689                
            #    f1b:f2v -0.850  0.269 -0.395  -0.786        
            #    f1a:f2w -0.881  0.430 -0.101  -0.504  0.819 
            #
            #    > REMLcrit(mod)
            #    [1] 273.3275

            it "finds the minimal REML deviance correctly" do
              expect(model_fit.deviance).to be_within(1e-1).of(273.3275)
            end

            it "estimates the residual standard deviation correctly" do
              expect(model_fit.sigma).to be_within(1e-2).of(0.9290)
            end

            it "estimates the fixed effects terms correctly" do
              fix_ef_from_R = [3.6323, 4.1465, 5.1997, 1.7693, 2.8555, -0.2549]
              model_fit.fix_ef.values.each_with_index do |e, i|
                expect(e).to be_within(1e-2).of(fix_ef_from_R[i])
              end
            end
            
            it "names the fixed effects correctly" do
              fix_ef_names = [:intercept, :f1_lvl_a_interaction_with_f2_lvl_u, 
                              :f1_lvl_a_interaction_with_f2_lvl_v, :f1_lvl_a_interaction_with_f2_lvl_w,
                              :f1_lvl_b_interaction_with_f2_lvl_u, :f1_lvl_b_interaction_with_f2_lvl_v]
              expect(model_fit.fix_ef.keys).to eq(fix_ef_names)
              expect(model_fit.fix_ef_names).to eq(fix_ef_names)
            end

            it "names the random effects correctly" do
              ran_ef_names = [:f1_lvl_a_g1, :f1_lvl_b_g1,
                              :f1_lvl_a_interaction_with_f2_lvl_v_g1,
                              :f1_lvl_a_interaction_with_f2_lvl_w_g1,
                              :f1_lvl_b_interaction_with_f2_lvl_v_g1,
                              :f1_lvl_b_interaction_with_f2_lvl_w_g1,
                              :f1_lvl_a_g2, :f1_lvl_b_g2,
                              :f1_lvl_a_interaction_with_f2_lvl_v_g2,
                              :f1_lvl_a_interaction_with_f2_lvl_w_g2,
                              :f1_lvl_b_interaction_with_f2_lvl_v_g2,
                              :f1_lvl_b_interaction_with_f2_lvl_w_g2]
              expect(model_fit.ran_ef.keys).to eq(ran_ef_names)
              expect(model_fit.ran_ef_names).to eq(ran_ef_names)
            end
          end

          context "with one non-interaction fixed effects term, with fixed intercept, without random non-intercept terms" do
            
            case constructor_method
            when "#from_formula"
              subject(:model_fit) { LMM.from_formula(formula: "y ~ f2 + f1:f2 + (1 | gr)", data: df) }
            when "#from_daru"
              subject(:model_fit) do
                LMM.from_daru(response: :y, fixed_effects: [:intercept, :f2, [:f1, :f2]],
                              random_effects: [[:intercept]], grouping: [:gr], data: df)
              end
            end

            # Result from R for comparison:
            #
            #  > mod <- lmer(y ~ f2 + f1:f2 + (1 | gr), data=df)
            #  > summary(mod)
            #  Linear mixed model fit by REML ['lmerMod']
            #  Formula: y ~ f2 + f1:f2 + (1 | gr)
            #     Data: df
            #
            #  REML criterion at convergence: 300.6
            #
            #  Scaled residuals: 
            #       Min       1Q   Median       3Q      Max 
            #  -2.70785 -0.74301  0.00105  0.54929  2.49694 
            #
            #  Random effects:
            #   Groups   Name        Variance Std.Dev.
            #   gr       (Intercept) 0.000    0.000   
            #   Residual             1.199    1.095   
            #  Number of obs: 100, groups:  gr, 2
            #
            #  Fixed effects:
            #              Estimate Std. Error t value
            #  (Intercept)   7.7302     0.3037  25.457
            #  f2v           1.2335     0.4217   2.925
            #  f2w          -2.3396     0.3985  -5.871
            #  f2u:f1b      -1.1198     0.3941  -2.842
            #  f2v:f1b      -5.5480     0.3778 -14.686
            #  f2w:f1b      -1.7754     0.3828  -4.638
            #
            #  Correlation of Fixed Effects:
            #          (Intr) f2v    f2w    f2:f1b f2v:f1
            #  f2v     -0.720                            
            #  f2w     -0.762  0.549                     
            #  f2u:f1b -0.771  0.555  0.587              
            #  f2v:f1b  0.000 -0.537  0.000  0.000       
            #  f2w:f1b  0.000  0.000 -0.437  0.000  0.000

            it "finds the minimal REML deviance correctly" do
              expect(model_fit.deviance).to be_within(1e-1).of(300.6)
            end

            it "estimates the residual standard deviation correctly" do
              expect(model_fit.sigma).to be_within(1e-2).of(1.095)
            end

            it "estimates the fixed effects terms correctly" do
              fix_ef_from_R = [7.7302, 1.2335, -2.3396, -1.1198, -5.5480, -1.7754]
              model_fit.fix_ef.values.each_with_index do |e, i|
                expect(e).to be_within(1e-2).of(fix_ef_from_R[i])
              end
            end
            
            it "names the fixed effects correctly" do
              fix_ef_names = [:intercept, :f2_lvl_v, :f2_lvl_w, :f1_lvl_b_interaction_with_f2_lvl_u, 
                              :f1_lvl_b_interaction_with_f2_lvl_v, :f1_lvl_b_interaction_with_f2_lvl_w]
              expect(model_fit.fix_ef.keys).to eq(fix_ef_names)
              expect(model_fit.fix_ef_names).to eq(fix_ef_names)
            end

            it "names the random effects correctly" do
              ran_ef_names = [:intercept_g1, :intercept_g2]
              expect(model_fit.ran_ef.keys).to eq(ran_ef_names)
              expect(model_fit.ran_ef_names).to eq(ran_ef_names)
            end
          end
        end
      end
    end
  end
end
