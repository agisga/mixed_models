require 'mixed_models'

#TODO: add better spec for LMM#sigma_mat; add spec for LMM#residuals, LMM#sse

describe LMM do

  context "with numeric and categorical fixed effects and numeric random effects" do
    describe "#from_formula" do
      context "using REML deviance" do
        subject(:model_fit) do
          LMM.from_formula(formula: "Aggression ~ Age + Species + (Age | Location)", reml: true,
                           data: Daru::DataFrame.from_csv("spec/data/alien_species.csv"))
        end

        # compare the obtained estimates to the ones obtained for the same data 
        # by the function lmer from the package lme4 in R:
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
          expect(model_fit.deviance).to be_within(1e-4).of(333.7155)
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

        it "names the fixed effects correctly" do
          fix_ef_names = [:intercept, :Age, :Species_lvl_Human, :Species_lvl_Ood, :Species_lvl_WeepingAngel]
          expect(model_fit.fix_ef.keys).to eq(fix_ef_names)
        end

        it "names the random effects correctly" do
          ran_ef_names = [:intercept_Asylum, :Age_Asylum, :intercept_Earth, 
                          :Age_Earth, :intercept_OodSphere, :Age_OodSphere]
          expect(model_fit.ran_ef.keys).to eq(ran_ef_names)
        end
      end

      context "using deviance function instead of REML" do
        subject(:model_fit) do
          LMM.from_formula(formula: "Aggression ~ Age + Species + (Age | Location)", reml: false,
                           data: Daru::DataFrame.from_csv("spec/data/alien_species.csv"))
        end

        # compare the obtained estimates to the ones obtained for the same data 
        # by the function lmer from the package lme4 in R:
        #
        #  > mod <- lmer("Aggression~Age+Species+(Age|Location)", data=alien.species, REML=FALSE)
        #  > deviance(mod)
        #  [1] 337.5662
        #  > fixef(mod)
        #          (Intercept)                 Age        SpeciesHuman          SpeciesOod SpeciesWeepingAngel 
        #        1016.28645465         -0.06531612       -499.69371063       -899.56876058       -199.58889858 
        #  > ranef(mod)
        #  $Location
        #            (Intercept)         Age
        #  Asylum     -116.68051 -0.03353435
        #  Earth        83.86296 -0.13612453
        #  OodSphere    32.81756  0.16965888
        #  > sigma(mod)
        #  [1] 0.9588506

        it "finds the minimal deviance correctly" do
          expect(model_fit.deviance).to be_within(1e-4).of(337.5662)
        end

        it "estimates the residual standard deviation correctly" do
          expect(model_fit.sigma).to be_within(1e-4).of(0.9588)
        end

        it "estimates the fixed effects terms correctly" do
          fix_ef_from_R = [1016.2864, -0.06531, -499.6937, -899.5688, -199.5889] 
          model_fit.fix_ef.values.each_with_index do |e, i|
            expect(e).to be_within(1e-4).of(fix_ef_from_R[i])
          end
        end

        it "estimates the random effects terms correctly" do
          ran_ef_from_R = [-116.6805, -0.0335, 83.8630, -0.1361, 32.8176, 0.1697] 
          model_fit.ran_ef.values.each_with_index do |e, i|
            expect(e).to be_within(1e-4).of(ran_ef_from_R[i])
          end
        end
      end
    end

    subject(:model_fit) do
      LMM.from_formula(formula: "Aggression ~ Age + Species + (Age | Location)",
                       data: Daru::DataFrame.from_csv("spec/data/alien_species.csv"))
    end

    # compare the obtained estimates to the ones obtained for the same data 
    # by the function lmer from the package lme4 in R:
    #
    #  > mod <- lmer(Aggression~Age+Species+(Age|Location), data=alien.species)
    #  
    #  # predictions
    #  > newdata <- read.table("alien_species_newdata.csv",sep=",", header=T)
    #  > predict(mod, newdata)
    #            1           2           3           4           5           6           7           8           9          10 
    #  1070.912575  182.452063  -17.064468  384.788159  876.124072  674.711338 1092.698558  871.150884  687.462998   -4.016258 
    #  > pred <- predict(mod, newdata, re.form=NA)
    #  > pred
    #          1         2         3         4         5         6         7         8         9        10 
    #  1002.6356  110.8389  105.4177  506.5997  800.0421  799.9768 1013.8700  807.1616  808.4026  114.0394 
    #
    #  # confidence intervals for the predictions
    #  > newdata$Aggression <- 0 
    #  > m <- model.matrix(terms(mod), newdata)
    #  > pred.stdev <- sqrt(diag(m %*% tcrossprod(vcov(mod),m)))
    #  # lower bounds of 88% CI
    #  > pred - qnorm(1-0.06) * pred.stdev
    #          1         2         3         4         5         6         7         8         9        10 
    #  906.32839  17.21062  10.21883 411.90672 701.96039 701.85218 920.50198 712.62678 714.24725  20.67199 
    #  # upper bounds of 88% CI
    #  > pred + qnorm(1-0.06) * pred.stdev
    #          1         2         3         4         5         6         7         8         9        10 
    #  1098.9429  204.4673  200.6166  601.2926  898.1239  898.1015 1107.2381  901.6964  902.5580  207.4069 
    #
    #  # prediction intervals
    #
    #  # standard deviations of the fixed effects estimates
    #  > sqrt(diag(vcov(mod)))
    #  [1] 60.15942377  0.08987599  0.26825658  0.28145153  0.27578794
    #
    #  # covariance matrix of the random effects coefficient estimates
    #  > vcov(mod)
    #  5 x 5 Matrix of class "dpoMatrix"
    #                        (Intercept)           Age  SpeciesHuman    SpeciesOod SpeciesWeepingAngel
    #  (Intercept)         3619.15626823 -3.231326e-01 -2.558786e-02 -2.882389e-02       -3.208552e-02
    #  Age                   -0.32313265  8.077694e-03 -4.303052e-05 -3.332581e-05        4.757580e-06
    #  SpeciesHuman          -0.02558786 -4.303052e-05  7.196159e-02  3.351678e-02        3.061001e-02
    #  SpeciesOod            -0.02882389 -3.332581e-05  3.351678e-02  7.921496e-02        3.165401e-02
    #  SpeciesWeepingAngel   -0.03208552  4.757580e-06  3.061001e-02  3.165401e-02        7.605899e-02
    #
    #  # conditional covariance matrix of the random effects estimates
    #  > re <- ranef(mod, condVar=TRUE)
    #  > m <- attr(re[[1]], which='postVar')
    #  > bdiag(m[,,1],m[,,2],m[,,3])
    #    6 x 6 sparse Matrix of class "dgCMatrix"
    #                                                                              
    #    [1,]  0.1098433621 -5.462386e-04  .             .             .             .           
    #    [2,] -0.0005462386  3.465442e-06  .             .             .             .           
    #    [3,]  .             .             0.1872216782 -8.651231e-04  .             .           
    #    [4,]  .             .            -0.0008651231  4.845105e-06  .             .           
    #    [5,]  .             .             .             .             0.1481118862 -7.468634e-04
    #    [6,]  .             .             .             .            -0.0007468634  4.748243e-06
    #
    #  # Wald Z-test for fixed effects coefficients
    #  > vc = vcov(mod)
    #  > z = fixef(mod) / sqrt(diag(vc))
    #  > z
    #          (Intercept)                 Age        SpeciesHuman          SpeciesOod SpeciesWeepingAngel 
    #           16.8932256          -0.7267364       -1862.7453318       -3196.1784666        -723.7044506 
    #  > pval = 2*(1-pnorm(abs(z)))
    #  > pval
    #          (Intercept)                 Age        SpeciesHuman          SpeciesOod SpeciesWeepingAngel 
    #            0.0000000           0.4673875           0.0000000           0.0000000           0.0000000
    #
    #  # Confidence intervals for fixed effects coefficients
    #  > confint(mod, method="Wald")
    #                             2.5 %       97.5 %
    #  (Intercept)          898.3764163 1134.1970241
    #  Age                   -0.2414699    0.1108376
    #  SpeciesHuman        -500.2194685 -499.1679220
    #  SpeciesOod          -900.1209556 -899.0176859
    #  SpeciesWeepingAngel -200.1294926 -199.0484237

    describe "#predict" do
      context "with Daru::DataFrame new data input" do
        let(:newdata) { Daru::DataFrame.from_csv("spec/data/alien_species_newdata.csv") }

        it "computes correct predictions when with_ran_ef is true"do
          result_from_R = [1070.912575, 182.452063, -17.064468, 384.788159, 876.124072, 
                           674.711338, 1092.698558, 871.150884, 687.462998, -4.016258]
          predictions = model_fit.predict(newdata: newdata, with_ran_ef: true)
          predictions.each_with_index { |p,i| expect(p).to be_within(1e-4).of(result_from_R[i]) }
        end

        it "computes correct predictions when with_ran_ef is false"do
          result_from_R = [1002.6356, 110.8389, 105.4177, 506.5997, 800.0421, 
                           799.9768, 1013.8700, 807.1616, 808.4026, 114.0394]
          predictions = model_fit.predict(newdata: newdata, with_ran_ef: false)
          predictions.each_with_index { |p,i| expect(p).to be_within(1e-4).of(result_from_R[i]) }
        end
      end
    end

    describe "#predict_with_intervals" do
      context "with DaruDataFrame newdata" do
        context "using type: :confidence" do
          let(:newdata) { Daru::DataFrame.from_csv("spec/data/alien_species_newdata.csv") }

          it "computes confidence intervals for predictions correctly" do
            lower88_from_R = [906.32839, 17.21062, 10.21883,411.90672,701.96039,701.85218,920.50198,712.62678,714.24725, 20.67199] 
            upper88_from_R = [1098.9429, 204.4673, 200.6166, 601.2926, 898.1239, 898.1015,1107.2381, 901.6964, 902.5580, 207.4069]
            result = model_fit.predict_with_intervals(newdata: newdata, level: 0.88, type: :confidence)
            result[:lower88].each_with_index { |l,i| expect(l/lower88_from_R[i]).to be_within(1e-2).of(1.0) }
            result[:upper88].each_with_index { |u,i| expect(u/upper88_from_R[i]).to be_within(1e-2).of(1.0) }
          end
          
          # a unit test for prediction intervals is implemented form raw model matrices below
        end
      end
    end

    describe "#fix_ef_sd" do
      it "computes the standard deviations of the fixed effects coefficients correctly" do
        result_from_R = [60.15942377, 0.08987599, 0.26825658, 0.28145153, 0.27578794]
        result = model_fit.fix_ef_sd.values
        result.each_index { |i| expect(result[i]/result_from_R[i]).to be_within(1e-2).of(1.0) }
      end
    end

    describe "#fix_ef_cov_mat" do
      it "computes the covariance matrix of the fixed effects coefficients correctly" do
        result_from_R = [3619.15626823, -3.231326e-01, -2.558786e-02, -2.882389e-02, -3.208552e-02, -0.32313265, 8.077694e-03, -4.303052e-05, -3.332581e-05, 4.757580e-06, -0.02558786, -4.303052e-05, 7.196159e-02, 3.351678e-02, 3.061001e-02, -0.02882389, -3.332581e-05, 3.351678e-02, 7.921496e-02, 3.165401e-02, -0.03208552, 4.757580e-06, 3.061001e-02, 3.165401e-02, 7.605899e-02]
        result = model_fit.fix_ef_cov_mat.to_flat_a
        result.each_index { |i| expect(result[i]/result_from_R[i]).to be_within(1e-2).of(1.0) }
      end
    end

    describe "#cond_cov_mat_ran_ef" do
      it "computes the conditional covariance matrix of " +
         "the random effects coefficient estimates correstly" do
        result_from_R = [0.1098433621, -5.462386e-04, 0.0, 0.0, 0.0, 0.0, -0.0005462386, 3.465442e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1872216782, -8.651231e-04, 0.0, 0.0, 0.0, 0.0, -0.0008651231, 4.845105e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1481118862, -7.468634e-04, 0.0, 0.0, 0.0, 0.0, -0.0007468634, 4.748243e-06]
        result = model_fit.cond_cov_mat_ran_ef.to_flat_a
        result.each_index do |i| 
          if result_from_R[i] == 0 then
            expect(result[i]).to eq(0)
          else
            expect(result[i]/result_from_R[i]).to be_within(1e-2).of(1.0)
          end
        end
      end
    end

    describe "#fix_ef_z" do
      it "computes the Wald z statistics correctly" do
        result_from_R = [16.8932256, -0.7267364, -1862.7453318, -3196.1784666, -723.7044506]
        result = model_fit.fix_ef_z.values
        result.each_index { |k| expect(result[k]/result_from_R[k]).to be_within(1e-3).of(1.0) }
      end
    end

    describe "#fix_ef_p" do
      context "with method: :wald" do
        it "computes the p-values from the Wald test correctly" do
          result_from_R = [0.0, 0.4673875, 0.0, 0.0, 0.0]
          result = model_fit.fix_ef_p(method: :wald).values
          result.each_index do |k| 
            if result_from_R[k] == 0 then
              expect(result[k]).to eq(0)
            else
              expect(result[k]/result_from_R[k]).to be_within(1e-4).of(1.0)
            end
          end
        end
      end
    end

    describe "#fix_ef_conf_int" do
      context "with method: :wald" do
        it "computes 95% confidence intervals for the fixed effects correctly" do
          result_from_R = [ [898.3764163, 1134.1970241], [-0.2414699, 0.1108376], 
                            [-500.2194685, -499.1679220], [-900.1209556, -899.0176859],
                            [-200.1294926, -199.0484237] ]
          result = model_fit.fix_ef_conf_int(level: 0.95, method: :wald).values
          result.each_index do |k| 
            expect(result[k][0]/result_from_R[k][0]).to be_within(1e-3).of(1.0)
            expect(result[k][1]/result_from_R[k][1]).to be_within(1e-3).of(1.0)
          end
        end
      end
    end

    describe "#refit" do
      let(:new_model) do
        model_fit.refit(newdata: Daru::DataFrame.from_csv("spec/data/alien_species_refit.csv"))
      end

      # compare the obtained estimates to the ones obtained for the same data 
      # by the function lmer from the package lme4 in R:
      #
      #  > mod <- lmer(Aggression ~ Age + Species + (Age | Location), data=df1)
      #  > REMLcrit(mod)
      #  [1] 363.913
      #  > fixef(mod)
      #          (Intercept)                 Age        SpeciesHuman 
      #           833.686766            2.329822         -400.714091 
      #           SpeciesOod SpeciesWeepingAngel 
      #          -838.990305          -99.530996 
      #  > ranef(mod)
      #  $Location
      #            (Intercept)        Age
      #  Asylum     -132.75250  0.1874363
      #  Earth       166.43088 -1.3432293
      #  OodSphere   -33.67838  1.1557931
      #
      #  > sigma(mod)
      #  [1] 7.535235

      it "finds the minimal REML deviance correctly" do
        expect(new_model.deviance).to be_within(1e-3).of(363.913)
      end

      it "estimates the residual standard deviation correctly" do
        expect(new_model.sigma).to be_within(1e-4).of(7.535235)
      end

      it "estimates the fixed effects terms correctly" do
        fix_ef_from_R = [833.686766, 2.329822, -400.714091, -838.990305, -99.530996] 
        new_model.fix_ef.values.each_with_index do |e, i|
          expect(e).to be_within(1e-4).of(fix_ef_from_R[i])
        end
      end

      it "estimates the random effects terms correctly" do
        ran_ef_from_R = [-132.75250, 0.1874363, 166.43088, -1.3432293, -33.67838, 1.1557931]
        new_model.ran_ef.values.each_with_index do |e, i|
          expect(e).to be_within(1e-4).of(ran_ef_from_R[i])
        end
      end

      it "names the fixed effects correctly" do
        fix_ef_names = [:intercept, :Age, :Species_lvl_Human, :Species_lvl_Ood, :Species_lvl_WeepingAngel]
        expect(new_model.fix_ef.keys).to eq(fix_ef_names)
      end

      it "names the random effects correctly" do
        ran_ef_names = [:intercept_Asylum, :Age_Asylum, :intercept_Earth, 
                        :Age_Earth, :intercept_OodSphere, :Age_OodSphere]
        expect(new_model.ran_ef.keys).to eq(ran_ef_names)
      end
    end

    describe "#simulate_new_response" do
      context "with type: :parametric" do
        let(:new_response) { model_fit.simulate_new_response(type: :parametric) }

        it "generates an Array" do
          expect(new_response.is_a?(Array)).to be_truthy
        end

        it "gives an Array of the correct length" do
          expect(new_response.length).to eq(model_fit.model_data.n)
        end

        it "gives an Array of Floats" do
          expect(new_response.all?{|y| y.is_a?(Float)}).to be_truthy
        end
      end
    end
  end

  context "fit from raw matrices with fixed and random intercept and slope" do
    subject(:model_fit) do
      # generate the 500x2 fixed effects design matrix
      x_array = Array.new(100) { 1 }
      x_array.each_index { |i| x_array[i]=(i+1)/2 if (i+1)%2==0 } 
      x = NMatrix.new([50,2], x_array, dtype: :float64)
      # create the fixed effects coefficient vector
      beta = NMatrix.new([2,1], [1,1], dtype: :float64)
      # generate the mixed effects model matrix
      # (assume a group structure with five groups of equal size)
      grp_mat = NMatrix.zeros([50,5], dtype: :float64)
      [0,10,20,30,40].each { |i| grp_mat[i...(i+10), i/10] = 1.0 }
      # (create matrix for random intercept and slope)
      z = grp_mat.khatri_rao_rows x
      # generate the random effects vector 
      b_array = [ -1.34291864, 0.37214635,-0.42979766, 0.03111855, 1.98241161, 
                  0.71735038, 0.40448848,-0.28236437, 0.33479745,-0.11086452 ]
      b = NMatrix.new([10,1], b_array, dtype: :float64)
      # generate the random residuals vector
      epsilon_array = [0.496502397340098, -0.577678887521082, -1.21173791274225, 0.0446417152929314, 0.339674378865471, 0.104784564191674, -0.0460565196653141, 0.285440902222387, 0.843345193001128, 1.27994921528088, 0.694924670755951, 0.577415255292851, 0.370159180245536, 0.35881413147769, -1.69691116206306, -0.233385719476208, 0.480331989945522, -1.09503905124389, -0.610978188869429, 0.984812801235286, 0.282422385731771, 0.763463942012845, -1.03154373185159, -0.374926162762322, -0.650793255606928, 0.793247584007507, -1.30007701703736, -2.522510645489, 0.0246284050971783, -1.73792367490139, 0.0267032433302985, 1.09659910679367, 0.747140189824456, -0.527345699932755, 1.24561748663327, 0.20905974976202, 0.00753104790432846, -0.0866226204494824, -1.61282076369275, -1.25760486584371, -0.885299440717284, 1.07254194203703, 0.101861345622785, -1.86859557570558, -0.0660433241114955, 0.684044990424631, 0.266888559603417, 0.763767965816189, 0.427908801177724, -0.146381705894295]
      epsilon = NMatrix.new([50,1], epsilon_array, dtype: :float64)
      # generate the response vector
      y = (x.dot beta) + (z.dot b) + epsilon
      # set up the covariance parameters
      parametrization = Proc.new do |th| 
        diag_blocks = Array.new(5) { NMatrix.new([2,2], [th[0],th[1],0,th[2]], dtype: :float64) }
        NMatrix.block_diagonal(*diag_blocks, dtype: :float64) 
      end
      # fit the model
      model_fit = LMM.new(x: x, y: y, zt: z.transpose,
                          start_point: [1,0,1], 
                          lower_bound: Array[0,-Float::INFINITY,0],
                          &parametrization) 
    end

    # compare the obtained estimates to the ones obtained for the same data
    # by the function lmer from the package lme4 in R:
    #
    #  library(lme4)
    #  library(MASS)
    #  X <- cbind(rep(1,50), 1:50)
    #  beta <- c(1,1)
    #  f <- gl(5,10)
    #  J <- t(as(f, Class="sparseMatrix"))
    #  Z <- t(KhatriRao(t(J),t(X)))
    #  b <- c(-1.34291864, 0.37214635, -0.42979766, 0.03111855, 1.98241161, 
    #         0.71735038, 0.40448848, -0.28236437, 0.33479745, -0.11086452)
    #  eps <- c(0.496502397340098, -0.577678887521082, -1.21173791274225, 0.0446417152929314, 0.339674378865471, 0.104784564191674, -0.0460565196653141, 0.285440902222387, 0.843345193001128, 1.27994921528088, 0.694924670755951, 0.577415255292851, 0.370159180245536, 0.35881413147769, -1.69691116206306, -0.233385719476208, 0.480331989945522, -1.09503905124389, -0.610978188869429, 0.984812801235286, 0.282422385731771, 0.763463942012845, -1.03154373185159, -0.374926162762322, -0.650793255606928, 0.793247584007507, -1.30007701703736, -2.522510645489, 0.0246284050971783, -1.73792367490139, 0.0267032433302985, 1.09659910679367, 0.747140189824456, -0.527345699932755, 1.24561748663327, 0.20905974976202, 0.00753104790432846, -0.0866226204494824, -1.61282076369275, -1.25760486584371, -0.885299440717284, 1.07254194203703, 0.101861345622785, -1.86859557570558, -0.0660433241114955, 0.684044990424631, 0.266888559603417, 0.763767965816189, 0.427908801177724, -0.146381705894295)
    #  y <- as.vector(X%*%beta + Z%*%b + eps)
    #  dat.frm <- as.data.frame(cbind(y, X[,2], rep(1:5,each=10)))
    #  names(dat.frm) <- c("y", "x", "grp")
    #  lmer.fit <- lmer(y~x+(x|grp), dat.frm)
    #  summary(lmer.fit)
    #  # Linear mixed model fit by REML ['lmerMod']
    #  # Formula: y ~ x + (x | grp)
    #  #    Data: dat.frm
    #  # 
    #  # REML criterion at convergence: 162.9
    #  # 
    #  # Scaled residuals: 
    #  #     Min      1Q  Median      3Q     Max 
    #  # -2.2889 -0.5211  0.1697  0.5040  1.7797 
    #  # 
    #  # Random effects:
    #  #  Groups   Name        Variance Std.Dev. Corr 
    #  #  grp      (Intercept) 15.4126  3.9259        
    #  #           x            0.1801  0.4244   -0.23
    #  #  Residual              0.6802  0.8247        
    #  # Number of obs: 50, groups:  grp, 5
    #  # 
    #  # Fixed effects:
    #  #             Estimate Std. Error t value
    #  # (Intercept)   2.7998     2.0484   1.367
    #  # x             1.1004     0.1938   5.679
    #  # 
    #  # Correlation of Fixed Effects:
    #  #   (Intr)
    #  # x -0.287
    #  ranef(lmer.fit)
    #  # $grp
    #  #   (Intercept)          x
    #  # 1   -3.644820  0.3937428
    #  # 2   -1.120587 -0.1415112
    #  # 3    3.558049  0.4611666
    #  # 4    3.490354 -0.5211430
    #  # 5   -2.282996 -0.1922552
    #  newdata <- as.data.frame(cbind(c(3.13, 55.12, -0.98, -99.34, 12.12), c(1,2,3,4,5)))
    #  names(newdata) <- c("x", "grp")
    #  predict(lmer.fit, newdata)
    #  #          1          2          3          4          5 
    #  #   3.831657  54.534064   4.827453 -51.254978  11.523682 
    #  predict(lmer.fit, newdata, re.form=NA)
    #  #           1           2           3           4           5 
    #  #    6.244062   63.454747    1.721348 -106.515678   16.136812 
    # 
    #  # prediction intervals on data that was used to fit the model
    #  yhat <- X %*% fixef(lmer.fit)  
    #  t(yhat)
    #  #          [,1]     [,2]     [,3]     [,4]     [,5]     [,6]     [,7]     [,8]
    #  # [1,] 3.900173 5.000591 6.101008 7.201425 8.301842 9.402259 10.50268 11.60309
    #  #          [,9]    [,10]    [,11]    [,12]    [,13]   [,14]    [,15]    [,16]
    #  # [1,] 12.70351 13.80393 14.90434 16.00476 17.10518 18.2056 19.30601 20.40643
    #  #         [,17]    [,18]    [,19]   [,20]    [,21]    [,22]    [,23]    [,24]
    #  # [1,] 21.50685 22.60726 23.70768 24.8081 25.90852 27.00893 28.10935 29.20977
    #  #         [,25]   [,26]    [,27]    [,28]    [,29]    [,30]    [,31]   [,32]
    #  # [1,] 30.31018 31.4106 32.51102 33.61144 34.71185 35.81227 36.91269 38.0131
    #  #         [,33]    [,34]    [,35]    [,36]    [,37]    [,38]    [,39]    [,40]
    #  # [1,] 39.11352 40.21394 41.31435 42.41477 43.51519 44.61561 45.71602 46.81644
    #  #         [,41]    [,42]    [,43]    [,44]    [,45]    [,46]    [,47]    [,48]
    #  # [1,] 47.91686 49.01727 50.11769 51.21811 52.31853 53.41894 54.51936 55.61978
    #  #         [,49]    [,50]
    #  # [1,] 56.72019 57.82061
    #  tmp <- VarCorr(lmer.fit)$grp
    #  Sigma.mat <- bdiag(tmp,tmp,tmp,tmp,tmp)
    #  pred.var <- diag(X %*% tcrossprod(vcov(lmer.fit),X)) + 
    #              diag(as.matrix(Z) %*% tcrossprod(Sigma.mat, Z)) + 
    #              sigma(lmer.fit)^2
    #  pred.sd <- sqrt(pred.var)
    #  #lower 95% CI
    #  t(yhat - qnorm(0.975) * pred.sd) 
    #  #           [,1]    [,2]      [,3]      [,4]       [,5]      [,6]      [,7]
    #  # [1,] -4.754873 -3.5758 -2.493847 -1.508396 -0.6157143 0.1904709 0.9181226
    #  #          [,8]     [,9]    [,10]    [,11]    [,12]    [,13]    [,14]    [,15]
    #  # [1,] 1.575996 2.172884 2.717093 3.216141 3.676635 4.104267 4.503878 4.879549
    #  #         [,16]    [,17]    [,18]    [,19]    [,20]   [,21]    [,22]    [,23]
    #  # [1,] 5.234712 5.572243 5.894558 6.203684 6.501333 6.78895 7.067761 7.338811
    #  #         [,24]    [,25]    [,26]    [,27]    [,28]    [,29]    [,30]    [,31]
    #  # [1,] 7.602995 7.861079 8.113725 8.361506 8.604919 8.844399 9.080326 9.313033
    #  #         [,32]    [,33]    [,34]    [,35]    [,36]    [,37]    [,38]  [,39]
    #  # [1,] 9.542814 9.769931 9.994614 10.21707 10.43748 10.65601 10.87281 11.088
    #  #         [,40]    [,41]    [,42]    [,43]   [,44]    [,45]    [,46]    [,47]
    #  # [1,] 11.30172 11.51406 11.72513 11.93502 12.1438 12.35155 12.55833 12.76422
    #  #         [,48]    [,49]    [,50]
    #  # [1,] 12.96926 13.17351 13.37701
    #  #upper 95% CI
    #  t(yhat + qnorm(0.975) * pred.sd)
    #  #          [,1]     [,2]     [,3]     [,4]    [,5]     [,6]     [,7]     [,8]
    #  # [1,] 12.55522 13.57698 14.69586 15.91125 17.2194 18.61405 20.08723 21.63019
    #  #          [,9]    [,10]    [,11]    [,12]    [,13]    [,14]    [,15]    [,16]
    #  # [1,] 23.23414 24.89076 26.59255 28.33289 30.10609 31.90731 33.73248 35.57815
    #  #         [,17]    [,18]    [,19]    [,20]    [,21]   [,22]    [,23]    [,24]
    #  # [1,] 37.44145 39.31997 41.21168 43.11486 45.02808 46.9501 48.87989 50.81654
    #  #         [,25]    [,26]    [,27]    [,28]    [,29]    [,30]    [,31]    [,32]
    #  # [1,] 52.75929 54.70748 56.66053 58.61795 60.57931 62.54421 64.51234 66.48339
    #  #         [,33]    [,34]    [,35]    [,36]    [,37]    [,38]    [,39]    [,40]
    #  # [1,] 68.45711 70.43326 72.41164 74.39206 76.37437 78.35841 80.34404 82.33116
    #  #         [,41]    [,42]    [,43]    [,44]    [,45]    [,46]   [,47]    [,48]
    #  # [1,] 84.31965 86.30942 88.30037 90.29242 92.28551 94.27955 96.2745 98.27029
    #  #         [,49]    [,50]
    #  # [1,] 100.2669 102.2642
    #  
    #  # fitted values
    #  fitted(lmer.fit)
    #  #          1          2          3          4          5          6          7 
    #  #  0.6490963  2.1432563  3.6374162  5.1315761  6.6257360  8.1198959  9.6140558 
    #  #          8          9         10         11         12         13         14 
    #  # 11.1082158 12.6023757 14.0965356 12.2271343 13.1860403 14.1449462 15.1038521 
    #  #         15         16         17         18         19         20         21 
    #  # 16.0627581 17.0216640 17.9805700 18.9394759 19.8983818 20.8572878 39.1510627 
    #  #         22         23         24         25         26         27         28 
    #  # 40.7126464 42.2742300 43.8358137 45.3973974 46.9589811 48.5205648 50.0821485 
    #  #         29         30         31         32         33         34         35 
    #  # 51.6437321 53.2053158 24.2476076 24.8268817 25.4061558 25.9854299 26.5647040 
    #  #         36         37         38         39         40         41         42 
    #  # 27.1439781 27.7232522 28.3025263 28.8818004 29.4610745 37.7513966 38.6595585 
    #  #         43         44         45         46         47         48         49 
    #  # 39.5677203 40.4758822 41.3840441 42.2922059 43.2003678 44.1085296 45.0166915 
    #  #         50 
    #  # 45.9248534 

    describe "#initialize" do
      it "estimates the fixed effects terms correctly" do
        fix_ef_from_R = [2.7998, 1.1004] 
        model_fit.fix_ef.values.each_with_index do |e, i|
          expect(e).to be_within(1e-2).of(fix_ef_from_R[i])
        end
      end

      it "estimates the random effects terms correctly" do
        ran_ef_from_R = [-3.644820, 0.3937428, -1.120587, -0.1415112, 3.558049, 
                         0.4611666, 3.490354, -0.5211430, -2.282996, -0.1922552]
        model_fit.ran_ef.values.each_with_index do |e, i|
          expect(e).to be_within(1e-2).of(ran_ef_from_R[i])
        end
      end
    end

    describe "#deviance" do
      it "finds the minimal REML deviance correctly" do
        expect(model_fit.deviance).to be_within(1e-2).of(162.90)
      end
    end

    describe "#sigma_mat" do
      let(:intercept_sd) { Math::sqrt(model_fit.sigma_mat[0,0]) }
      let(:slope_sd) { Math::sqrt(model_fit.sigma_mat[1,1]) }

      it "estimates the random intercept standard deviation correctly" do
        expect(intercept_sd).to be_within(1e-2).of(3.9259)
      end

      it "estimates the random slope standard deviation correctly" do
        expect(slope_sd).to be_within(1e-2).of(0.4244)
      end

      it "estimates the correlation of random intercept and slope correctly" do
        expect(model_fit.sigma_mat[0,1] / (intercept_sd * slope_sd)).to be_within(1e-2).of(-0.23)
      end
    end

    describe "#sigma" do
      it "estimates the residual standard deviation correctly" do
        expect(model_fit.sigma).to be_within(1e-2).of(0.8247)
      end
    end

    describe "#predict" do
      context "with raw matrix input" do
        let(:x) { NMatrix.new([5,2], [1.0, 3.13, 1.0, 55.12, 1.0, -0.98, 1.0, -99.34, 1.0, 12.12], 
                              dtype: :float64) }
        let(:z) { NMatrix.identity(5, dtype: :float64).khatri_rao_rows(x) }

        it "computes correct predictions when with_ran_ef is true"do
          result_from_R = [3.831657, 54.534064, 4.827453, -51.254978, 11.523682]
          predictions = model_fit.predict(x: x, z: z, with_ran_ef: true)
          predictions.each_with_index { |p,i| expect(p).to be_within(1e-2).of(result_from_R[i]) }
        end

        it "computes correct predictions when with_ran_ef is false"do
          result_from_R = [6.244062, 63.454747, 1.721348, -106.515678, 16.136812]
          predictions = model_fit.predict(x: x, z: z, with_ran_ef: false)
          predictions.each_with_index { |p,i| expect(p).to be_within(1e-2).of(result_from_R[i]) }
        end
      end

      context "without data frame or matrix input (i.e. on old data)" do
        context "using with_ran_ef: false" do
          it "computes the predictions correctly" do
            result_from_R = [3.900173,5.000591,6.101008,7.201425,8.301842,9.402259,10.50268,11.60309,12.70351,13.80393,14.90434,16.00476,17.10518,18.2056,19.30601,20.40643,21.50685,22.60726,23.70768,24.8081,25.90852,27.00893,28.10935,29.20977,30.31018,31.4106,32.51102,33.61144,34.71185,35.81227,36.91269,38.0131,39.11352,40.21394,41.31435,42.41477,43.51519,44.61561,45.71602,46.81644,47.91686,49.01727,50.11769,51.21811,52.31853,53.41894,54.51936,55.61978,56.72019,57.82061]
            result = model_fit.predict(with_ran_ef: false)
            result_from_R.zip(result).each { |p| expect(p[0]).to be_within(1e-2).of(p[1]) }
          end
        end

        context "using with_ran_ef: true" do
          it "computes the predictions correctly" do
            result_from_R = [0.6490963,2.1432563,3.6374162,5.1315761,6.6257360,8.1198959,9.6140558,11.1082158,12.6023757,14.0965356,12.2271343,13.1860403,14.1449462,15.1038521,16.0627581,17.0216640,17.9805700,18.9394759,19.8983818,20.8572878,39.1510627,40.7126464,42.2742300,43.8358137,45.3973974,46.9589811,48.5205648,50.0821485,51.6437321,53.2053158,24.2476076,24.8268817,25.4061558,25.9854299,26.5647040,27.1439781,27.7232522,28.3025263,28.8818004,29.4610745,37.7513966,38.6595585,39.5677203,40.4758822,41.3840441,42.2922059,43.2003678,44.1085296,45.0166915,45.9248534]
            result = model_fit.predict(with_ran_ef: true)
            result_from_R.zip(result).each { |p| expect(p[0]).to be_within(1e-2).of(p[1]) }
          end
        end
      end
    end

    describe "#predict_with_intervals" do
      context "from original (old) data" do
        context "using type: :prediction" do
          it "computes prediction intervals correctly" do
            lower95_from_R = [-4.754873, -3.5758, -2.493847, -1.508396, -0.6157143, 0.1904709, 0.9181226, 1.575996, 2.172884, 2.717093, 3.216141, 3.676635, 4.104267, 4.503878, 4.879549,5.234712,5.572243,5.894558,6.203684,6.501333,6.78895,7.067761,7.338811,7.602995,7.861079,8.113725,8.361506,8.604919,8.844399,9.080326,9.313033,9.542814,9.769931,9.994614,10.21707,10.43748,10.65601,10.87281,11.088,11.30172,11.51406,11.72513,11.93502,12.1438,12.35155,12.55833,12.76422,12.96926,13.17351,13.37701]
            upper95_from_R = [12.55522,13.57698,14.69586,15.91125,17.2194,18.61405,20.08723,21.63019,23.23414,24.89076,26.59255,28.33289,30.10609,31.90731,33.73248,35.57815,37.44145,39.31997,41.21168,43.11486,45.02808,46.9501,48.87989,50.81654,52.75929,54.70748,56.66053,58.61795,60.57931,62.54421,64.51234,66.48339,68.45711,70.43326,72.41164,74.39206,76.37437,78.35841,80.34404,82.33116,84.31965,86.30942,88.30037,90.29242,92.28551,94.27955,96.2745,98.27029,100.2669,102.2642]
            result = model_fit.predict_with_intervals(level: 0.95, type: :prediction)
            result[:lower95].each_with_index { |l,i| expect(l).to be_within(1e-2).of(lower95_from_R[i]) }
            result[:upper95].each_with_index { |u,i| expect(u).to be_within(1e-2).of(upper95_from_R[i]) }
          end
        end
      end
    end

    describe "#fitted" do
      context "using with_ran_ef: false" do
        it "computes the fitted values correctly" do
          result_from_R = [3.900173,5.000591,6.101008,7.201425,8.301842,9.402259,10.50268,11.60309,12.70351,13.80393,14.90434,16.00476,17.10518,18.2056,19.30601,20.40643,21.50685,22.60726,23.70768,24.8081,25.90852,27.00893,28.10935,29.20977,30.31018,31.4106,32.51102,33.61144,34.71185,35.81227,36.91269,38.0131,39.11352,40.21394,41.31435,42.41477,43.51519,44.61561,45.71602,46.81644,47.91686,49.01727,50.11769,51.21811,52.31853,53.41894,54.51936,55.61978,56.72019,57.82061]
          result = model_fit.fitted(with_ran_ef: false)
          result_from_R.zip(result).each { |p| expect(p[0]).to be_within(1e-2).of(p[1]) }
        end
      end

      context "using with_ran_ef: true" do
        it "computes the fitted values correctly" do
          result_from_R = [0.6490963,2.1432563,3.6374162,5.1315761,6.6257360,8.1198959,9.6140558,11.1082158,12.6023757,14.0965356,12.2271343,13.1860403,14.1449462,15.1038521,16.0627581,17.0216640,17.9805700,18.9394759,19.8983818,20.8572878,39.1510627,40.7126464,42.2742300,43.8358137,45.3973974,46.9589811,48.5205648,50.0821485,51.6437321,53.2053158,24.2476076,24.8268817,25.4061558,25.9854299,26.5647040,27.1439781,27.7232522,28.3025263,28.8818004,29.4610745,37.7513966,38.6595585,39.5677203,40.4758822,41.3840441,42.2922059,43.2003678,44.1085296,45.0166915,45.9248534]
          result = model_fit.fitted(with_ran_ef: true)
          result_from_R.zip(result).each { |p| expect(p[0]).to be_within(1e-2).of(p[1]) }
        end
      end
    end

    describe "#refit" do
      let(:new_model) do
        # specify similar model matrices as before (see above for more explanation)
        x_array = Array.new(100) { 1 }
        x_array.each_index { |i| x_array[i]=(i+1.0)/4.0 if (i+1)%2==0 } 
        x = NMatrix.new([50,2], x_array, dtype: :float64)
        beta = NMatrix.new([2,1], [1,3], dtype: :float64)
        grp_mat = NMatrix.zeros([50,5], dtype: :float64)
        grp_mat[0...10, 0] = 1.0
        grp_mat[10...40, 1] = 1.0
        grp_mat[40...45, 2] = 1.0
        grp_mat[45...47, 3] = 1.0
        grp_mat[47...50, 4] = 1.0
        z = grp_mat.khatri_rao_rows x
        b_array = [ -1.34291864, 0.37214635,-0.42979766, 0.03111855, 1.98241161, 
                    0.71735038, 0.40448848,-0.28236437, 0.33479745,-0.11086452 ]
        b = NMatrix.new([10,1], b_array, dtype: :float64)
        epsilon_array = [0.496502397340098, -0.577678887521082, -1.21173791274225, 0.0446417152929314, 0.339674378865471, 0.104784564191674, -0.0460565196653141, 0.285440902222387, 0.843345193001128, 1.27994921528088, 0.694924670755951, 0.577415255292851, 0.370159180245536, 0.35881413147769, -1.69691116206306, -0.233385719476208, 0.480331989945522, -1.09503905124389, -0.610978188869429, 0.984812801235286, 0.282422385731771, 0.763463942012845, -1.03154373185159, -0.374926162762322, -0.650793255606928, 0.793247584007507, -1.30007701703736, -2.522510645489, 0.0246284050971783, -1.73792367490139, 0.0267032433302985, 1.09659910679367, 0.747140189824456, -0.527345699932755, 1.24561748663327, 0.20905974976202, 0.00753104790432846, -0.0866226204494824, -1.61282076369275, -1.25760486584371, -0.885299440717284, 1.07254194203703, 0.101861345622785, -1.86859557570558, -0.0660433241114955, 0.684044990424631, 0.266888559603417, 0.763767965816189, 0.427908801177724, -0.146381705894295]
        epsilon = NMatrix.new([50,1], epsilon_array, dtype: :float64)
        y = (x.dot beta) + (z.dot b) + epsilon
        # fit the model
        new_model = model_fit.refit(x: x, y: y, zt: z.transpose)
      end

      # results from R for comparison:
      #    > X <- cbind(rep(1,50), seq(0.5,25,by=0.5))
      #    > beta <- c(1,3)
      #    > f <- as.factor(c(rep(1,10), rep(2,30), rep(3,5), 4, 4, 5, 5, 5))
      #    > J <- t(as(f, Class="sparseMatrix"))
      #    > Z <- t(KhatriRao(t(J),t(X)))
      #    > b <- c(-1.34291864, 0.37214635, -0.42979766, 0.03111855, 1.98241161, 0.71735038, 0.40448848, -0.28236437, 0.33479745, -0.11086452)
      #    > eps <- c(0.496502397340098, -0.577678887521082, -1.21173791274225, 0.0446417152929314, 0.339674378865471, 0.104784564191674, -0.0460565196653141, 0.285440902222387, 0.843345193001128, 1.27994921528088, 0.694924670755951, 0.577415255292851, 0.370159180245536, 0.35881413147769, -1.69691116206306, -0.233385719476208, 0.480331989945522, -1.09503905124389, -0.610978188869429, 0.984812801235286, 0.282422385731771, 0.763463942012845, -1.03154373185159, -0.374926162762322, -0.650793255606928, 0.793247584007507, -1.30007701703736, -2.522510645489, 0.0246284050971783, -1.73792367490139, 0.0267032433302985, 1.09659910679367, 0.747140189824456, -0.527345699932755, 1.24561748663327, 0.20905974976202, 0.00753104790432846, -0.0866226204494824, -1.61282076369275, -1.25760486584371, -0.885299440717284, 1.07254194203703, 0.101861345622785, -1.86859557570558, -0.0660433241114955, 0.684044990424631, 0.266888559603417, 0.763767965816189, 0.427908801177724, -0.146381705894295)
      #    > y <- as.vector(X%*%beta + Z%*%b + eps)
      #    > dat.frm <- as.data.frame(cbind(y, X[,2], f))
      #    > names(dat.frm) <- c("y", "x", "grp")
      #    > newfit <- lmer(y~x+(x|grp), dat.frm)
      #    > REMLcrit(newfit)
      #    [1] 158.0308
      #    > fixef(newfit)
      #    (Intercept)           x 
      #      0.1679018   3.2344651 
      #    > sigma(newfit)
      #    [1] 0.8934167
      #    > ranef(newfit)
      #    $grp
      #      (Intercept)          x
      #    1  -0.9996009  0.3727742
      #    2   0.6367488 -0.2374583
      #    3  -1.8286214  0.6819349
      #    4   1.3388462 -0.4992865
      #    5   0.8526273 -0.3179643

      it "estimates the fixed effects terms correctly" do
        fix_ef_from_R = [0.1679018, 3.2344651] 
        new_model.fix_ef.values.each_with_index do |e, i|
          expect(e).to be_within(1e-4).of(fix_ef_from_R[i])
        end
      end

      it "estimates the random effects terms correctly" do
        ran_ef_from_R = [-0.9996009, 0.3727742, 0.6367488, -0.2374583, -1.8286214,  
                         0.6819349, 1.3388462, -0.4992865, 0.8526273, -0.3179643] 
        new_model.ran_ef.values.each_with_index do |e, i|
          expect(e).to be_within(1e-4).of(ran_ef_from_R[i])
        end
      end

      it "finds the minimal REML deviance correctly" do
        expect(new_model.deviance).to be_within(1e-4).of(158.0308)
      end

      it "estimates the residual standard deviation correctly" do
        expect(new_model.sigma).to be_within(1e-4).of(0.8934167)
      end
    end
  end

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
  end

  context "with interaction effects" do
    context "between two numeric variables" do
      describe "#from_formula" do
        subject(:model_fit) do
          LMM.from_formula(formula: "y ~ a + b + a:b + (0 + a:b | gr)", 
                           data: Daru::DataFrame.from_csv("spec/data/numeric_x_numeric_interaction.csv"))
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
      end
    end

    context "between a numeric and a categorical variable" do
      describe "#from_formula" do
        subject(:model_fit) do
          LMM.from_formula(formula: "y ~ num + cat + num:cat + (0 + num:cat | gr)",
                           data: Daru::DataFrame.from_csv("spec/data/numeric_x_categorical_interaction.csv"))
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
    end

    context "between a categorical and a numeric variable" do
      describe "#from_formula" do
        subject(:model_fit) do
          LMM.from_formula(formula: "y ~ cat + num + cat:num + (0 + cat:num | gr)",
                           data: Daru::DataFrame.from_csv("spec/data/numeric_x_categorical_interaction.csv"))
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
    end

    context "between two categorical variables" do
      context "with both non-interaction fixed effects and fixed intercept, but no random non-interaction effects and without random intercept" do
        describe "#from_formula" do
          subject(:model_fit) do
            LMM.from_formula(formula: "y ~ f1 + f2 + f1:f2 + (0 + f1:f2 | gr)",
                             data: Daru::DataFrame.from_csv("spec/data/categorical_x_categorical_interaction.csv"))
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
      end

      context "without non-interaction fixed effects, with fixed intercept, with one random non-interaction effect and without random intercept" do
        describe "#from_formula" do
          subject(:model_fit) do
            LMM.from_formula(formula: "y ~ f1:f2 + (0 + f1 + f1:f2 | gr)",
                             data: Daru::DataFrame.from_csv("spec/data/categorical_x_categorical_interaction.csv"))
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
      end

      context "with one non-interaction fixed effects term, with fixed intercept, without random non-intercept terms" do
        describe "#from_formula" do
          subject(:model_fit) do
            LMM.from_formula(formula: "y ~ f2 + f1:f2 + (1 | gr)",
                             data: Daru::DataFrame.from_csv("spec/data/categorical_x_categorical_interaction.csv"))
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
