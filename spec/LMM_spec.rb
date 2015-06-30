require 'mixed_models'

describe LMM do
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
  end

  describe "#from_formula" do
    context "with numeric and categorical fixed effects and numeric random effects" do
      context "using REML deviance" do
        subject(:model_fit) do
          LMM.from_formula(formula: "Aggression ~ Age + Species + (Age | Location)", 
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

    context "on categorical data with fixed intercept but no random intercept" do
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
      end

      it "names the random effects correctly" do
        ran_ef_names = [:x_lvl_A_g1, :x_lvl_B_g1, :x_lvl_C_g1, 
                        :x_lvl_A_g2, :x_lvl_B_g2, :x_lvl_C_g2]
        expect(model_fit.ran_ef.keys).to eq(ran_ef_names)
      end
    end

    context "on categorical data with random intercept but no fixed intercept" do
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
      end

      it "names the random effects correctly" do
        ran_ef_names = [:intercept_g1, :x_lvl_B_g1, :x_lvl_C_g1, 
                        :intercept_g2, :x_lvl_B_g2, :x_lvl_C_g2]
        expect(model_fit.ran_ef.keys).to eq(ran_ef_names)
      end
    end
  end
end
