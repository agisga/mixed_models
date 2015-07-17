require 'mixed_models'

#############################################################################################
# Model with interaction effects of two numeric variables, in the fixed and random effects # 
#############################################################################################

df_num_x_num = Daru::DataFrame.from_csv './data/numeric_x_numeric_interaction.csv'

num_x_num = LMM.from_formula(formula: "y ~ a + b + a:b + (0 + a:b | gr)", data: df_num_x_num)

# Print some results
puts "REML criterion: \t#{num_x_num.deviance}"
puts "Fixed effects:"
puts num_x_num.fix_ef
puts "Standard deviation: \t#{num_x_num.sigma}"

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

#############################################################################################
# Model with interaction effects of a numeric and a categorical  variable, 
# in the fixed and random effects  
#############################################################################################

df_num_x_cat = Daru::DataFrame.from_csv './data/numeric_x_categorical_interaction.csv'

num_x_cat = LMM.from_formula(formula: "y ~ num + cat + num:cat + (0 + num:cat | gr)", data: df_num_x_cat)

# Print some results
puts "REML criterion: \t#{num_x_cat.deviance}"
puts "Fixed effects:"
puts num_x_cat.fix_ef
puts "Random effects:"
puts num_x_cat.ran_ef
puts "Standard deviation: \t#{num_x_cat.sigma}"

# Result from R for comparison
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
