require 'mixed_models'

#############################################################################################
# Model with interaction effects of two numeric variables, in the fixed and random effects # 
#############################################################################################

df = Daru::DataFrame.from_csv './data/numeric_x_numeric_interaction.csv'

model_fit = LMM.from_formula(formula: "y ~ a + b + a:b + (0 + a:b | gr)", data: df)

# Print some results
puts "REML criterion: \t#{model_fit.deviance}"
puts "Fixed effects:"
puts model_fit.fix_ef
puts "Standard deviation: \t#{model_fit.sigma}"

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
