require 'mixed_models'

##############################################################################
# Model with numerical and categorical variables as fixed and random effects # 
##############################################################################

df = Daru::DataFrame.from_csv './data/alien_species.csv'

model_fit = LMM.from_formula(formula: "Aggression ~ Age + Species + (Age | Location)", data: df)
 
# Print some results
puts "REML criterion: \t#{model_fit.deviance}"
puts "Fixed effects:"
puts model_fit.fix_ef
puts "Standard deviation: \t#{model_fit.sigma}"
puts "Random effects:"
puts model_fit.ran_ef
puts "Predictions of aggression levels on a new data set:"
dfnew = Daru::DataFrame.from_csv './data/alien_species_newdata.csv'
puts model_fit.predict(newdata: dfnew)
