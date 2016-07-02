require 'mixed_models'

##############################################################################
# Model with numerical and categorical variables as fixed and random effects # 
##############################################################################

df = Daru::DataFrame.from_csv './data/alien_species.csv'
# mixed_models expects all variable names to be Symbols (not Strings):
df.vectors = Daru::Index.new( df.vectors.map { |v| v.to_sym } )

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
# mixed_models expects all variable names to be Symbols (not Strings):
dfnew.vectors = Daru::Index.new( dfnew.vectors.map { |v| v.to_sym } )

puts model_fit.predict(newdata: dfnew)
puts "88% confidence intervals for the predictions:"
ci = Daru::DataFrame.new(model_fit.predict_with_intervals(newdata: dfnew, level: 0.88, type: :confidence),
                         order: [:pred, :lower88, :upper88])
puts ci.inspect  
puts "88% prediction intervals for the predictions:"
pi = Daru::DataFrame.new(model_fit.predict_with_intervals(newdata: dfnew, level: 0.88, type: :prediction),
                         order: [:pred, :lower88, :upper88])
puts pi.inspect  
