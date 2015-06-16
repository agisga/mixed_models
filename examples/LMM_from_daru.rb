require 'MixedModels'

#########################################################################
# Model with numerical variables only; random intercept and slope #######
#########################################################################

# Fixed effects coefficient vector
beta = NMatrix.new([2,1], [1,1], dtype: :float64)

# Generate the random effects vector 
# Values generated by R from the multivariate distribution with mean 0
# and covariance matrix [ [1, 0.5], [0.5, 1] ]
b_array = [ -1.34291864, 0.37214635,-0.42979766, 0.03111855, 1.98241161, 
            0.71735038, 0.40448848,-0.28236437, 0.33479745,-0.11086452 ]
b = NMatrix.new([10,1], b_array, dtype: :float64)

# Generate the random residuals vector
# Values generated from the standard Normal distribution
epsilon_array = [0.496502397340098, -0.577678887521082, -1.21173791274225, 0.0446417152929314, 0.339674378865471, 0.104784564191674, -0.0460565196653141, 0.285440902222387, 0.843345193001128, 1.27994921528088, 0.694924670755951, 0.577415255292851, 0.370159180245536, 0.35881413147769, -1.69691116206306, -0.233385719476208, 0.480331989945522, -1.09503905124389, -0.610978188869429, 0.984812801235286, 0.282422385731771, 0.763463942012845, -1.03154373185159, -0.374926162762322, -0.650793255606928, 0.793247584007507, -1.30007701703736, -2.522510645489, 0.0246284050971783, -1.73792367490139, 0.0267032433302985, 1.09659910679367, 0.747140189824456, -0.527345699932755, 1.24561748663327, 0.20905974976202, 0.00753104790432846, -0.0866226204494824, -1.61282076369275, -1.25760486584371, -0.885299440717284, 1.07254194203703, 0.101861345622785, -1.86859557570558, -0.0660433241114955, 0.684044990424631, 0.266888559603417, 0.763767965816189, 0.427908801177724, -0.146381705894295]
epsilon = NMatrix.new([50,1], epsilon_array, dtype: :float64)

# Generate the response vector
x_array = Array.new(100) { 1 }
x_array.each_index { |i| x_array[i]=(i+1)/2 if (i+1)%2==0 } 
x = NMatrix.new([50,2], x_array, dtype: :float64)
grp_mat = NMatrix.zeros([50,5], dtype: :float64)
[0,10,20,30,40].each { |i| grp_mat[i...(i+10), i/10] = 1.0 }
z = grp_mat.khatri_rao_rows x
y = (x.dot beta) + (z.dot b) + epsilon

# data
grps = Array.new 
[0,10,20,30,40].each { |i| grps[i...(i+10)] = Array.new(10){i/10} }
df = Daru::DataFrame.new([y.to_flat_a, (1..50).to_a, grps], order: [:y, :x, :grp])

# Fit the model
model_fit = LMM.from_daru(response: :y, fixed_effects: [:intercept, :x], 
                          random_effects: [[:intercept, :x]], grouping: [:grp],
                          data: df)

# Print some results
puts "-------------------------------------------"
puts "Model with numerical variables only; random intercept and slope"
puts "-------------------------------------------"

puts "(1) Model fit"
puts "Optimal theta: \t#{model_fit.theta_optimal}"
puts "REML criterion: \t#{model_fit.dev_optimal}"

puts "(2) Fixed effects"
puts "Intercept: \t#{model_fit.fix_ef["x0"]}"
puts "Slope: \t#{model_fit.fix_ef["x1"]}"

puts "(3) Random effects"
sd1 = Math::sqrt(model_fit.sigma_mat[0,0])
puts "Random intercept sd: \t#{sd1}"
sd2 = Math::sqrt(model_fit.sigma_mat[1,1])
puts "Random slope sd: \t#{sd2}"
puts "Correlation of random intercept and slope: \t#{model_fit.sigma_mat[0,1] / (sd1*sd2)}"

puts "(4) Residuals"
puts "Variance: \t#{model_fit.sigma2}"
puts "Standard deviation: \t#{Math::sqrt(model_fit.sigma2)}"

puts "-------------------------------------------"


##############################################################################
# Model with numerical and categorical variables as fixed and random effects # 
##############################################################################

df = Daru::DataFrame.from_csv '/home/alexej/github/MixedModels/examples/data/categorical_and_crossed_ran_ef.csv'

# Fit the model
#model_fit = LMM.from_daru(response: :y, fixed_effects: [:intercept, :x, :a], 
#                          random_effects: [[:intercept, :x], [:intercept, :a]], 
#                          grouping: [:grp_for_x, :grp_for_a],
#                          data: df)

model_fit = LMM.from_daru(response: :y, fixed_effects: [:intercept, :x, :a], 
                          random_effects: [[:intercept, :x]], 
                          grouping: [:grp_for_x],
                          data: df)

# Print some results
puts "\n-------------------------------------------"
puts "Model with numerical and categorical variables as fixed and random effects"
puts "-------------------------------------------"
puts "REML criterion: \t#{model_fit.dev_optimal}"
puts "Fixed effects:"
puts model_fit.fix_ef
puts "Variance: \t#{model_fit.sigma2}"
puts "Standard deviation: \t#{Math::sqrt(model_fit.sigma2)}"
puts "-------------------------------------------"
