require 'mixed_models'

def read_csv_into_array(filename)
  f = File.new(filename)
  lines_array = Array.new
  f.each_line { |line| lines_array.push(line) }
  f.close
  lines_array.each_index do |i| 
    lines_array[i] = lines_array[i].split(",") 
    lines_array[i].each_index { |j| lines_array[i][j] = lines_array[i][j].to_f }
  end
  return lines_array
end

# fixed effects design matrix
x_array = read_csv_into_array("./data/design_matrix.csv")
n = x_array.length
m = x_array[0].length
x_array.unshift(Array.new(n) {1.0}) # intercept
x = NMatrix.new([n,m+1], x_array.flatten, dtype: :float64)

# response vector
y_array = read_csv_into_array("./data/phenotype.csv")
y = NMatrix.new([n,1], y_array.flatten, dtype: :float64)

# random effects model matrix
z = NMatrix.identity([n,n], dtype: :float64)

# kinship matrix, which determines the random effects covariance matrix
kinship_array = read_csv_into_array("./data/kinship_matrix.csv")
kinship_mat = NMatrix.new([n,n], kinship_array.flatten, dtype: :float64)
# upper triangulat Cholesky factor
kinship_mat_cholesky_factor = kinship_mat.factorize_cholesky[0] 

# fixed effects names
x_names = [:Intercept]
1.upto(m) { |i| x_names.push("SNP#{i}".to_sym) }

# Fit the model
model_fit = LMM.new(x: x, y: y, zt: z,
                    x_col_names: x_names, 
                    start_point: [2.0], 
                    lower_bound: [0.0]) { |th| kinship_mat_cholesky_factor * th[0] }


# Print some results
puts "Optimal theta: \t#{model_fit.theta}"
puts "REML criterion: \t#{model_fit.deviance}"

p_vals = model_fit.fix_ef_p
significant = Array.new
p_vals.each_key { |k| significant.push(k) if p_vals[k] < 0.05 }
puts "Fixed effects with Wald p-values <0.05:"
puts significant.join(', ')

puts "Variance due to family relatedness: \t#{model_fit.theta[0]**2.0 / 2}"
puts "Residual variance: \t#{model_fit.sigma2}"
