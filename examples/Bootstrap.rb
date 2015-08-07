require 'mixed_models'

df = Daru::DataFrame.from_csv './data/alien_species.csv'
model_fit = LMM.from_formula(formula: "Aggression ~ Age + Species + (Age | Location)", data: df)
 
########################################################################
# Get bootstap estimates for the model intercept,
# and plot a histogram of the bootstrap distribution
########################################################################

nsim = 1000
result = model_fit.bootstrap(nsim: nsim)
y = nsim.times.map { |i| result[i][:intercept] }

# plot a histogram of the results
require 'gnuplotrb'
include GnuplotRB
bin_width = (y.max - y.min)/30.0
bins = (y.min..y.max).step(bin_width).to_a
rel_freq = Array.new(bins.length-1){0.0}
y.each do |r|
  0.upto(bins.length-2) do |i|
    if r >= bins[i] && r < bins[i+1] then
      rel_freq[i] += 1.0/y.length
    end
  end
end
bins_center = bins[0...-1].map { |b| b + bin_width/2.0 }
plot = Plot.new([[bins_center, rel_freq], with: 'boxes', notitle: true],
                style: 'fill solid 0.5')
plot.to_png('./plot.png', size: [600, 600])

########################################################################
# Compute confidence intervals from the bootstrap distribution,
# and compare them to Wald confidence intervals
########################################################################

# Compute basic bootstrap confidence intervals for the fixed effects coefficient estimates
ci_bootstrap = model_fit.fix_ef_conf_int(method: :bootstrap, boottype: :basic, nsim: nsim)
puts "Basic bootstrap confidence intervals:"
puts ci_bootstrap

# Compute Wald Z confidence intervals for the fixed effects coefficient estimates
ci_wald = model_fit.fix_ef_conf_int(method: :wald)
puts "Wald Z confidence intervals:"
puts ci_wald

# Compute confidence intervals for the fixed effects coefficient estimates with all available methods,
# and display the results in a table
ci = model_fit.fix_ef_conf_int(method: :all, nsim: nsim)
# round all results to two decimal places
ci.each_vector do |v|
  v.each_index { |i| v[i][0], v[i][1] = v[i][0].round(2), v[i][1].round(2)}
end
puts "Confidence intervals obtained with each of the available methods:"
puts ci.inspect(20)

# Benchmark parallel vs. single-threaded execution
require 'benchmark'

puts "Computation of studentized bootstrap confidence intervals single-threaded vs. in parallel:"
ci_bootstrap = nil
Benchmark.bm(17) do |bm|
  bm.report('single-threaded') do
    ci_bootstrap = model_fit.fix_ef_conf_int(method: :bootstrap, nsim: nsim, parallel: false)
  end

  bm.report('parallel') do
    ci_bootstrap = model_fit.fix_ef_conf_int(method: :bootstrap, nsim: nsim, parallel: true)
  end
end
