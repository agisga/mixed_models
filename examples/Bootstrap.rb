require 'mixed_models'

df = Daru::DataFrame.from_csv './data/alien_species.csv'
model_fit = LMM.from_formula(formula: "Aggression ~ Age + Species + (Age | Location)", data: df)
 
# get 1000 bootstap estimates for the model intercept
nsim = 1000
result = model_fit.bootstrap(nsim: nsim)
y = Array.new
nsim.times do |i|
  y << result[i][:intercept]
end

# plot a histogram of the results
require 'gnuplotrb'
include GnuplotRB
bin_width = (y.max - y.min)/10.0
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