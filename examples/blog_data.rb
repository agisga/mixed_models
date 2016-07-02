##############################################################################
# If a blog post has received a comment in the first 24 hours after 
# publication, how many more comments will it receive before 24 hours 
# after its publication have passed?
#
# Use LMM in order to model the number of comments that a blog 
# post receives as a function of the length of the post, the 
# number of parent blog post, the number of comments of the parents, etc.
##############################################################################

# daru requires csv files to have a header line; we add a header
# to the data file and resave it
without_header = './data/blogData_train.csv'
with_header = './data/blogData_train_with_header.csv'
colnames = (1..281).to_a.map { |x| "v#{x}" }
header = colnames.join(',')
File.open(with_header, 'w') do |fo|
  fo.puts header
  File.foreach(without_header) do |li|
    fo.puts li
  end
end

# load the data with daru
require 'daru'
df = Daru::DataFrame.from_csv './data/blogData_train_with_header.csv'

# select a subset of columns of the data frame
keep = ['v16', 'v41', 'v54', 'v62', 'v270', 'v271', 'v272', 
        'v273', 'v274', 'v275', 'v276', 'v277', 'v280']
blog_data = df[*keep]
df = nil

# assign meaningful names for those columns
meaningful_names = [:host_comments_avg, :host_trackbacks_avg, 
                    :comments, :length, :mo, :tu, :we, :th, 
                    :fr, :sa, :su, :parents, :parents_comments]
blog_data.vectors = Daru::Index.new(meaningful_names)

# extract observations with text length >0 and >0 comments
nonzero_ind = blog_data[:length].each_index.select do |i| 
  blog_data[:length][i] > 0 && blog_data[:comments][i] > 0
end
blog_data = blog_data.row[*nonzero_ind]

# combine the seven day-of-week columns into one column;
# and remove the seven columns
days = Array.new(blog_data.nrows) { :unknown }
[:mo, :tu, :we, :th, :fr, :sa, :su].each do |d|
  ind = blog_data[d].to_a.each_index.select { |i| blog_data[d][i]==1 }
  ind.each { |i| days[i] = d.to_s }
  blog_data.delete_vector(d)
end
blog_data[:day] = days

# create a binary indicator vector specifying if a blog post has at least 
# one parent post which has comments
hpwc = (blog_data[:parents] * blog_data[:parents_comments]).to_a
blog_data[:has_parent_with_comments] = hpwc.map { |t| t == 0 ? 'no' : 'yes'} 
blog_data.delete_vector(:parents)
blog_data.delete_vector(:parents_comments)

# log transforms make the residuals look more normal
log_comments = blog_data[:comments].to_a.map { |c| Math::log(c) }
log_host_comments_avg = blog_data[:host_comments_avg].to_a.map { |c| Math::log(c) }
blog_data[:log_comments] = log_comments
blog_data[:log_host_comments_avg] = log_host_comments_avg

# fit a LMM
require 'mixed_models'
model_fit = LMM.from_formula(formula: "log_comments ~ log_host_comments_avg + host_trackbacks_avg + length + has_parent_with_comments + (1 | day)", 
                             data: blog_data)

# Print some results
puts "Obtained fixed effects coefficient estimates:"
puts model_fit.fix_ef
puts "Wald p-values for the fixed effects:"
puts model_fit.fix_ef_p(method: :wald)

conf_int = model_fit.fix_ef_conf_int(level: 0.95, method: :wald)
ci = Daru::DataFrame.rows(conf_int.values, order: [:lower95, :upper95], index: model_fit.fix_ef_names)
ci[:coef] = model_fit.fix_ef.values
puts "Wald 95% confidence intervals:"
puts ci

puts "Obtained random effects coefficient estimates:"
puts model_fit.ran_ef
puts "Random effects variance:"
puts model_fit.sigma_mat[0,0]

