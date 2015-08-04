# mixed_models

[![Build Status](https://travis-ci.org/agisga/mixed_models.svg?branch=master)](https://travis-ci.org/agisga/mixed_models)

Fit statistical linear models with fixed and mixed (random) effects in Ruby.

**Early stage work in progress.**

## Installation

<!-- Add this line to your application's Gemfile:

```ruby
gem 'mixed_models'
```
And then execute:

    $ bundle

Or install it yourself as:

    $ gem install mixed_models -->

The development version of `mixed_models` can be installed using the command line:

```
git clone https://github.com/agisga/mixed_models.git
cd mixed_models/
bundle install
bundle exec rake install
```

Ruby is required in version >=2.0 because keyword arguments are excessively used in `mixed_models`.

Feel free to contact me at alexej [dot] go [at] googlemail [dot] com, in case of difficulties with the installation.

## Usage Examples

* [This IRuby notebook](http://nbviewer.ipython.org/github/agisga/mixed_models/blob/master/notebooks/LMM.ipynb) shows an example linear mixed model fit and most of the methods available for objects of class `LMM`.

* [This IRuby notebook](http://nbviewer.ipython.org/github/agisga/mixed_models/blob/master/notebooks/blog_data.ipynb) shows an application of `mixed_models` to real data, which originate from blog posts from various sources in 2010-2012. The analyzed data set is of nontrivial size (~50000x300) and nontrivial structure (redundant variables, missing values, data transformations, etc.).

Some other examples in form of Ruby code can be found in [the examples folder](https://github.com/agisga/mixed_models/tree/master/examples).

Further examples can be found in several BLOG posts (see below).

## Some relevant BLOG posts

* [A (naive) application of linear mixed models to genetics](http://agisga.github.io/mixed_models_applied_to_family_SNP_data/)

* [Wald p-values and confidence intervals for fixed effects coefficients](http://agisga.github.io/MixedModels_p_values_and_CI/)

* [Fitting linear mixed models with a user-friendly R-formula-like interface](http://agisga.github.io/MixedModels_from_formula/)

* [A rudimentary linear mixed models fit from raw matrices and vectors](http://agisga.github.io/First-linear-mixed-model-fit/)

<!-- ## Usage

TODO: Write usage instructions here

## Development

After checking out the repo, run `bin/setup` to install dependencies. Then, run `bin/console` for an interactive prompt that will allow you to experiment.

To install this gem onto your local machine, run `bundle exec rake install`. To release a new version, update the version number in `version.rb`, and then run `bundle exec rake release` to create a git tag for the version, push git commits and tags, and push the `.gem` file to [rubygems.org](https://rubygems.org).
-->

## Contributing

1. Fork it ( https://github.com/[my-github-username]/mixed_models/fork )
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create a new Pull Request
