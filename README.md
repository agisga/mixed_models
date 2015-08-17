# mixed_models

[![Build Status](https://travis-ci.org/agisga/mixed_models.svg?branch=master)](https://travis-ci.org/agisga/mixed_models)

Fit statistical (linear) models with fixed and mixed (random) effects in Ruby.

## Features

#### Linear mixed models

* Support for the formula language of the `R` package `lme4` makes model specification convenient and user friendly. An expanation of the `lme4` formula interface can be found in the `lme4` [vignette](https://cran.r-project.org/web/packages/lme4/vignettes/lmer.pdf), or on stackexchange ([example](http://stats.stackexchange.com/questions/13166/rs-lmer-cheat-sheet)). Types of linear mixed models that can be fit by `mixed_models` with the formula interface include:
  - Models with interaction fixed or random effects (e.g. `y ~ x + z + x:z + (x + z + x:z | g)`),
  - Models with multiple crossed random effects (e.g. `y ~ x + (x|g) + (x|h)`),
  - Models with nested random effects (e.g. `y ~ x + (x|g) + (x|g:h)`). 

* Flexible model specification capabilities with the possibility to pass the random effects covariance structure as a `Proc`, when fitting a model from raw madel matrices.

* Possibility of singular fits (i.e models with random effects variance equal to zero).

* Likelihood ratio test for nested models (Chi squared or bootstrap based).

* Many types of hypotheses tests for the fixed and random effects (based on LRT, bootstrap or the Wald Z statistic).

* Four types of bootstrap confidence intervals for the fixed effects coefficient estimates, as well as Wald Z confidence intervals. All bootstrap based methods are performed in parallel by default.

* Prediction on new data and prediction intervals.

## Usage

#### Linear mixed models (LMM) tutorials

* [Fitting a linear mixed model](http://nbviewer.ipython.org/github/agisga/mixed_models/blob/master/notebooks/LMM_model_fitting.ipynb), accessing the estimated parameters, and assessing the quality of the model fit (to some extent)

* [Hypothesis tests and confidence intervals](http://nbviewer.ipython.org/github/agisga/mixed_models/blob/master/notebooks/LMM_tests_and_intervals.ipynb)

* [Predictions and prediction intervals](http://nbviewer.ipython.org/github/agisga/mixed_models/blob/master/notebooks/LMM_predictions.ipynb)

* [Models with multiple crossed or nested random effects](http://nbviewer.ipython.org/github/agisga/mixed_models/blob/master/notebooks/LMM_multiple_random_effects.ipynb)

#### Analysis of real data

* [This IRuby notebook](http://nbviewer.ipython.org/github/agisga/mixed_models/blob/master/notebooks/blog_data.ipynb) shows an application of `mixed_models` to real data, which originate from blog posts from various sources in 2010-2012. The analyzed data set is of nontrivial size (~50000x300) and nontrivial structure (redundant variables, missing values, data transformations, etc.).

#### More usage examples

Some other examples in form of Ruby code can be found in [the examples folder](https://github.com/agisga/mixed_models/tree/master/examples).

Further examples can be found in several BLOG posts (see below).

<!-- ## Installation

Ruby is required in version >=2.0 because keyword arguments are excessively used in `mixed_models`.

Add this line to your application's Gemfile:

```ruby
gem 'mixed_models'
```
And then execute:

    $ bundle

Or install it yourself as:

    $ gem install mixed_models -->

## Some relevant BLOG posts

* [Bootstrapping and bootstrap confidence intervals](http://agisga.github.io/bootstap_confidence_intervals/)

* [A (naive) application of linear mixed models to genetics](http://agisga.github.io/mixed_models_applied_to_family_SNP_data/)

* [Wald p-values and confidence intervals for fixed effects coefficients](http://agisga.github.io/MixedModels_p_values_and_CI/)

* [Fitting linear mixed models with a user-friendly R-formula-like interface](http://agisga.github.io/MixedModels_from_formula/)

* [A rudimentary linear mixed models fit from raw matrices and vectors](http://agisga.github.io/First-linear-mixed-model-fit/)


## Development

The development version of `mixed_models` can be installed using the command line:

```
git clone https://github.com/agisga/mixed_models.git
cd mixed_models/
bundle install
bundle exec rake install
```

The automatic tests can be run with `bundle exec rspec spec`.

<!-- To install this gem onto your local machine, run `bundle exec rake install`. To release a new version, update the version number in `version.rb`, and then run `bundle exec rake release` to create a git tag for the version, push git commits and tags, and push the `.gem` file to [rubygems.org](https://rubygems.org).
-->

## Contributing

1. Fork it ( https://github.com/[my-github-username]/mixed_models/fork )
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create a new Pull Request
