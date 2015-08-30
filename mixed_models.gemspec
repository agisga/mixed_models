# coding: utf-8
lib = File.expand_path('../lib', __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require 'mixed_models/version.rb'

Gem::Specification.new do |spec|
  spec.name          = "mixed_models"
  spec.version       = MixedModels::VERSION
  spec.authors       = ["Alexej Gossmann"]
  spec.email         = ["alexej.go@googlemail.com"]

  spec.summary       = %q{Statistical mixed effects models}
  spec.description   = %q{Fit statistical (linear) models with fixed and mixed (random) effects in Ruby}
  spec.homepage      = "https://github.com/agisga/mixed_models.git"
  spec.license       = "BSD-3-Clause"

  spec.files         = `git ls-files -z`.split("\x0").reject { |f| f.match(%r{^(test|spec|features)/}) }
  spec.bindir        = "exe"
  spec.executables   = spec.files.grep(%r{^exe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]

  spec.add_development_dependency "bundler", "~> 1.9"
  spec.add_development_dependency "rake", "~> 10.0"
  spec.add_development_dependency "rspec", "~> 3.2"

  spec.add_runtime_dependency "nmatrix", "~> 0.2.0"
  spec.add_runtime_dependency "nmatrix-lapacke", "~> 0.2.0"
  spec.add_runtime_dependency "daru", "= 0.1.0"
  spec.add_runtime_dependency "distribution", "~> 0.7.3"
  spec.add_runtime_dependency "parallel", "~> 1.6.1"

  # This gem will work with Ruby 2.1 or newer, because it uses required 
  # keyword arguments (i.e. keyword arguments without default values)
  spec.required_ruby_version = '>= 2.1'
end
