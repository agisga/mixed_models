# coding: utf-8
lib = File.expand_path('../lib', __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require 'mixed_models/version.rb'

Gem::Specification.new do |spec|
  spec.name          = "mixed_models"
  spec.version       = MixedModels::VERSION
  spec.authors       = ["agisga"]
  spec.email         = ["alexej.go@googlemail.com"]

  spec.summary       = %q{Mixed effects models in Ruby}
  spec.description   = %q{Fit statistical linear models with fixed and mixed (random) effects in Ruby}
  spec.homepage      = "https://github.com/agisga/mixed_models.git"
  spec.license       = "BSD-3-Clause"

  # Prevent pushing this gem to RubyGems.org by setting 'allowed_push_host', or
  # delete this section to allow pushing this gem to any host.
  if spec.respond_to?(:metadata)
    spec.metadata['allowed_push_host'] = "TODO: Set to 'http://mygemserver.com'"
  else
    raise "RubyGems 2.0 or newer is required to protect against public gem pushes."
  end

  spec.files         = `git ls-files -z`.split("\x0").reject { |f| f.match(%r{^(test|spec|features)/}) }
  spec.bindir        = "exe"
  spec.executables   = spec.files.grep(%r{^exe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]

  spec.add_development_dependency "bundler", "~> 1.9"
  spec.add_development_dependency "rake", "~> 10.0"
  spec.add_development_dependency "rspec", "~> 3.2"

  spec.add_dependency "nmatrix", "~> 0.1.0"
  spec.add_dependency "daru", "~> 0.1.0"
  spec.add_dependency "distribution", "~> 0.7.3"

  # This gem will work with 2.0 or greater...
  spec.required_ruby_version = '>= 2.0'
end
