#Copyright (c) 2015 Alexej Gossmann

module MixedModels
  # Defines a syntax to specify and store a model formula.
  # The stored formula can be transformed into Arrays that can be used
  # as input to LMM#from_daru.
  # Not intended for direct use by the user of the MixedModels gem.
  #
  # === Usage
  #
  #   intercept = MixedModels::lmm_variable(:intercept)
  #   x         = MixedModels::lmm_variable(:x)
  #   y         = MixedModels::lmm_variable(:y)
  #   u         = MixedModels::lmm_variable(:u)
  #   w         = MixedModels::lmm_variable(:w)
  #   z         = intercept + x + y + x*y + 
  #               (intercept + x + y + x*y| u) + 
  #               (intercept | w)
  #   input     = z.to_input_for_lmm_from_daru
  #   input[:fixed_effects] # => [:intercept, :x, :y, [:x, :y]]
  #   input[:random_effects] # => [[:intercept, :x, :y, [:x, :y]], [:intercept]]
  #   input[:grouping] # => [:u, :w]
  #
  class LMMFormula
    def initialize(content)
      @content = content
    end

    attr_reader :content

    def +(x)
      LMMFormula.new(@content + x.content)
    end

    def *(x)
      raise "can only call if both operands are single variables" if content.size !=1 || x.content.size !=1
      LMMFormula.new([["interaction_effect", @content[0], x.content[0]]])
    end

    def |(x)
      LMMFormula.new([["random_effect"] + @content + x.content])
    end

    # Transform +@content+ into a Hash containing multiple Arrays, which can be used as
    # input to LMM#from_daru
    #
    # === Usage
    #
    #  intercept = MixedModels::lmm_variable(:intercept)
    #  x         = MixedModels::lmm_variable(:x)
    #  u         = MixedModels::lmm_variable(:u)
    #  y         = intercept + x + (intercept + x | u)
    #  input     = y.to_input_for_lmm_from_daru
    #  input[:fixed_effects] # => [:intercept, :x]
    #  input[:random_effects] # => [[:intercept, :x]]
    #  input[:grouping] # => [:u]
    #
    def to_input_for_lmm_from_daru
      lmm_from_daru_input = Hash.new
      lmm_from_daru_input[:fixed_effects]  = Array.new
      lmm_from_daru_input[:random_effects] = Array.new
      lmm_from_daru_input[:grouping]       = Array.new
      @content.each do |item|
        if item.is_a?(Symbol) then
          lmm_from_daru_input[:fixed_effects].push(item)
        elsif item.is_a?(Array) then
          c = item.clone # in order to keep @content unchanged
          if c[0] == "interaction_effect" then
            c.shift
            raise "bi-variate interaction effects allowed only" unless c.length == 2
            lmm_from_daru_input[:fixed_effects].push(c)
          elsif c[0] == "random_effect" then
            c.shift
            lmm_from_daru_input[:grouping].push(c.pop)
            ran_ef = Array.new
            c.each do |cc|
              case
              when cc.is_a?(Symbol)
                ran_ef.push(cc)
              when cc.is_a?(Array) && cc[0] == "interaction_effect"
                cc.shift
                raise "bi-variate interaction effects allowed only" unless cc.length == 2
                ran_ef.push(cc)
              else
                raise "invalid formulation of random effects in LMMFormula"
              end
            end
            lmm_from_daru_input[:random_effects].push(ran_ef)
          else
            raise "invalid formulation of LMMFormula.content"
          end
        else
          raise "invalid formulation of LMMFormula.content"
        end
      end
      return lmm_from_daru_input
    end
  end

  def MixedModels.lmm_variable(x)
    raise(ArgumentError, "Variable name must by a Symbol") unless x.is_a? Symbol
    LMMFormula.new([x])
  end
end
