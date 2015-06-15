# Not intended for direct use by the user of the MixedModels gem.
#
# === Usage
#
# intercept = lmm_variable(:intercept)
# x         = lmm_variable(:x)
# y         = lmm_variable(:y)
# u         = lmm_variable(:u)
# w         = lmm_variable(:w)
# z         = intercept + x + y + x*y + 
#             (intercept + x + y + x*y| u) + 
#             (intercept | w)
# z.to_input_for_lmm_from_daru
# z.fixed_effects # => [:intercept, :x, :y, [:x, :y]]
# z.random_effects # => [[:intercept, :x, :y, [:x, :y]], [:intercept]]
# z.grouping # => [:u, :w]
#
class LMMFormula
  def initialize(content)
    @content = content
  end

  attr_reader :content, :fixed_effects, :random_effects, :grouping

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

  def to_input_for_lmm_from_daru
    @fixed_effects  = Array.new
    @random_effects = Array.new
    @grouping       = Array.new
    @content.each do |item|
      if item.is_a?(Symbol) then
        @fixed_effects.push(item)
      elsif item.is_a?(Array) then
        c = item.clone # in order to keep @content unchanged
        if c[0] == "interaction_effect" then
          c.shift
          raise "bi-variate interaction effects allowed only" unless c.length == 2
          @fixed_effects.push(c)
        elsif c[0] == "random_effect" then
          c.shift
          @grouping.push(c.pop)
          ran_ef = Array.new
          c.each do |cc|
            if cc.is_a?(Symbol) then
              ran_ef.push(cc)
            elsif cc.is_a?(Array) then
              if cc[0] == "interaction_effect" then
                cc.shift
                raise "bi-variate interaction effects allowed only" unless cc.length == 2
                ran_ef.push(cc)
              else
                raise "invalid formulation of LMMFormula.content"
              end
            else
              raise "invalid formulation of LMMFormula.content"
            end
          end
          @random_effects.push(ran_ef)
        else
          raise "invalid formulation of LMMFormula.content"
        end
      else
        raise "invalid formulation of LMMFormula.content"
      end
    end
  end
end

def lmm_variable(x)
  LMMFormula.new([x])
end
