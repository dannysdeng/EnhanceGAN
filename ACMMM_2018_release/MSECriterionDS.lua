local MSECriterionDS, parent = torch.class('nn.MSECriterionDS', 'nn.Criterion')

function MSECriterionDS:__init(sizeAverage)
   parent.__init(self)
   if sizeAverage ~= nil then
     self.sizeAverage = sizeAverage
   else
     self.sizeAverage = true
   end
end

function MSECriterionDS:updateOutput(input, target)
   self.comparison = self.comparison or input.new()
   self.comparison:resizeAs(input)
   self.output_tensor = self.output_tensor or input.new(1)
   input.THNN.MSECriterion_updateOutput(
      input:cdata(),
      target:cdata(),
      self.output_tensor:cdata(),
      self.sizeAverage
   )
   self.output = self.output_tensor[1]
   return self.output
end

function MSECriterionDS:updateGradInput(input, target)
   input.THNN.MSECriterion_updateGradInput(
      input:cdata(),
      target:cdata(),
      self.gradInput:cdata(),
      self.sizeAverage
   )
   temp = input - target
   self.comparison:copy(temp:lt(0))
   -- We want input is smaller than target, so if it is true, then no gradient
   self.gradInput:cmul(self.comparison)
   return self.gradInput
end