lightingContrast, _ = torch.class('nn.lightingContrast', 'nn.Module')

function lightingContrast:updateOutput(inputTable)
   assert(torch.type(inputTable) == 'table')
   input, constantList = unpack(inputTable)
   self.output = self.output or input.new()
   self.output:resizeAs(input):copy(input)
   for i = 1,input:size(1) do
      local alpha = constantList:narrow(1,i,1)[1][1]
      local beta = constantList:narrow(1,i,1)[1][2]

      

      self.output:narrow(1, i, 1):mul(alpha):add(-128/255*alpha+128/255):add(beta)
   end
   return self.output
end

function lightingContrast:updateGradInput(inputTable, gradOutput)
   assert(torch.type(inputTable) == 'table')
   input, constantList = unpack(inputTable)
   --Img
   self.gradInput1 = self.gradInput1 or input.new()
   self.gradInput1:resizeAs(gradOutput)
   self.gradInput1:copy(gradOutput)
   for i = 1,input:size(1) do
      local alpha = constantList:narrow(1,i,1)[1][1]
      local beta = constantList:narrow(1,i,1)[1][2]
      self.gradInput1:mul(alpha)
   end
   -- Constant
   self.gradInput2 = self.gradInput2 or input.new()
   self.gradInput2:resizeAs(input):copy(input)
   self.gradInput2:resize(self.gradInput2:size(1), 
                          self.gradInput2:size(2)*self.gradInput2:size(3)*self.gradInput2:size(4))

   self.gradInput3 = self.gradInput3 or torch.ones(input:size(1), 1)
   self.gradInput3:cuda()

   self.gradInput = {}
   self.gradInput[1] = self.gradInput1
   self.gradInput[2] = torch.cat(torch.sum(self.gradInput2, 2), 
end