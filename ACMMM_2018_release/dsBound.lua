dsBound, _ = torch.class('nn.dsBound', 'nn.Module')
function dsBound:updateOutput(inputTable)
   assert(torch.type(inputTable) == 'table')

   --  traX, traY,   scaX, scaY
   local input, sca = unpack(inputTable)
   self.output = self.output or input.new()
   self.output:resizeAs(input):copy(input)

   for i = 1,input:size(1) do
      local c1 = sca:narrow(1,i,1)[1][1]
      local bound1 = (1.0-c1) / c1

      local c2 = sca:narrow(1,i,1)[1][2]
      local bound2 = (1.0-c2) / c2

      self.output:narrow(1,i,1):narrow(2,1,1):clamp(-bound1, bound1)
      self.output:narrow(1,i,1):narrow(2,2,1):clamp(-bound2, bound2)
   end
   return self.output --self.output
end

function dsBound:updateGradInput(inputTable, gradOutput)
   assert(torch.type(inputTable) == 'table')
   local input, sca = unpack(inputTable)
   --TraX, TraY, 2-dim
   self.gradInput1 = self.gradInput1 or input.new()
   self.gradInput1:resizeAs(gradOutput):copy(gradOutput)
   -- scaX, scaY, 2-dim
   self.gradInput2 = self.gradInput2 or input.new()
   self.gradInput2:resizeAs(gradOutput):copy(gradOutput)

   self.multiplier = self.multiplier or input.new()
   self.multiplier:resizeAs(gradOutput:narrow(1,1,1)):fill(1)

   --print('gradInput2 is initiliazed as ', self.gradInput2)

   for i = 1,input:size(1) do
      local c1 = sca:narrow(1,i,1)[1][1]
      local c2 = sca:narrow(1,i,1)[1][2]
      local input1 = input:narrow(1,i,1)[1][1] -- TraX, input
      local input2 = input:narrow(1,i,1)[1][2] -- TraY, input

      ---------------------------------------------------------
      ----------Working on getting d/dx, x = traslation--------
      ---------------------------------------------------------
      self.multiplier:fill(1)

      local thisBound1 = (1.0-c1) / c1
      if input1 < -thisBound1 or input1 > thisBound1 then
         self.multiplier[1][1] = 0
      end

      local thisBound2 = (1.0-c2) / c2
      if input2 < -thisBound2 or input2 > thisBound2 then
         self.multiplier[1][2] = 0
      end
      self.gradInput1:narrow(1,i,1):cmul(self.multiplier)


      ---------------------------------------------------------
      ----------Working on getting d/dx, x = scale 1    -------
      ---------------------------------------------------------
      self.multiplier:fill(0)
      if input1 < -thisBound1 then
         self.multiplier[1][1] =  1.0 / (c1^2)
         --print('setting multiplier x1, too large --- traslation')
         --print(self.multiplier)
      elseif input1 > thisBound1 then
         self.multiplier[1][1] = -1.0 / (c1^2)
         --print('setting multiplier x1, too +++ large traslation')
         --print(self.multiplier)
      else
         self.multiplier[1][1] = 0
      end

      ---------------------------------------------------------
      ----------Working on getting d/dx, x = scale 2    -------
      ---------------------------------------------------------
      if input2 < -thisBound2 then
         --print('setting multiplier x2, too large --- traslation')
         self.multiplier[1][2] =  1.0 / (c2^2)
         --print(self.multiplier)
      elseif input2 > thisBound2 then
         --print('setting multiplier x2, too large +++ traslation')
         self.multiplier[1][2] = -1.0 / (c2^2)
      else
         self.multiplier[1][2] = 0
      end
      self.gradInput2:narrow(1,i,1):cmul(self.multiplier)
   end
   --print(self.gradInput2)

   -- local myNorm = self.gradInput1:norm()
   -- if myNorm > 1.0 then
   --    self.gradInput1:mul(1.0 / myNorm)
   -- end

   -- local myNorm = self.gradInput2:norm()
   -- if myNorm > 1.0 then
   --    self.gradInput2:mul(1.0 / myNorm)
   -- end
   
   self.gradInput = {torch.Tensor(), torch.Tensor()}
   self.gradInput[1] = self.gradInput1 -- tra
   self.gradInput[2] = self.gradInput2 -- sca



   return self.gradInput
end