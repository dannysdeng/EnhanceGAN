labFilter, _ = torch.class('nn.labFilter', 'nn.Module')
function labFilter:updateOutput(inputTable)
   assert(torch.type(inputTable) == 'table')
   local input, constantList = unpack(inputTable)
   self.output = self.output or input.new()
   self.output:resizeAs(input):copy(input)

   self.A = self.A or input.new()
   self.A:resizeAs(self.output:narrow(1,1,1):narrow(2,2,1))

   self.B = self.B or input.new()
   self.B:resizeAs(self.output:narrow(1,1,1):narrow(2,3,1)) 

   for i = 1,input:size(1) do
      local alpha = constantList:narrow(1,i,1)[1][1]
      local beta =  constantList:narrow(1,i,1)[1][2]
      local thisImg = self.output:narrow(1,i,1)
      self.A:copy(thisImg:narrow(2,2,1))
      self.B:copy(thisImg:narrow(2,3,1))
      local k = 1.0 / (1 - 2*alpha)
      local b = -k*alpha
      self.A:apply(function(m) 
        if m < alpha then
            return 0
        elseif m > 1-alpha then
            return 1
        else 
            return k*m + b
        end
      end
      )
      local k = 1.0 / (1 - 2*beta)
      local b = -k*beta
      self.B:apply(function(m) 
        if m < beta then
            return 0
        elseif m > 1-beta then
            return 1
        else 
            return k*m + b
        end
      end
      )
      thisImg:narrow(2,2,1):copy(self.A)
      thisImg:narrow(2,3,1):copy(self.B)
      self.output:narrow(1,i,1):copy(thisImg)    
   end
   return self.output
end

function labFilter:updateGradInput(inputTable, gradOutput)
   assert(torch.type(inputTable) == 'table')
   local input, constantList = unpack(inputTable)
   --Img
   self.gradInput1 = self.gradInput1 or input.new()
   self.gradInput1:resizeAs(gradOutput):copy(gradOutput)
   -- Constant alpha: X = alpha * x + b
   self.gradInput2 = self.gradInput2 or input.new()
   self.gradInput2:resizeAs(gradOutput:narrow(2,2,1)):copy(gradOutput:narrow(2,2,1))
   self.gradInput3 = self.gradInput3 or input.new()
   self.gradInput3:resizeAs(gradOutput:narrow(2,3,1)):copy(gradOutput:narrow(2,3,1))


   for i = 1,input:size(1) do
      local alpha = constantList:narrow(1,i,1)[1][1]
      local beta = constantList:narrow(1,i,1)[1][2]
      local thisImg = input:narrow(1,i,1)
      self.A:copy(thisImg:narrow(2,2,1))
      self.B:copy(thisImg:narrow(2,3,1))
      self.A:apply(function(m) 
        if m < alpha then
            return 0
        elseif m > 1 - alpha then
            return 0
        else 
            return (2*m - 1.0) / (1.0-2*alpha) / (1.0-2*alpha)
        end
      end
      )
      self.B:apply(function(m) 
        if m < beta then
            return 0
        elseif m > 1 - beta then
            return 0
        else 
            return (2*m - 1.0) / (1.0-2*beta) / (1.0-2*beta)
        end
      end
      )
      self.gradInput2:narrow(1,i,1):cmul(self.A)
      self.gradInput3:narrow(1,i,1):cmul(self.B)

      self.A:apply(function(m) 
        if m ~= 0 then 
            return 1 / (1 - 2*alpha)
        end
      end
      )
      -- b = (127-beta)*k-127
      self.B:apply(function(m) 
        if m ~= 0 then 
            return 1 / (1 - 2*beta)
        end
      end
      )
      
      thisImg:narrow(2,2,1):copy(self.A)
      thisImg:narrow(2,3,1):copy(self.B)      
      self.gradInput1:narrow(1,i,1):cmul(thisImg)
   end

   self.gradInput2:resize(self.gradInput2:size(1), 
                          self.gradInput2:size(2)*self.gradInput2:size(3)*self.gradInput2:size(4))
   self.gradInput3:resize(self.gradInput3:size(1), 
                          self.gradInput3:size(2)*self.gradInput3:size(3)*self.gradInput3:size(4))

   self.gradInput = {torch.Tensor(), torch.Tensor()}
   self.gradInput[1] = self.gradInput1
   self.gradInput[2] = torch.cat(torch.sum(self.gradInput2, 2), 
                                 torch.sum(self.gradInput3, 2),
                                 2)
   return self.gradInput
end