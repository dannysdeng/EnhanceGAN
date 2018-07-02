contrastFilter, _ = torch.class('nn.contrastFilter', 'nn.Module')
function contrastFilter:updateOutput(inputTable)
   local input, constantList = unpack(inputTable)
   self.output = self.output or input.new()
   self.output:resizeAs(input):copy(input)

   for i = 1,input:size(1) do
      local alpha = constantList:narrow(1,i,1)[1][1]
      local beta =  constantList:narrow(1,i,1)[1][2]
      local thisImg = self.output:narrow(1,i,1):narrow(2,1,1)
      thisImg:apply(function(m) 
          if alpha * m + beta > 1 then 
              return 1
          else
              return alpha*m + beta
          end
        end
        )        
      self.output:narrow(1,i,1):narrow(2,1,1):copy(thisImg)    
   end
   return self.output
end

function contrastFilter:updateGradInput(inputTable, gradOutput)
   assert(torch.type(inputTable) == 'table')
   local input, constantList = unpack(inputTable)
   --Img
   self.gradInput1 = self.gradInput1 or input.new()
   self.gradInput1:resizeAs(gradOutput):copy(gradOutput)
   -- Constant alpha: X = alpha * x + b
   self.gradInput2 = self.gradInput2 or input.new()
   self.gradInput2:resizeAs(gradOutput:narrow(2,1,1)):copy(gradOutput:narrow(2,1,1))
   self.gradInput3 = self.gradInput3 or input.new()
   self.gradInput3:resizeAs(gradOutput:narrow(2,1,1)):copy(gradOutput:narrow(2,1,1))


   for i = 1,input:size(1) do
      local alpha = constantList:narrow(1,i,1)[1][1]
      local beta = constantList:narrow(1,i,1)[1][2]
      local thisImg = input:narrow(1,i,1):narrow(2,1,1)

      
       thisImg:apply(function(m) 
            return m
            -- if alpha * m + beta > 1 then 
            --     return 1
            -- else
            --     return m
            -- end
            end
          )
       self.gradInput2:narrow(1,i,1):cmul(thisImg)  ----  df/da

       --thisImg:fill(1)
       thisImg:apply(function(m) 
          return 1
          --   if m == 0 then 
          --       return 1
          --   else
          --       return 1
          --   end
          end
          )
       self.gradInput3:narrow(1,i,1):cmul(thisImg)  ---- df/db

       -- thisImg:fill(1)
       thisImg:apply(function(m) 
          return alpha
          --   if m == 0 then 
          --       return 1
          --   else
          --       return alpha
          --   end
          end
          ) 
      self.gradInput1:narrow(1,i,1):narrow(2,1,1):cmul(thisImg)   -- df/dx
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