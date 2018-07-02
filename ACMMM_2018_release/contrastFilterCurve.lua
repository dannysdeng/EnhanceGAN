contrastFilterCurve, _ = torch.class('nn.contrastFilterCurve', 'nn.Module')
function contrastFilterCurve:updateOutput(inputTable)
   local input, constantList = unpack(inputTable)
   self.output = self.output or input.new()
   self.output:resizeAs(input):copy(input)

   for i = 1,input:size(1) do
      local a = constantList:narrow(1,i,1)[1][1]
      local b =  constantList:narrow(1,i,1)[1][2]
      local p =  constantList:narrow(1,i,1)[1][3]
      local q =  constantList:narrow(1,i,1)[1][4]

      local thisImg = self.output:narrow(1,i,1):narrow(2,1,1)
      local k1 = a / ((a - b)^(1/p))
      local k2 = a / ((a - b)^(1/q))
      thisImg:apply(function(m) 
          if m < b then 
              return 0
          elseif m >= b and m < a then
              return k1* (         (m -      b)^(1/p)    ) +  0
          elseif m >= a and m < 1-a then
              return m
          elseif m >= 1-a and m < 1-b then
              return k2* (         (m - (1-a))^(1/q)    ) + 1-a
          else
              return 1
          end
        end
        )        
      self.output:narrow(1,i,1):narrow(2,1,1):copy(thisImg)    
   end
   return self.output
end

function contrastFilterCurve:updateGradInput(inputTable, gradOutput)
   assert(torch.type(inputTable) == 'table')
   local input, constantList = unpack(inputTable)
   --Img
   self.gradInput1 = self.gradInput1 or input.new()
   self.gradInput1:resizeAs(gradOutput):copy(gradOutput)
   -- Constant a: X = a * x + b
   self.gradInput2 = self.gradInput2 or input.new()
   self.gradInput2:resizeAs(gradOutput:narrow(2,1,1)):copy(gradOutput:narrow(2,1,1))
   self.gradInput3 = self.gradInput3 or input.new()
   self.gradInput3:resizeAs(gradOutput:narrow(2,1,1)):copy(gradOutput:narrow(2,1,1))
   self.gradInput4 = self.gradInput4 or input.new()
   self.gradInput4:resizeAs(gradOutput:narrow(2,1,1)):copy(gradOutput:narrow(2,1,1))
   self.gradInput5 = self.gradInput5 or input.new()
   self.gradInput5:resizeAs(gradOutput:narrow(2,1,1)):copy(gradOutput:narrow(2,1,1))
   local thisImg = self.output:narrow(1,1,1):narrow(2,1,1):clone()
   for i = 1,input:size(1) do
      local a = constantList:narrow(1,i,1)[1][1]
      local b =  constantList:narrow(1,i,1)[1][2]
      local p =  constantList:narrow(1,i,1)[1][3]
      local q =  constantList:narrow(1,i,1)[1][4]

      thisImg:copy(self.output:narrow(1,i,1):narrow(2,1,1))
      local k1 = a / ((a - b)^(1/p))
      local k2 = a / ((a - b)^(1/q))

      -- df / dx
      thisImg:apply(function(m) 
          if m < b then 
              return 0
          elseif m >= b and m < a then
              return k1/p * (        m -b)^(1/p - 1)
          elseif m >= a and m < 1-a then
              return 1
          elseif m >= 1-a and m < 1-b then
              return k2/q * (a + m - 1  )^(1/q - 1)
          else
              return 0
          end
        end
       )   
       self.gradInput1:narrow(1,i,1):narrow(2,1,1):cmul(thisImg)   -- df/dx



      -- df / da
      thisImg:copy(self.output:narrow(1,i,1):narrow(2,1,1))
      thisImg:apply(function(m) 
          if m <= b then 
              return 0
          elseif m > b and m < a then
              return             (a-b)^(-1/p    )*(m - b )^(1/p) - a/p * (a - b)^(-1/p - 1) * (m - b) ^ (1/p)
          elseif m >= a and m <= 1-a then
              return 0
          elseif m > 1-a and m < 1-b then
              return -a/q * (a-b)^(-1/q - 1)*(a+m-1)^(1/q) + a/q * (a - b)^(-1/q)*(a+m-1)^(1/q -1) + (a-b)^(-1/q)*(a+m-1)^(1/q) - 1
          else
              return 0
          end
        end
       )   
       self.gradInput2:narrow(1,i,1):cmul(thisImg)  ----  df/da


       -- df / db
      thisImg:copy(self.output:narrow(1,i,1):narrow(2,1,1))
      thisImg:apply(function(m) 
          if m <= b then 
              return 0
          elseif m > b and m < a then
              return a/p * (a-b)^(-1/p - 1)*(m-b)^(1/p) - a/p * (a - b)^(-1/p)*(m-b)^(1/p - 1)
          elseif m >= a and m <= 1-a then
              return 0
          elseif m > 1-a and m < 1-b then
              return a/q * (a-b)^( -1 - 1/q ) * (a + m - 1)^(1/q)
          else
              return 0
          end
        end
       )   
       self.gradInput3:narrow(1,i,1):cmul(thisImg)  ---- df/db

       -- df / dp
      thisImg:copy(self.output:narrow(1,i,1):narrow(2,1,1))
      thisImg:apply(function(m) 
          if m <= b then 
              return 0
          elseif m > b and m < a then
              return a/(p^2)*(a-b)^(-1/p)*(m-b)^(1/p)*   (   torch.log(a-b) - torch.log(m - b)      )
          elseif m >= a and m <= 1-a then
              return 0
          elseif m > 1-a and m < 1-b then
              return 0
          else
              return 0
          end
        end
       )   
       self.gradInput4:narrow(1,i,1):cmul(thisImg) 

       -- df / dq
      thisImg:copy(self.output:narrow(1,i,1):narrow(2,1,1))
      thisImg:apply(function(m) 
          if m < b then 
              return 0
          elseif m >= b and m < a then
              return 0
          elseif m >= a and m <= 1-a then
              return 0
          elseif m > 1-a and m < 1-b then
              return a / (q^2) * (a-b)^(-1/q)*(a + m - 1)^(1/q) * (  torch.log(a - b) - torch.log(a + m - 1)    )
          else
              return 0
          end
        end
       )   
       self.gradInput5:narrow(1,i,1):cmul(thisImg)        
   end

   self.gradInput2:resize(self.gradInput2:size(1), 
                          self.gradInput2:size(2)*self.gradInput2:size(3)*self.gradInput2:size(4))
   self.gradInput3:resize(self.gradInput3:size(1), 
                          self.gradInput3:size(2)*self.gradInput3:size(3)*self.gradInput3:size(4))
   self.gradInput4:resize(self.gradInput4:size(1), 
                          self.gradInput4:size(2)*self.gradInput4:size(3)*self.gradInput4:size(4))
   self.gradInput5:resize(self.gradInput5:size(1), 
                          self.gradInput5:size(2)*self.gradInput5:size(3)*self.gradInput5:size(4))                             

   self.gradInput = {torch.Tensor(), torch.Tensor()}
   self.gradInput[1] = self.gradInput1
   self.gradInput[2] = torch.cat(
                       torch.cat(torch.sum(self.gradInput2, 2), 
                                 torch.sum(self.gradInput3, 2),
                                 2),
                       torch.cat(torch.sum(self.gradInput4, 2),
                                 torch.sum(self.gradInput5, 2),
                                 2),
                       2)
   return self.gradInput
end