local TopK, parent = torch.class('nn.TopK', 'nn.Module')

function TopK:__init(k, dimension, dir, sort)
  parent.__init(self)
  self.k = k or 1
  self.dimension = dimension or 1
  self.dir = dir or false
  self.sort = sort or false
  self:_lazyInit()
end

function TopK:_lazyInit()
  self._indices = self._indices or torch.CudaLongTensor()
  -- (torch.type(self.output) == 'torch.CudaTensor' and torch.CudaLongTensor() or torch.LongTensor())
  if torch.type(self._indices) == 'torch.CudaTensor' then
      self._indices = self._indices:cudaLong()
  end
end

function TopK:updateOutput(inputTable)
    assert(torch.type(inputTable) == 'table')
    --  traX, traY,   scaX, scaY
    local input, probMap = unpack(inputTable)
    self:_lazyInit()

    local dimension = self.dimension
    local k = self.k

    
    self.output2 = self.output2 or (torch.type(input) == 'torch.CudaTensor' and torch.CudaTensor() or torch.Tensor())
    torch.topk(self.output2, self._indices, probMap, k, dimension, self.dir, self.sort)


    self.output1 = self.output1 or (torch.type(input) == 'torch.CudaTensor' and torch.CudaTensor() or torch.Tensor())
    self.output1:resize(input:size(1), input:size(2), k)-- 2x4x3
    for i = 1,input:size(1) do 
      self.output1:narrow(1,i,1):copy(   input:narrow(1,i, 1):index(3,self._indices:narrow(1,i,1):squeeze():long())         )
    end

    -- print('self.indices is ')
    -- print(self._indices)
    self.output = {torch.Tensor(), torch.Tensor()}
    self.output[1] = self.output1
    self.output[2] = self.output2
  return self.output
end

function TopK:updateGradInput(inputTable, gradOutputTable)
  self:_lazyInit()

  local input, probMap = unpack(inputTable)
  local grad1, grad2 = unpack(gradOutputTable)

  local dimension = self.dimension

  self.gradInput1 = self.gradInput1 or (torch.type(grad1) == 'torch.CudaTensor' and torch.CudaTensor() or torch.Tensor())
  self.gradInput2 = self.gradInput2 or (torch.type(grad2) == 'torch.CudaTensor' and torch.CudaTensor() or torch.Tensor())


  self.gradInput2:resizeAs(probMap):zero():scatter(dimension, self._indices, grad2) -- [probMap, easy]


  local sameIndex = torch.expand(self._indices, input:size(1),input:size(2),self._indices:size(3)) -- [Resize first]
  self.gradInput1:resizeAs(input):zero():scatter(dimension, sameIndex, grad1) -- self.indices = 2,1,3

  self.gradInput = {torch.Tensor(), torch.Tensor()}
  self.gradInput[1] = self.gradInput1
  self.gradInput[2] = self.gradInput2
  return self.gradInput
end

-- function TopK:type(type, tensorCache)
--   -- torch.max expects a LongTensor as indices, whereas cutorch.max expects a CudaTensor.
--   if type == 'torch.CudaTensor' then
--     parent.type(self, type, tensorCache)
--   else
--     -- self._indices must be a LongTensor. Setting it to nil temporarily avoids
--     -- unnecessary memory allocations.
--     local indices
--     indices, self._indices = self._indices, nil
--     parent.type(self, type, tensorCache)
--     self._indices = indices and indices:long() or nil
--   end
--   return self
-- end

function TopK:clearState()
  print('We are clearing TopK.lua module, removing _indices')
  nn.utils.clear(self, '_indices')
  return parent.clearState(self)
end