local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local data2 = {}

local result = {}
local unpack = unpack and unpack or table.unpack

function data2.new(n, dataset_name, opt_)
   opt_ = opt_ or {}
   local self = {}
   for k,v in pairs(data) do
      self[k] = v
   end

   local donkey_file
   if dataset_name == 'imagenet' or dataset_name == 'folder' then
       donkey_file = 'donkey_folder_2.lua'
   else
      error('Unknown dataset: ' .. dataset_name)
   end

   if n > 0 then
      local options = opt_
      self.threads = Threads(n,
                             function() require 'torch' end,
                             function(idx)
                                opt = options
                                tid = idx
                                local seed = (opt.manualSeed and opt.manualSeed or 0) + idx
                                torch.manualSeed(seed)
                                torch.setnumthreads(1)
                                print(string.format('Starting donkey with id: %d seed: %d', tid, seed))
                                assert(options, 'options not found')
                                assert(opt, 'opt not given')
                                print(opt)
                                paths.dofile(donkey_file)
                             end
      )
      print('DS: data2: successfuly loading the datasets')
   else
      if donkey_file then paths.dofile(donkey_file) end
      self.threads = {}
      function self.threads:addjob(f1, f2) f2(f1()) end
      function self.threads:dojob() end
      function self.threads:synchronize() end
      
   end

   print('DS2: successfuly loading the datasets')
   local nSamples = 0
   self.threads:addjob(function() return trainLoader:size() end,
         function(c) nSamples = c end)
   self.threads:synchronize()
   self._size = nSamples

   for i = 1, n do
      self.threads:addjob(self._getFromThreads,
                          self._pushResult)
   end

   return self
end

function data2._getFromThreads()
   assert(opt.batchSize, 'opt.batchSize not found')
   return trainLoader:sample(opt.batchSize)
end

function data2._pushResult(...)
   local res = {...}
   if res == nil then
      self.threads:synchronize()
   end
   result[1] = res
end



function data2:getBatch()
   -- queue another job
   self.threads:addjob(self._getFromThreads, self._pushResult)
   self.threads:dojob()
   local res = result[1]
   result[1] = nil
   if torch.type(res) == 'table' then
      return unpack(res)
   end
   print(type(res))
   return res
end

function data2:size()
   return self._size
end

return data2
