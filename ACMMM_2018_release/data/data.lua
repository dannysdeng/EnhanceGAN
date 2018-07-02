local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local data = {}
local result = {}
local unpack = unpack and unpack or table.unpack

function data.new(n, dataset_name, opt_)
   opt_ = opt_ or {}
   local self = {}
   for k,v in pairs(data) do
      self[k] = v
   end

   local donkey_file
   if dataset_name == 'imagenet' or dataset_name == 'folder' then
       donkey_file = 'donkey_folder.lua'
   elseif dataset_name == 'rrrrr' then
       donkey_file = 'donkey_rrrrr.lua'
   elseif dataset_name == 'lsun' then
       donkey_file = 'donkey_lsun.lua'
       if n > 6 then n = 6 end -- lmdb complains beyond 6 donkeys. wtf.
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
      self.threads2 = Threads(n,
                             function() require 'torch' end,
                             function(idx)
                                opt = options
                                tid = idx
                                local seed = torch.random(1, 10000) + idx
                                torch.manualSeed(seed)
                                torch.setnumthreads(1)
                                print(string.format('Starting donkey with id: %d seed: %d', tid, seed))
                                assert(options, 'options not found')
                                assert(opt, 'opt not given')
                                print(opt)
                                paths.dofile(donkey_file)
                             end
      )      
   else
      if donkey_file then paths.dofile(donkey_file) end
      self.threads = {}
      function self.threads:addjob(f1, f2) f2(f1()) end
      function self.threads:dojob() end
      function self.threads:synchronize() end
   end
   local nSamples = 0

   -- self.threads:addjob(function() return trainLoader:size() end,
   --                     function(c) nSamples = c end)  
   -- self.threads:synchronize()
   -- self._size = nSamples

   self.threads2:addjob(function() return trainLoader2:size() end,
                       function(c) nSamples = c end) 
   self.threads2:synchronize()
   self._size2 = nSamples

   -- For i = 1 to number of threads
   half_n = torch.floor(n/2)
   -- for i = 1, n do
   --    self.threads:addjob(self._getFromThreads,
   --                        self._pushResult)
   -- end
   for i = 1, n do
      self.threads2:addjob(self._getFromThreads2,
                          self._pushResult2)
   end 

   return self
end

function data._getFromThreads()
   assert(opt.batchSize, 'opt.batchSize not found')
   return trainLoader:sample(opt.batchSize)
end

function data._getFromThreads2()
   assert(opt.batchSize, 'opt.batchSize not found')
   return trainLoader2:sample(opt.batchSize)
end

function data._pushResult(...)
   local res = {...}
   if res == nil then
      self.threads:synchronize()
   end
   result[1] = res
end
function data._pushResult2(...)
   local res = {...}
   if res == nil then
      self.threads2:synchronize()
   end
   result[1] = res
end

function data:getBatch()
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

function data:getBatch2()
   -- queue another job
   self.threads2:addjob(self._getFromThreads2, self._pushResult2)
   self.threads2:dojob()
   local res = result[1]
   result[1] = nil
   if torch.type(res) == 'table' then
      return unpack(res)
   end
   print(type(res))
   return res
end

function data:size()
   return self._size2
end

function data:size2()
   return self._size2
end

return data
