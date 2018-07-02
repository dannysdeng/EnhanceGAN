--[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--
local matio = require 'matio'
require 'image'
paths.dofile('dataset.lua')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
-------- COMMON CACHES and PATHS
-- Check for existence of opt.data
opt.data = os.getenv('DATA_ROOT') 
opt.data2 = os.getenv('DATA_ROOT_2') 
if not paths.dirp(opt.data) then
    error('Did not find directory1: ', opt.data)
end
if not paths.dirp(opt.data2) then
    error('Did not find directory2: ', opt.data2)
end

-- a cache file of the training metadata (if doesnt exist, will be created)
local cache = "cache"
local cache_prefix = opt.data:gsub('/', '_')
os.execute('mkdir -p cache')
local trainCache = paths.concat(cache, cache_prefix .. '_trainCache.t7')
local trainCache2 = paths.concat(cache, cache_prefix .. '_trainCache2.t7')


--------------------------------------------------------------------------------------------
local loadSize   = {3, opt.loadSize}
local sampleSize = {3, opt.fineSize}

local function loadImage(path)
   local input = image.load(path, 3, 'float')
   input = image.rgb2lab(input)
   for i=1,3 do
      input[i]:add(-meanstd.mean[i])
      input[i]:div(meanstd.std[i])
   end
   -- find the smaller dimension, and resize it to loadSize[2] (while keeping aspect ratio)
   local iW = input:size(3)
   local iH = input:size(2)
   if iW < iH then
      input = image.scale(input, loadSize[2], loadSize[2] * iH / iW)
   else
      input = image.scale(input, loadSize[2] * iW / iH, loadSize[2])
   end
   return input
end

-- channel-wise mean and std. Calculate or load them from disk later in the script.
meanstd = {
   mean = { 0,-128,-128 },
   std = { 100,255,255 },
   scale = 1,
}

--------------------------------------------------------------------------------
-- Hooks that are used for each image that is loaded

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, path)
   error('This should be disabled')
   collectgarbage()
   local input = loadImage(path)
   local iW = input:size(3)
   local iH = input:size(2)

   -- do random crop
   local oW = sampleSize[2];
   local oH = sampleSize[2]
   local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
   local w1 = math.ceil(torch.uniform(1e-2, iW-oW))

   local out = image.crop(input, w1, h1, w1 + oW, h1 + oH)
   assert(out:size(2) == oW)
   assert(out:size(3) == oH)
   -- do hflip with probability 0.5
   if torch.uniform() > 0.5 then out = image.hflip(out); end   
   --out:mul(2):add(-1) -- make it [0, 1] -> [-1, 1]
   return out
end

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook2 = function(self, path, path2)
   collectgarbage()
   local input = loadImage(path)
   local iW = input:size(3)
   local iH = input:size(2)

   -- do random crop
   local oW = sampleSize[2];
   local oH = sampleSize[2]
   local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
   local w1 = math.ceil(torch.uniform(1e-2, iW-oW))

   local out = image.crop(input, w1, h1, w1 + oW, h1 + oH)
   assert(out:size(2) == oW)
   assert(out:size(3) == oH)
   -- do hflip with probability 0.5
   if torch.uniform() > 0.5 then out = image.hflip(out); end
   --out:mul(2):add(-1) -- make it [0, 1] -> [-1, 1]

   -- out is 3x224x224

   local input = loadImage(path2)
   local out2 = image.scale(input, oW, oH, 'bilinear')
   assert(out:size(2) == oW)
   assert(out:size(3) == oH)
   -- do hflip with probability 0.5
   if torch.uniform() > 0.5 then out = image.hflip(out); end
   --out:mul(2):add(-1) -- make it [0, 1] -> [-1, 1]
   
   return torch.cat(out, out2, 1)
end

--------------------------------------
-- trainLoader
if paths.filep(trainCache) then
   print('Loading train metadata from cache')
   trainLoader = torch.load(trainCache)

   print('Loading KNN matrix for FAKE data in trainLoader2')
   local KNN_precomputed = matio.load('../train_data/KNN_Matrix.mat')
   trainLoader.KNN_Matrix = KNN_precomputed.KNN_matrix

   
   trainLoader.sampleHookTrain = trainHook
   trainLoader.loadSize = {3, opt.loadSize, opt.loadSize}
   trainLoader.sampleSize = {3, sampleSize[2], sampleSize[2]}
else
   print(string.format('Creating train metadata: %s', opt.data))
   trainLoader = dataLoader{
      paths = {opt.data},
      loadSize = {3, loadSize[2], loadSize[2]},
      sampleSize = {3, sampleSize[2], sampleSize[2]},
      split = 100,
      verbose = true
   }
   torch.save(trainCache, trainLoader)
   print('saved metadata cache at', trainCache)
   trainLoader.sampleHookTrain = trainHook
end

if paths.filep(trainCache2) then
   print('Loading train metadata from cache')
   trainLoader2 = torch.load(trainCache2)


   print('Loading KNN matrix for FAKE data in trainLoader2')
   local KNN_precomputed = matio.load('../train_data/KNN_Matrix.mat')
   trainLoader2.KNN_Matrix = KNN_precomputed.KNN_matrix

   trainLoader2.sampleHookTrain = trainHook2
   trainLoader2.loadSize = {3, opt.loadSize, opt.loadSize}
   trainLoader2.sampleSize = {3, sampleSize[2], sampleSize[2]}
else
   print(string.format('Creating train metadata: %s', opt.data2))
   trainLoader2 = dataLoader{
      paths = {opt.data2},
      loadSize = {3, loadSize[2], loadSize[2]},
      sampleSize = {3, sampleSize[2], sampleSize[2]},
      split = 100,
      verbose = true
   }
   torch.save(trainCache2, trainLoader2)
   print('saved metadata cache2 at', trainCache2)
   trainLoader2.sampleHookTrain = trainHook2
end


collectgarbage()

-- do some sanity checks on trainLoader
do
   local class = trainLoader.imageClass
   local nClasses = #trainLoader.classes
   assert(class:max() <= nClasses, "class logic has error")
   assert(class:min() >= 1, "class logic has error")
end
