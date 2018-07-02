require 'torch'
require 'nn'
require 'optim'
require 'cudnn'
require 'image';
require 'tds'
--require 'lmdb'
--require 'ffi'
require 'nngraph'
require 'stn'
require 'dpnn'

matio = require 'matio'
meanstd = {
   mean = { 0,-128,-128 },
   std = { 100,255,255 },
   scale = 1,
}

addDimLayer = nn.Unsqueeze(1)
addDimLayer:cuda()
function maskColor(M)
    if M:nDimension() == 3 then
        M = addDimLayer:forward(M)
    end  
    -- M is of size batchSize x 3 x 224 x 224
    M:narrow(2,2,2):fill(0)
    return M
end

function scandir(directory)
    local i, t, popen = 0, {}, io.popen
    local pfile = popen('ls -a "'..directory..'"')
    for filename in pfile:lines() do
        i = i + 1
        t[i] = filename
    end
    pfile:close()
    return t
end
function getRGBback(LAB_img_batch)
    if LAB_img_batch:nDimension() == 3 then
        LAB_img_batch = nn.Unsqueeze(1):forward(LAB_img_batch)
    end  
    for i = 1,LAB_img_batch:size(1) do
      thisImg = LAB_img_batch[i]
      thisImg = thisImg:squeeze():float()
      for channel=1,3 do
          thisImg[channel]:mul(meanstd.std[channel])
          thisImg[channel]:add(meanstd.mean[channel])                
      end               
      LAB_img_batch[i]:copy(image.lab2rgb(thisImg))
  end
  return LAB_img_batch
end

ds_verbose = false

opt = {
   dataset = 'lsun',       -- imagenet / lsun / folder
   batchSize = 8,
   loadSize = 256,
   fineSize = 224,
   nz = 100,               -- #  of dim for Z
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nThreads = 4,           -- #  of data loading threads to use
   niter = 500,             -- #  of iter at starting learning rate
   lr =  0.00005, -- 0.000001, --0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'fullconv_KNN_2',
   noise = 'normal',       -- uniform / normal
   rot = false,       -- uniform / normal
   tra = true,
   sca = true,
   locnet = '',
   inputSize = 224,
   inputChannel = 3,
   no_cuda = false,
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data =  DataLoader.new(opt.nThreads, opt.dataset, opt)



print("Dataset1: " .. opt.dataset, " Size: ", data:size())
print("Dataset2: " .. opt.dataset, " Size: ", data:size2())
----------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local nc = 3
local nz = opt.nz    -- #  of dim for Z
local ndf = opt.ndf  -- #  of gen filters in first conv layer
local ngf = opt.ngf  -- #  of discrim filters in first conv layer
local real_label = 1
local fake_label = 0

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = cudnn.SpatialConvolution
local SpatialFullConvolution = cudnn.SpatialFullConvolution


   -- nz = 100,               -- #  of dim for Z
   -- ngf = 64,               -- #  of gen filters in first conv layer
   -- ndf = 64,               -- #  of discrim filters in first conv layer
--local netG = nn.Sequential()
local networks = {}
-- These are the basic modules used when creating any macro-module
-- Can be modified to use for example cudnn
networks.modules = {}
networks.modules.convolutionModule = cudnn.SpatialConvolutionMM
networks.modules.poolingModule = cudnn.SpatialMaxPooling
networks.modules.nonLinearityModule = cudnn.ReLU

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization


local alphaA_grad_index = 14
local alphaA_output_index = 16

local curveA_grad_index = 19
local curveA_output_index = 21

local curveB_grad_index = 23
local curveB_output_index = 25

local curveP1_grad_index = 27
local curveP1_output_index = 29

local curveP2_grad_index = 31
local curveP2_output_index = 33


local scale_grad_index = 48
local tra_grad_index = 52
local crop_output_index = 53

local conv_touch_index = 3
local conv_crop_index = 37



   local input_img = nn.Identity()()
   -- input_scale = nn.Constant(0.5)
  local localization_network = nn.Sequential()
  print('Using pretrained ResNet-101 as the loc-net')
  local VGG16_loc = torch.load('./pretrained_ResNet/model_best.t7') -- torch.load('../ICCV-netD-2-finetuneWithCrop/checkpoints/model_best.t7')
  VGG16_loc:get(1).gradInput = torch.Tensor()
  VGG16_loc:remove(11)
  VGG16_loc:remove(10)
  VGG16_loc:remove(9) 
  VGG16_loc:evaluate()
  local conv_fea0 = VGG16_loc(input_img) -- output is 1 x 2048 x 7 x 7 

  require 'InstanceNormalization'
  local convLayer = nn.SpatialConvolution(2048,5,1,1);
  convLayer.weight:normal(0.0, 0.02)
  convLayer:noBias()
  local instanceNorm = nn.InstanceNormalization(5)
  local conv = instanceNorm(convLayer(conv_fea0))
  local crop_window0 = nn.Narrow(2,1,4)(conv)
  local probMap0 =     nn.Narrow(2,5,1)(conv)
  local crop_window = nn.Reshape(4,49,true)(crop_window0)
  local probMap =     nn.Reshape(1,49,true)(probMap0)
  require 'TopK'
  local topK_layer = nn.TopK(3,3,true,true)
  local o = topK_layer({crop_window, probMap})
  local parallel = nn.ParallelTable()
  local path1 = nn.Identity()
  local path2 = nn.Sequential()
  path2:add(nn.Reshape(3,true))
  path2:add(nn.SoftMax())
  path2:add(nn.Replicate(4,2))
  parallel:add(path1):add(path2)
  local myPair = parallel(o)
  local Weighted_Crop0 = nn.CMulTable()(myPair)
  local Weighted_Crop = nn.Sum(3,1,false)(Weighted_Crop0)


  local convLayer2 = nn.SpatialConvolution(2048,7,1,1);
  convLayer2.weight:normal(0, 0.02)
  convLayer2:noBias()
  local instanceNorm2 = nn.InstanceNormalization(7)
  local conv2 = instanceNorm2(convLayer2(conv_fea0))
  local crop_window1 = nn.Narrow(2,1,6)(conv2)
  local probMap1 =     nn.Narrow(2,7,1)(conv2)
  local crop_window2 = nn.Reshape(6,49,true)(crop_window1)
  local probMap2 =     nn.Reshape(1,49,true)(probMap1)
  require 'TopK'
  local topK_layer2 = nn.TopK(3,3,true,true)
  local o2 = topK_layer2({crop_window2, probMap2})
  local parallel2 = nn.ParallelTable()
  local path21 = nn.Identity()
  local path22 = nn.Sequential()
  path22:add(nn.Reshape(3,true))
  path22:add(nn.SoftMax())
  path22:add(nn.Replicate(6,2))
  parallel2:add(path21):add(path22)
  local myPair2 = parallel2(o2)
  local Weighted_Touch0 = nn.CMulTable()(myPair2)
  local Weighted_Touch = nn.Sum(3,1,false)(Weighted_Touch0)

   local m_sca = nn.HardTanh()(nn.Narrow(2,1,2)(Weighted_Crop))
   local const_k = (0.999 - 0.3) / 2; -- 0.3495
   local const_b = 0.999 - const_k;   -- 0.6495
   m_sca = nn.AddConstant(  const_b  )(nn.MulConstant( const_k )(m_sca)) -- This is [0.3, 0.95]

   local m_tra = nn.Narrow(2,3,2)(Weighted_Crop)
   require 'dsBound'
   dsBound = nn.dsBound()
   local m_tra_bounded = dsBound({m_tra, m_sca})

   require 'labFilter'
   alpha = nn.HardTanh()(nn.Narrow(2,1,2)(Weighted_Touch))
   alpha_const_k = (0.49 - 0.05) / 2; -- 0.22
   alpha_const_b = 0.49 - alpha_const_k; -- 0.27
   alpha = nn.AddConstant(  alpha_const_b  )(nn.MulConstant( alpha_const_k )(alpha)) -- This is [0.3, 0.95]
   lab_moved = nn.labFilter()({input_img, alpha})   

   require 'contrastFilterCurve'
   curveA = nn.HardTanh()(nn.Narrow(2,3,1)(Weighted_Touch))
   curveA_const_k = (0.4 - 0.1) / 2; -- 0.15
   curveA_const_b =  0.4 - curveA_const_k -- 0.25
   curveA = nn.AddConstant( curveA_const_b  )(nn.MulConstant(curveA_const_k)(curveA))

   curveB = nn.HardTanh()(nn.Narrow(2,4,1)(Weighted_Touch))
   curveB_const_k = (0.1 - 0.01) / 2; -- 0.045
   curveB_const_b = 0.1 - curveB_const_k -- 0.055
   curveB = nn.AddConstant( curveB_const_b  )(nn.MulConstant(curveB_const_k)(curveB))

   curveP1 = nn.HardTanh()(nn.Narrow(2,5,1)(Weighted_Touch))
   curveP1_const_k = (3 - 1) / 2; -- 1
   curveP1_const_b = 3 - curveP1_const_k -- 2
   curveP1 = nn.AddConstant( curveP1_const_b  )(nn.MulConstant(curveP1_const_k)(curveP1))

   curveP2 = nn.HardTanh()(nn.Narrow(2,6,1)(Weighted_Touch))
   curveP2_const_k = (0.99 - 0.5) / 2; -- 0.245
   curveP2_const_b = 0.99 - curveP2_const_k -- 0.745
   curveP2 = nn.AddConstant( curveP2_const_b  )(nn.MulConstant(curveP2_const_k)(curveP2))

   contrast_adj = nn.contrastFilterCurve()({lab_moved, nn.JoinTable(2)({curveA, curveB, curveP1, curveP2})})

   m_fc7_1 = nn.JoinTable(1,1){m_sca, m_tra_bounded}

   m_transp1 = nn.Transpose({2,3},{3,4})(contrast_adj) -- rot, sca or tra
   m_affineT = nn.AffineTransformMatrixGenerator(false, true, true)(m_fc7_1)
   m_affineG = nn.AffineGridGeneratorBHWD(224,224)(m_affineT)
   m_bilinear = nn.BilinearSamplerBHWD(){m_transp1, m_affineG}
   output_img = nn.Transpose({3,4},{2,3})(m_bilinear)
   STN = nn.gModule({input_img}, {output_img})

netG = nn.Sequential()
netG:add(STN)


netG:get(1):get(conv_touch_index).weight:fill(0)

netG:get(1):get(alphaA_output_index-1).constant_scalar = 0
netG:get(1):get(alphaA_output_index).constant_scalar = 0

netG:get(1):get(curveA_output_index-1).constant_scalar = 0
netG:get(1):get(curveA_output_index).constant_scalar = 0

netG:get(1):get(curveB_output_index-1).constant_scalar = 0
netG:get(1):get(curveB_output_index).constant_scalar = 0

netG:get(1):get(curveP1_output_index-1).constant_scalar = 1
netG:get(1):get(curveP1_output_index).constant_scalar = 1
netG:get(1):get(curveP2_output_index-1).constant_scalar = 1
netG:get(1):get(curveP2_output_index).constant_scalar = 1








 dummy_input_img =nn.Identity()()
 dummy_m_ds_sca = nn.Identity()()
 dummy_m_ds_tra = nn.Identity()()
 dummy_dsBound = nn.dsBound()
 dummy_m_ds_tra_bounded = dsBound({dummy_m_ds_tra, dummy_m_ds_sca})
 dummy_m_fc7_1 = nn.JoinTable(2)({dummy_m_ds_sca, dummy_m_ds_tra_bounded})
 dummy_m_transp1 = nn.Transpose({2,3},{3,4})(dummy_input_img) -- rot, sca or tra
 dummy_m_affineT = nn.AffineTransformMatrixGenerator(false, true, true)(dummy_m_fc7_1)
 dummy_m_affineG = nn.AffineGridGeneratorBHWD(224,224)(dummy_m_affineT)
 dummy_m_bilinear = nn.BilinearSamplerBHWD(){dummy_m_transp1, dummy_m_affineG}
 dummy_output_img = nn.Transpose({3,4},{2,3})(dummy_m_bilinear)
 dummy_stn  = nn.gModule({dummy_input_img, dummy_m_ds_sca, dummy_m_ds_tra}, {dummy_output_img})
VGG16_loc = nil
collectgarbage()


local ds_timer = torch.Timer()
require 'cunn'
parametersG, gradParametersG = netG:getParameters()
netG:cuda();input = torch.CudaTensor(8,3,224,224):uniform(); o1 = netG:forward(input);b1=netG:backward(input, o1:fill(1))
input=nil
local model_copy = netG:float():clone()
print('clone copy created')
local lrs1 = model_copy:getParameters()
lrs1:fill(1);
local VGG_loc_module_conv = model_copy:get(1):get(2)
assert(VGG_loc_module_conv ~= nil)
VGG_loc_module_conv:apply(function(m) 
  if m.weight then
     m.weight:fill(0)
  end
  if m.bias then
     m.bias:fill(0)
  end
end
)
model_copy:get(1):get(conv_touch_index):apply(function(m) 
  if m.weight then
     m.weight:fill(0)
  end
  if m.bias then
     m.bias:fill(0)
  end
end
)
lr_multiplierG = lrs1:clone():cuda()
print('size of lr_multiplierG is ')
print(lr_multiplierG:size())
lrs1 = nil
model_copy = nil
collectgarbage()
print('NetG created successfully')

local netD = nn.Sequential()
local VGG16 = torch.load('./pretrained_ResNet/model_best.t7') -- torch.load('../ICCV-netD-2-finetuneWithCrop/checkpoints/model_best.t7') --
VGG16:get(1).gradInput = torch.Tensor()
VGG16:evaluate()
local new_layer = VGG16:get(11):clone()
VGG16:remove(11)
netD:add(VGG16)
netD:add(new_layer)
netD:add(nn.PReLU())
netD:add(nn.Linear(2,1))
netD:add(nn.Mean(1))

local model_copy = netD:clearState():float():clone()
local lrs2 = model_copy:getParameters()
lrs2:fill(1);
local VGG_loc_module_conv = model_copy:get(1)
assert(VGG_loc_module_conv ~= nil)
VGG_loc_module_conv:apply(function(m) 
  if m.weight then
     m.weight:fill(0.01)
  end
  if m.bias then
     m.bias:fill(0.01)
  end
end
)
lr_multiplierD = lrs2:clone():cuda()
lrs2 = nil
model_copy = nil
VGG16 = nil
collectgarbage()
print('netD created successfully')



local Lab2RGB_module = torch.load('./pretrained_ResNet/Lab2RGB_50_net_G.t7') --'./Lab2RGB_50_net_G/'
cudnn.convert(Lab2RGB_module, nn)
Lab2RGB_module:evaluate()
local input_img = nn.Identity()()
local input255 = nn.MulConstant(255)(input_img)
local R = nn.Narrow(2,1,1)(input255)
local r_con_1 = nn.Constant(-103.939, 3)(R)
r_con_1 = nn.Replicate(224,3)(r_con_1)
r_con_1 = nn.Replicate(224,3)(r_con_1)
local R_new = nn.CAddTable()({R, r_con_1})
local G = nn.Narrow(2,2,1)(input255)
local g_con_1 = nn.Constant(-116.779, 3)(G)
g_con_1 = nn.Replicate(224,3)(g_con_1)
g_con_1 = nn.Replicate(224,3)(g_con_1)
local G_new = nn.CAddTable()({G, g_con_1})
local B = nn.Narrow(2,3,1)(input255)
b_con_1 = nn.Constant(-123.68, 3)(B)
b_con_1 = nn.Replicate(224,3)(b_con_1)
b_con_1 = nn.Replicate(224,3)(b_con_1)
local B_new = nn.CAddTable()({B, b_con_1})
local normalized_rgb = nn.JoinTable(2)({R_new, G_new, B_new})
local normalization_module = nn.gModule({input_img}, {normalized_rgb})
local Lab2RGB_FULL = nn.Sequential()
Lab2RGB_FULL:add(Lab2RGB_module):add(normalization_module)
Lab2RGB_FULL:cuda()
local perceptual_loss_net = torch.load('./pretrained_ResNet/vgg16.t7')
perceptual_loss_net:evaluate()
perceptual_loss_net:remove(40)
perceptual_loss_net:remove(39)

local DS_FULL_Perceptual_NETWORK = nn.Sequential()
DS_FULL_Perceptual_NETWORK:add(Lab2RGB_FULL):add(perceptual_loss_net)
DS_FULL_Perceptual_NETWORK:evaluate()
DS_FULL_Perceptual_NETWORK:cuda()

local percep_crit = nn.MSECriterion()
local criterion = nn.BCECriterion()
require 'MSECriterionDS'
local criterionG = nn.MSECriterionDS()
---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local input2 = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local input_per1 = torch.Tensor(opt.batchSize, 4096)
local input_per2 = torch.Tensor(opt.batchSize, 4096)
local output_good = torch.Tensor(opt.batchSize, 1)
local output_bad = torch.Tensor(opt.batchSize, 1)

local dummy_zero_translation = torch.Tensor(opt.batchSize, 2):fill(0):cuda()
local dummy_identity_scale = torch.Tensor(opt.batchSize, 2):fill(1):cuda()

-- GAN loss
local df_dg_gan = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
-- regularization
local df_dg_reg = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
-- perceptual loss
local df_dg_per = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)


local noise = torch.Tensor(opt.batchSize, nz, 1, 1)
local tempSource = torch.Tensor(opt.batchSize, 6, opt.fineSize, opt.fineSize)
local fakeSource = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local errD
local errG = torch.Tensor(1)
local errD_real = torch.Tensor(1)
local errD_fake = torch.Tensor(1)
local grad_of_ones = torch.Tensor(1):fill(1)
local grad_of_mones = torch.Tensor(1):fill(-1)
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------
if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   input = input:cuda();  noise = noise:cuda();  fakeSource = fakeSource:cuda(); 
   input2 = input2:cuda()
   input_per1 = input_per1:cuda()
   input_per2 = input_per2:cuda()
   output_good = output_good:cuda()
   output_bad = output_bad:cuda()
   df_dg_gan = df_dg_gan:cuda()
   df_dg_reg = df_dg_reg:cuda()
   df_dg_per = df_dg_per:cuda()
   errD_real = errD_real:cuda()
   errD_fake = errD_fake:cuda()
   errG = errG:cuda()
   if pcall(require, 'cudnn') then
      require 'cudnn'
      cudnn.benchmark = true
   end
   netD:cuda();
   netG:cuda();
   criterion:cuda()
   criterionG:cuda()
   percep_crit:cuda()
end

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

if opt.display then disp = require 'display' end

one_time_counter = true
-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
   gradParametersD:zero()
  
   errD_real:zero()
   errD_fake:zero()
   
       -- train with real
       data_tm:reset(); data_tm:resume() -- timer
       tempSource:copy(data:getBatch2())

       local real = tempSource:narrow(2,4,3)--data:getBatch()
       data_tm:stop()
       -- (1) Make data go through bilinear identity sampling 
       input:copy(dummy_stn:forward({real:float(), dummy_identity_scale, dummy_zero_translation}))
       -- (2) Getting Loss of REAL
       -- Masking color channel
       input:copy(maskColor(input))

       local output = netD:forward(input)
       errD_real:add(output)

       -- (3) Getting gradient of REAL -- 1's
       netD:backward(input, grad_of_ones)
       if ds_verbose then
         before_zeroing = gradParametersD:norm()
         io.write(string.format('gradient-updateD-REAL: (%f) %f\n', before_zeroing, gradParametersD:norm()))
       end
       gradParametersD:cmul(lr_multiplierD)
       -- train with fake
       
       local myfakeSource = tempSource:narrow(2,1,3)
       if one_time_counter then
          noise_vis = myfakeSource:clone()
          one_time_counter = false
       end
       data_tm:stop()
       fakeSource:copy(myfakeSource)
       local fake = netG:forward(fakeSource)
       input:copy(fake)
       input:copy(maskColor(input))
       local output = netD:forward(input)
       errD_fake:add(output)
       netD:backward(input, grad_of_mones)
       if ds_verbose then
         before_zeroing = gradParametersD:norm()
         io.write(string.format('gradient-updateD-FAKE: (%f) %f\n', before_zeroing, gradParametersD:norm()))
       end   
       gradParametersD:cmul(lr_multiplierD:cuda())

   errD = errD_real - errD_fake
   return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
   gradParametersG:zero()

   errG:zero()
   local myNorm_G 
   for effectiveBatchSize = 1, 64 / opt.batchSize do
       tempSource:copy(data:getBatch2())
       local myfakeSource = tempSource:narrow(2,1,3)
       fakeSource:copy(myfakeSource)
       input2:copy(fakeSource)
       input2:copy(dummy_stn:forward({myfakeSource:float(),dummy_identity_scale, dummy_zero_translation}))
       -- (3) Perceptual Loss
        local fc7_2 = DS_FULL_Perceptual_NETWORK:forward(input2)    -- original bad image
        input_per2:copy(fc7_2)
       input2:copy(maskColor(input2))
       target = netD:forward(input2)
       output_bad:copy(netD:get(4).output) -- Regularization, this should be larger or positive
       local fake = netG:forward(fakeSource)
       input:copy(fake)
        -- (3) Perceptual Loss
        local fc7_1 = DS_FULL_Perceptual_NETWORK:forward(input)    -- cropped image
        input_per1:copy(fc7_1)
       input:copy(maskColor(input))
       output = netD:forward(input)         
       output_good:copy(netD:get(4).output) -- Regularization, this should be smaller or negative
       errG:add(output)
       -- (1) Gan Loss
       local df_dg = netD:updateGradInput(input, grad_of_ones)
       df_dg_gan:copy(df_dg)

       df_dg_gan:copy(maskColor(df_dg_gan))

       local df_dg_gan_norm = df_dg_gan:norm()
       if   (df_dg_gan_norm >= 1.0) then
             df_dg_gan:mul(1.0/df_dg_gan_norm)
       end      
        -- (2) Regularization Loss
       local dummy = criterionG:forward(output_good, output_bad)
       local df_dD = criterionG:backward(output_good, output_bad)
       for d = 4,1,-1 do
          if d > 1 then 
            df_dD = netD:get(d):updateGradInput(netD:get(d-1).output, df_dD)
          else
            df_dD = netD:get(1):updateGradInput(input, df_dD)
          end
       end
       df_dg_reg:copy(df_dD)
       df_dg_reg:copy(maskColor(df_dg_reg))
       local df_dg_reg_norm = df_dg_reg:norm()
       if   (df_dg_reg_norm >= 1) then
             df_dg_reg:mul(1.0/df_dg_reg_norm)
       end  

      -- (3) Perceptual Loss
      ----------------------------------------pred,            y
      local percep_loss = percep_crit:forward(input_per1, input_per2)
      local grad_out_percep = percep_crit:backward(input_per1, input_per2)
      grad_out_percep = DS_FULL_Perceptual_NETWORK:updateGradInput(input, grad_out_percep)
      df_dg_per:copy(grad_out_percep)
      df_dg_per:copy(maskColor(df_dg_per))
       local df_dg_per_norm = df_dg_per:norm()
       if   (df_dg_per_norm >= 1.0) then
             df_dg_per:mul(1.0/df_dg_per_norm)
       end  
       io.write(string.format('Gradient contribution G1-gan: %f, G2-reg: %f, G3-per: %f\n', df_dg_gan:norm(), df_dg_reg:norm(), df_dg_per:norm()))
       myNorm_G = (df_dg_gan+df_dg_reg+df_dg_per):norm()
       netG:backward(fakeSource, df_dg_gan+df_dg_reg+df_dg_per)
    end

   gradParametersG:cmul(lr_multiplierG)

   local str =     string.format('Gradient to attack STN fc params, [scale_x: %f, scale_y: %f, x: %f, y: %f] - total gradient from D at G: %f - ', 
    netG:get(1):get(scale_grad_index).gradInput[1][1],  netG:get(1):get(scale_grad_index).gradInput[1][2],
    netG:get(1):get(tra_grad_index).gradInput[1][1][1], netG:get(1):get(tra_grad_index).gradInput[1][1][2],
    myNorm_G);
    print(str)

   str = string.format('Gradient to attack STN fc params, [alphaA--: %f, betaB-----: %f] -- \n',  
    netG:get(1):get(alphaA_grad_index).gradInput[1][1],  netG:get(1):get(alphaA_grad_index).gradInput[1][2])
   io.write(str)

   str = string.format('Gradient to attack STN fc params, [a: %f, b: %f, p: %f, q: %f] -- ',  
    netG:get(1):get(curveA_grad_index).gradInput[1][1],  netG:get(1):get(curveB_grad_index).gradInput[1][1],
    netG:get(1):get(curveP1_grad_index).gradInput[1][1], netG:get(1):get(curveP2_grad_index).gradInput[1][1])
   io.write(str)
   
   local myNorm = gradParametersG:norm()
   io.write(string.format('Total GradientUpdate norm = %f -- not clip at 100\n',myNorm))
   --local absERROR = torch.abs(errD_real[1] - errG[1])
   if myNorm > 100 then
      gradParametersG:mul( 100 / myNorm)
   end

   errG:div(64 / opt.batchSize)
   return errG-errD_real, gradParametersG
end

-- train
generation_iter = 0
for epoch = 1, opt.niter do
    netG:training()
    netD:training()
   epoch_tm:reset()
   local counter = 0
   i = 0
   while i <= math.min(data:size(), opt.ntrain) do
      tm:reset()
      -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      j = 0
      Diters = 0
      if generation_iter == 0 then
          Diters = 50
      else
          Diters = 1
      end
      
      local absERROR = torch.abs(errD_real[1] - errG[1])
      if absERROR < 1 or generation_iter == 0 or errD_real[1] > errG[1] then  
        while j < Diters and i <= math.min(data:size(), opt.ntrain) do
          optim.rmsprop(fDx, parametersD, optimStateD)       
          parametersD:clamp(-2,2)
          j = j + 1
          i = i + opt.batchSize
          counter = counter + 1
        end        
      end
      Giters = 1
      k = 0
      while k < Giters and i <= math.min(data:size(), opt.ntrain) do
        -- (2) Update G network: maximize log(D(G(z)))
        generation_iter = generation_iter + 1
        optim.rmsprop(fGx, parametersG, optimStateG)
        i = i + opt.batchSize
        k = k + 1
      end
      -- display
      counter = counter + 1
      if generation_iter % 1 == 0 and opt.display then

           tempSource:copy(data:getBatch2())
           local myfakeSource = tempSource:narrow(2,1,3)    
           

          noise_vis:copy(myfakeSource)
          local fake = netG:forward(noise_vis:cuda())
          local real = tempSource:narrow(2,4,3) --data:getBatch()
          noise_vis1 = getRGBback(noise_vis)
          fake1 = getRGBback(fake)       
          real1 = getRGBback(real)          
          savePath1 = string.format('./generated_imgs/Iter_%d_input.jpg',generation_iter)
          savePath2 = string.format('./generated_imgs/Iter_%d_enhanced.jpg',generation_iter)
          if generation_iter % 10 == 0 then
            disp.image(noise_vis1, {normalize=false,win=opt.display_id,    title=string.format('low-quality: %s',opt.name), saveThisOne=false, saveName=savePath1})
            disp.image(fake1,      {normalize=false,win=opt.display_id * 3,title=string.format('Enhanced:%s',opt.name),    saveThisOne=false, saveName=savePath2})
            disp.image(real1,      {normalize=false,win=opt.display_id * 9,title=string.format('High-quality:%s',opt.name)})
          else
            disp.image(noise_vis1, {normalize=false,win=opt.display_id,    title=string.format('low-quality:%s',opt.name), saveThisOne=false, saveName=savePath1})
            disp.image(fake1,      {normalize=false,win=opt.display_id * 3,title=string.format('Enhanced:%s',opt.name),    saveThisOne=false, saveName=savePath2})
            disp.image(real1,      {normalize=false,win=opt.display_id * 9,title=string.format('High-quality:%s',opt.name)})
          end
      end

        -- logging
        ds_n = ((i-1) / opt.batchSize)
        ds_trainSize = math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize)


        for batchIdx = 1, opt.batchSize do
                    io.write(string.format('[outputs]: alpha = %.3f beta = %.3f | a = %.3f b = %.3f p = %.3f q = %.3f\n[outputs]: s1 = %.3f, s2 = %.3f, t1 = %.3f, t2 = %.3f\n', 
                    netG:get(1):get(alphaA_output_index).output[batchIdx][1], netG:get(1):get(alphaA_output_index).output[batchIdx][2],
                    netG:get(1):get(curveA_output_index).output[batchIdx][1], netG:get(1):get(curveB_output_index).output[batchIdx][1],
                    netG:get(1):get(curveP1_output_index).output[batchIdx][1], netG:get(1):get(curveP2_output_index).output[batchIdx][1],
                             netG:get(1):get(crop_output_index).output[batchIdx][1],
                             netG:get(1):get(crop_output_index).output[batchIdx][2],
                             netG:get(1):get(crop_output_index).output[batchIdx][3],
                             netG:get(1):get(crop_output_index).output[batchIdx][4]))
        end

        -- print(string.format('LR_Mul_D = %f', lr_multiplierD:sum()))
        local fc_index = 2
        print(string.format('netD-FC: %f (%f), %f; gradInput: %f\n', 
        netD:get(fc_index).weight:norm(),netD:get(fc_index).weight[1][1], netD:get(fc_index).bias[1], netD:get(fc_index).gradInput:norm()))
        local fc_index = 4
        print(string.format('netD-FC: %f (%f), %f; gradInput: %f\n', 
        netD:get(fc_index).weight:norm(),netD:get(fc_index).weight[1][1], netD:get(fc_index).bias[1], netD:get(fc_index).gradInput:norm()))
        print(('Epoch: [%d][%8d %8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                 .. ' Loss_D %.4f, Loss_G %.4f, Loss_D_real %.4f, Loss_D_fake %.4f  Time %.3f  ETA: %7.3f'):format(
               epoch, generation_iter, ((i-1) / opt.batchSize),
               math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
               tm:time().real, data_tm:time().real,
               errD and errD[1] or -1, errG and errG[1] or -1, errD_real and errD_real[1] or -1, errD_fake and errD_fake[1] or -1,
               epoch_tm:time().real, epoch_tm:time().real/(ds_n/ds_trainSize)))
       if generation_iter % 200 == 0 then -- in generation 
          -- Checkpointing
          paths.mkdir('checkpoints')
          print(string.format('Size of parameters before saving'))
          print(parametersG:size())
          print(gradParametersG:size())
          parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
          parametersG, gradParametersG = nil, nil
          torch.save('checkpoints/' .. opt.name .. '_' .. generation_iter .. '_net_G.t7', netG:clearState())
          torch.save('checkpoints/' .. opt.name .. '_' .. generation_iter .. '_net_D.t7', netD:clearState())

          parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
          parametersG, gradParametersG = netG:getParameters()
          print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
                  epoch, opt.niter, epoch_tm:time().real))         
          netG:training()
          netD:training()
          local o1 = netG:forward(torch.CudaTensor(8,3,224,224):uniform());
          local b1 = netG:backward(torch.CudaTensor(8,3,224,224):uniform(), o1:fill(1))
          print(string.format('Size of parameters after saving'))
          print(parametersG:size())
          print(gradParametersG:size())
      end         
      if generation_iter >= 5000 then 
          break
      end      
   end -- while i <= math.min(data:size(), opt.ntrain) do
   if epoch % 4 == 0 then
      optimStateD.learningRate = optimStateD.learningRate / 10.0
      print(string.format('Learning Rate is Changed to %f at the end of epoch %d', optimStateD.learningRate, epoch))
   end -- for epoch = 1, opt.niter do

   if generation_iter >= 5000 then 
      parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
      parametersG, gradParametersG = nil, nil          
      netG = nil
      collectgarbage()
      print('Stage 1 finished')
      return 
   end
end
