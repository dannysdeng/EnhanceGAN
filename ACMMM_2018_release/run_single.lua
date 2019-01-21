require 'torch'
require 'nn'
require 'optim'
require 'image';
require 'tds'
require 'nngraph'
require 'dpnn'
require 'stn'
require 'dsBound'
require 'labFilter'
require 'contrastFilterCurve'
require 'TopK'
require 'InstanceNormalization'
require 'contrastFilter'

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:option('--filename', '', 'filename of user uploaded image')
cmd:option('--gpu', 1, 'Use GPU: 1 | Use CPU: 0')

cmd:text()
local opt = cmd:parse(arg or {})


if opt.gpu == 1 then
  print('Using GPU mode. This requires CUDNN 5 and CUNN')
  require 'cunn'
  require 'cudnn'
end



function randomCrop224(input)
    local size =224
    local w, h = input:size(3), input:size(2)
    if w == size and h == size then
       return input
    end
    local x1, y1 = torch.random(0, w - size), torch.random(0, h - size)
    -- print(string.format('[x1, y1] = [%.2f, %.2f] -- ',x1,y1))
    local out = image.crop(input, x1, y1, x1 + size, y1 + size)
    assert(out:size(2) == size and out:size(3) == size, 'wrong crop size')
    return out
end


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
          matio = require 'matio'
meanstd = {
   mean = { 0,-128,-128 },
   std = { 100,255,255 },
   scale = 1,
}

vgg_meanstd = {
   mean = { 103.939, 116.779, 123.68 },
   std = { 1,1,1 },
   scale = 255,
}

function getRGBback(LAB_img_batch)
    if LAB_img_batch:nDimension() == 3 then
        LAB_img_batch = nn.Unsqueeze(1):forward(LAB_img_batch:double())
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
function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end


  I = torch.FloatTensor(1,3,224,224) --I = torch.CudaTensor(1,3,224,224)
  if opt.gpu == 1 then
    print('Loading pretrained model in GPU mode') 
    netG = torch.load('./checkpoints/model_best.t7')
  else
    print('Loading pretrained model in CPU mode')
    netG = torch.load('./checkpoints/model_best_cpu.t7')
  end
  netG:evaluate()
  netG:float()

  if opt.gpu == 1 then
    netG:cuda()
    I = I:cuda()
  end


  os.execute("mkdir demo_result")
  new_results = ("demo_result")
  -- for i = 1, total_img do
     i = 1
     filename = opt.filename
     original_img = image.load(string.format('%s',filename), 3, 'float')
     original_img_save = original_img:clone():float()

     -- original_size = torch.Tensor({original_img:size(1), original_img:size(2), original_img:size(3)})
     big_width = original_img:size(3) --original_size[1][3]
     big_height = original_img:size(2)--original_size[1][2]

     I_224 = image.scale(original_img_save, 224,224)
     I_224= image.rgb2lab(I_224)
     I:copy(I_224)

     -------------------------------------------------------------
     ----------------Getting cropping Parameters------------------
     -------------------------------------------------------------
     I_cropped = netG:forward(I)

     alpha_beta = netG:get(1):get(alphaA_output_index).output
     alpha = alpha_beta[1][1]
     beta = alpha_beta[1][2]

     a = netG:get(1):get(curveA_output_index).output[1][1]
     b = netG:get(1):get(curveB_output_index).output[1][1]
     p = netG:get(1):get(curveP1_output_index).output[1][1]
     q = netG:get(1):get(curveP2_output_index).output[1][1]

     crop_conv_output = netG:get(1):get(conv_crop_index).output
     crop_conv_output2 = netG:get(1):get(conv_crop_index+1).output
     lab_conv_output = netG:get(1):get(conv_touch_index).output
     lab_conv_output2 = netG:get(1):get(conv_touch_index+1).output
     --save_name = string.format('./%s/crop_%d.mat',new_results, i)
     --matio.save(save_name, {t1=crop_conv_output:float(), t2=crop_conv_output2:float(), t3=lab_conv_output:float(), t4=lab_conv_output2:float()})


     LAB = image.rgb2lab(original_img)
     for lab=1,3 do
      LAB[lab]:add(-meanstd.mean[lab])
      LAB[lab]:div(meanstd.std[lab])
     end     

     L = LAB:narrow(1,1,1)
     A = LAB:narrow(1,2,1)
     B = LAB:narrow(1,3,1)

      local k1 = a / ((a - b)^(1/p))
      local k2 = a / ((a - b)^(1/q))
      L:apply(function(m) 
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


      k = 1.0 / (1 - 2*alpha)
      b = -k*alpha
      A:apply(function(m) 
        if m < alpha then
            return 0
        elseif m > 1-alpha then
            return 1
        else 
            return k*m + b
        end
      end
      )
      k = 1.0 / (1 - 2*beta)
      b = -k*beta
      B:apply(function(m) 
        if m < beta then
            return 0
        elseif m > 1-beta then
            return 1
        else 
            return k*m + b
        end
      end
      )
     LAB:narrow(1,1,1):copy(L)
     LAB:narrow(1,2,1):copy(A)
     LAB:narrow(1,3,1):copy(B)
     filtered_img = getRGBback(LAB):squeeze():float()
     save_name = string.format('./%s/%s_filteredONLY_%d.jpg',new_results, filename,i)
     image.save(save_name, filtered_img)         
      
     crop_params = netG:get(1):get(crop_output_index).output
     -----------------------------------------------------------------------------------------------------
     ----------------Getting parameters of crop, apply it onto the original image------------------
     -----------------------------------------------------------------------------------------------------

     s1 = crop_params[1][1]
     s2 = crop_params[1][2]
     t1 = crop_params[1][3]
     t2 = crop_params[1][4]
     --save_name = string.format('./%s/crop_coordinates_%d.mat',new_results, i)
     --matio.save(save_name, {s1=s1, s2=s2, t1=t1, t2=t2})        
     -- row, col = I:size(3), I:size(4)
     row, col = big_height, big_width
     sub_rows = torch.round(row * s1)
     sub_cols = torch.round(col * s2)
     I_cropped_output = torch.rand(1, 3, sub_rows, sub_cols)
     croppedONLY = torch.rand(1,3,sub_rows, sub_cols)
     I_mask = original_img:clone():fill(0)
     row_start = torch.floor((row - sub_rows)/2)+1   +   torch.round(t1 * sub_rows/2);
     col_start = torch.floor((col - sub_cols)/2)+1   +   torch.round(t2 * sub_cols/2);
     if row_start <= 0 then 
        row_start = 1
     end
     if col_start <= 0 then 
        col_start = 1
     end

     if row_start + sub_rows - 1 > row then
         row_start = row - sub_rows + 1
     end

     if col_start + sub_cols - 1 > col then
        col_start = col - sub_cols + 1
     end


     -- print(filtered_img:size())
     -- print(row_start, sub_rows, col_start, sub_cols)
     croppedONLY:copy(original_img:narrow(2,row_start,sub_rows):narrow(3, col_start, sub_cols))
     save_name = string.format('./%s/%s_cropONLY_%d.jpg',new_results, filename, i)
     image.save(save_name, croppedONLY:squeeze())     

     I_cropped_output:copy(filtered_img:narrow(2,row_start,sub_rows):narrow(3, col_start, sub_cols))
     I_mask:narrow(2,row_start,sub_rows):narrow(3, col_start, sub_cols):fill(1):float()
     I_cropped_output = I_cropped_output:squeeze():float()
     save_name = string.format('./%s/%s_final_%d.jpg',new_results, filename,i)
     image.save(save_name, I_cropped_output)     
    
     






