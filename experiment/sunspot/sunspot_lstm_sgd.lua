require 'nn'
require 'csvigo'
require 'rnn'
require 'optim'
local nninit = require 'nninit'

-- parameters
torch.manualSeed(1234)
local epochs = 1000
local t = 4  -- time-steps

-- data prepare
-- ANN: 9-5-1, 1~221 train, 222~288 test, total 289
loaded = csvigo.load('sunspot_data.csv')
src_data = torch.Tensor(loaded.x)
-- nomalize data
src_data_normalized = (src_data / (src_data:max() - src_data:min()))

train_inputs = torch.Tensor(212, t, 1)
train_targets = torch.Tensor(212, 1)
test_inputs = torch.Tensor(67, t, 1)
test_targets = torch.Tensor(67, 1)
   
for i = 1, 212 do
   for j = 1, t do
      train_inputs[{ i, {j}, 1 }] = src_data_normalized[{ i+j-1 }]
   end
   train_targets[{ {i},1 }] = src_data_normalized[{ i+t }]
end

for i = 214, 280 do
   for j = 1, t do
      test_inputs[{ i-213, {j}, 1 }] = src_data_normalized[{ i+j-1 }]
   end
   test_targets[{ {i-213},1 }] = src_data_normalized[{ i+t }]
end

local model = nn.Sequential()
model:add(nn.SplitTable(1))
model:add(nn.Sequencer(nn.FastLSTM(1,50)))
model:add(nn.SelectTable(-1))
model:add(nn.Linear(50, 1):init('weight', nninit.xavier))

criterion = nn.MSECriterion()

-- 4. Train the model
local x, dl_dx = model:getParameters()
--x:copy(torch.randn(x:size()))

feval = function(x_new)
   if x ~= x_new then
      x:copy(x_new)
   end

   _nidx_ = (_nidx_ or 0) + 1
   if _nidx_ > (#train_inputs)[1] then _nidx_ = 1 end

   input = train_inputs[_nidx_]
   target = train_targets[_nidx_]

   -- dl_dx:zero()
   model:zeroGradParameters()
   local pred = model:forward(input)
   local loss = criterion:forward(pred, target)
   local gradOut = criterion:backward(pred, target)
   model:backward(input, gradOut)

   return loss, dl_dx
end

optim_params = {
   learningRate = 0.001,
   learningRateDecay = 1e-4,
--   weightDecay = 0, -- L2 regularization
--   momentum = 0
}

best_mae = 100
maxWait = 50
maxTries = 50

for i = 1, epochs do
   current_loss = 0

   for i = 1, (#train_inputs)[1] do
      _, fs = optim.adam(feval, x, optim_params)
      current_loss = current_loss + fs[1]
   end

   current_loss = current_loss / (#train_inputs)[1]
   
   --   logger:add{['training error'] = current_loss}
   --   logger:style{['training wrror'] = '-'}

   diff = 0
   for i = 1, (#test_inputs)[1] do
      local pred = model:forward(test_inputs[i])
      local target = test_targets[i]
      diff = diff + torch.abs(pred - target)
   end

   mae = (diff / (#test_inputs)[1]) * (src_data:max() - src_data:min())
   print(i .. ' current loss = ' .. current_loss .. ' mae = ' .. mae[1])

   if mae[1] < best_mae and epochs > maxWait then
      best_mae = mae[1]
      best_model = model:clone()
      try = 1
   else
      try = try + 1
   end

   if try > maxTries then
      break
   end
   
end

-- Test
print('---- Finished ---')
print('best MAE: ' .. best_mae)
torch.save('sunspot_lstm.t7', best_model)


