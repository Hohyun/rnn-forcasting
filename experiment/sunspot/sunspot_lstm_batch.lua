require 'nn'
require 'csvigo'
require 'rnn'
require 'optim'
local nninit = require 'nninit'

-- parameters
torch.manualSeed(1234)
local epochs = 5000
local t = 4  -- time-steps
local n = 1  -- number of input feature 
batchSize = 20

-- data prepare
-- ANN: 9-5-1, 1~221 train, 222~288 test, total 289
loaded = csvigo.load('sunspot_data.csv')
src_data = torch.Tensor(loaded.x)
-- nomalize data
src_normalized = (src_data / (src_data:max() - src_data:min()))

--train_inputs = torch.Tensor(212, t, 1)
--train_targets = torch.Tensor(212, 1)
--test_inputs = torch.Tensor(67, t, 1)
--test_targets = torch.Tensor(67, 1)

function make_inputs (s_row, e_row)
   local data = {}
   for i = s_row, e_row-t, batchSize do
      local batch = {}
      local remain_rows = math.min(batchSize, e_row-t-i+1)
      for j = 1, t do
	 local sub_batch = torch.Tensor(remain_rows, 1)
	 for k = 1, remain_rows do
	    sub_batch[{ k, 1 }] = src_normalized[i+j+k-2]
	 end
	 table.insert(batch, sub_batch)
      end
      table.insert(data, batch)
   end
   return data
end

function make_targets (s_row, e_row)
   local data = {}
   for i = s_row+t, e_row, batchSize do
      local remain_rows = math.min(batchSize, e_row-i+1)
      local batch = torch.Tensor(remain_rows, 1)
      for j = 1, remain_rows do
	 batch[{ j,1 }] = src_normalized[i+j-1]
      end
      table.insert(data, batch)
   end
   return data
end

train_inputs = make_inputs(1, 221)
train_targets = make_targets(1, 221)
test_inputs = make_inputs(223-t, 289)
test_targets = make_targets(223-t, 289)

local model = nn.Sequential()
--model:add(nn.SplitTable(1))
model:add(nn.Sequencer(nn.FastLSTM(1,200)))
model:add(nn.SelectTable(-1))
model:add(nn.Linear(200, 1):init('weight', nninit.xavier))

criterion = nn.MSECriterion()

-- 4. Train the model
local x, dl_dx = model:getParameters()
--x:copy(torch.randn(x:size()))

feval = function(x_new)
   if x ~= x_new then
      x:copy(x_new)
   end

   _nidx_ = (_nidx_ or 0) + 1
   if _nidx_ > #train_inputs then _nidx_ = 1 end

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
   learningRate = 0.0001,
   learningRateDecay = 1e-5,
--   weightDecay = 0, -- L2 regularization
--   momentum = 0
}

best_mae = 100
maxWait = 100
maxTries = 100

for i = 1, epochs do

   for i = 1, #train_inputs do
      _, fs = optim.adam(feval, x, optim_params)
      current_loss = fs[1]
   end
   
   --   logger:add{['training error'] = current_loss}
   --   logger:style{['training wrror'] = '-'}

   diff = 0
   test_cnt = 0
   for i = 1, #test_inputs do
      local pred = model:forward(test_inputs[i])
      local target = test_targets[i]
      diff = diff + torch.abs(pred - target):sum()
      test_cnt = test_cnt + (#test_inputs[1][1])[1]
   end

   local mae = (diff / test_cnt) * (src_data:max() - src_data:min())
   print(i .. ' current loss = ' .. current_loss .. ' mae = ' .. mae)

   if mae < best_mae and epochs > maxWait then
      best_mae = mae
      best_model = model:clone()
      try = 1
   else
      try = try + 1
   end

   if try > maxTries and try > maxWait then
      break
   end
   
end

-- Test
print('---- Finished ---')
print('best MAE: ' .. best_mae)
torch.save('sunspot_lstm.t7', best_model)


