require 'nn'
require 'csvigo'
require 'rnn'
require 'optim'
local nninit = require 'nninit'

-- results ----------------------------------------------
-- t = 8, LSTM_size = 200, 12.27
-- t = 9, LSTM_size = 200, 11.36
-- t = 9, GRU_size = 200, lr 0.001 , lrdecay 1e-4, weightDecay = 0.001, 11.16

-- parameters
torch.manualSeed(1234)
local epochs = 5000
local t = 9  -- time-steps
local n = 1  -- number of input feature 
batchSize = 20
maxWait = 200
maxTries = 200

-- data preparation -------------------------------------------
-- ANN: 9-5-1, 1~221 train, 222~288 test, total 289
loaded = csvigo.load('sunspot_data.csv')
src_data = torch.Tensor(loaded.x)
-- nomalize data
data_range = src_data:max() - src_data:min()
src_normalized = src_data / data_range

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

-- Model -------------------------------------------------
model = nn.Sequential()
model:add(nn.Sequencer(nn.GRU(1,200, 0.5)))
model:add(nn.SelectTable(-1))
model:add(nn.Linear(200, 1):init('weight', nninit.xavier))

-- Criterion ---------------------------------------------
criterion = nn.MSECriterion()

function test(testX, testY, model, criterion)
	local diff = 0
	local test_cnt = 0
	local loss = 0

	model:evaluate()
	for i = 1, #testX do
		local pred = model:forward(testX[i])
		local target = testY[i]
		loss = criterion:forward(pred, target)
		diff = diff + torch.abs(pred - target):sum()
		test_cnt = test_cnt + (#testX[1][1])[1]
	end

	local mae = (diff / test_cnt) * data_range
	print('loss = ' .. loss .. ' mae = ' .. mae)
end

-- Train -------------------------------------------------
function train(trainX, trainY, testX, testY, model, criterion)
   local x, dl_dx = model:getParameters()

   feval = function(x_new)
      if x ~= x_new then
			x:copy(x_new)
      end

      _nidx_ = (_nidx_ or 0) + 1
      if _nidx_ > #trainX then _nidx_ = 1 end

      local input = trainX[_nidx_]
      local target = trainY[_nidx_]

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
      weightDecay = 0.001 -- L2 regularization
		-- momentum = 0.9
   }

   best_mae = 100

   for i = 1, epochs do
		-- train
      model:training()
      
		for i = 1, #trainX do
			_, fs = optim.adam(feval, x, optim_params)
			current_loss = fs[1]
      end
   
      -- test
      diff = 0
      test_cnt = 0

      model:evaluate()
      for i = 1, #testX do
			local pred = model:forward(testX[i])
			local target = testY[i]
			diff = diff + torch.abs(pred - target):sum()
			test_cnt = test_cnt + (#testX[1][1])[1]
      end

      local mae = (diff / test_cnt) * data_range
      print(i .. ' current loss = ' .. current_loss .. ' mae = ' .. mae)

      --  early stopping
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

   print('---- Finished ---')
   print('best MAE: ' .. best_mae)
   torch.save('sunspot_gru.t7', best_model)
end


-- train(train_inputs, train_targets, test_inputs, test_targets, model, criterion)
-- model = torch.load('sunspot_lstm_10.55.t7')
-- test(test_inputs, test_targets, model, criterion)
