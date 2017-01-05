require 'nn'
local nninit = require 'nninit'
require 'optim'
require 'pretty-nn'
require 'csvigo'

-- parameters
torch.manualSeed(1234)
local epochs = 10000
local n_input = 9  -- number of input features
local n_hidden = 9  -- number of hidden 
-- data prepare
-- ANN: 9-5-1, 1~221 train, 222~288 test, total 289
local m_train = 221
local m_test = 67

loaded = csvigo.load('sunspot_data.csv')
src_data = torch.Tensor(loaded.x)
-- nomalize data
data_range = src_data:max() - src_data:min()
src_data_normalized = (src_data - src_data:min()) / data_range

function make_inputs (n_rows, n_tr, n_te, n_input)
	local tr = torch.Tensor(n_tr - n_input, n_input+1)
	local te = torch.Tensor(n_te, n_input+1)

	for i = 1, n_tr - n_input do
		tr[{ i, {} }] = src_data_normalized[{ {i,i+n_input} }]
	end

	local e = n_rows-n_input  -- 289-9 = 280
	local s = e - n_te + 1    -- 280-67+1
	for i = s, e do
		te[{ i, {} }] = src_data_normalized[{ {i,i+n_input} }]
	end
	return tr, te
end

train_data = torch.Tensor(212, 10)
test_data = torch.Tensor(67, 10)
      
for i = 1, 212 do
   train_data[{ i, {} }] = src_data_normalized[{ {i,i+9} }]
end

for i = 214, 280 do
   test_data[{ i-213, {} }] = src_data_normalized[{ {i,i+9} }]
end

-- model
model = nn.Sequential()
model:add(nn.Linear(n_input, n_hidden):init('weight', nninit.xavier, {dist = 'normal'}))
model:add(nn.Tanh())
model:add(nn.Linear(n_hidden, 1):init('weight', nninit.xavier, {dist = 'normal'}))

-- loss function
criterion = nn.MSECriterion()

-- test
function test(test_data, model, criterion)
	local diff = 0
	local loss = 0

	for i = 1, (#test_data)[1] do
		local pred = model:forward(test_data[i][{ {1,n_input} }])
		local target = test_data[i][{ {n_input+1} }]
		loss = loss + criterion:forward(pred, target)
		diff = diff + torch.abs(pred - target)
	end

	local mae = (diff / (#test_data)[1]) * data_range
	print('loss = ' .. loss .. ' mae = ' .. mae[1])
end

-- 4. Train the model
function train(tr_data, te_data, model, criterion)
	x, dl_dx = model:getParameters()

	feval = function(x_new)
		if x ~= x_new then
			x:copy(x_new)
		end

		_nidx_ = (_nidx_ or 0) + 1
		if _nidx_ > (#train_data)[1] then _nidx_ = 1 end

		local sample = train_data[_nidx_]
		local inputs = sample[{ {1, n_input} }]
		local target = sample[{ {n_input+1} }]

		-- reset gradients
		-- dl_dx:zero()
		model:zeroGradParameters()
		local pred = model:forward(inputs)
		local loss = criterion:forward(pred, target)
		local gradOut = criterion:backward(pred, target)
		model:backward(inputs, gradOut)

		return loss, dl_dx
	end

	adam_params = {
		learningRate = 1e-3,
		learningRateDecay = 1e-4,
		--   weightDecay = 0, -- L2 regularization
		--   momentum = 0
	}

	best_mae = 100
	maxWait = 200
	maxTries = 200
	try = 1

	for i = 1, epochs do
		-- train
		current_loss = 0

		for i = 1, (#train_data)[1] do
			_, fs = optim.adam(feval, x, adam_params)
			current_loss = current_loss + fs[1]
		end

		current_loss = current_loss / (#train_data)[1]

		-- test
		diff = 0
		for i = 1, (#test_data)[1] do
			local pred = model:forward(test_data[i][{ {1,n_input} }])
			local target = test_data[i][{ {n_input+1} }]
			diff = diff + torch.abs(pred - target)
		end

		mae = (diff / (#test_data)[1]) * (src_data:max() - src_data:min())
		print(i .. ' current loss = ' .. current_loss .. ' mae = ' .. mae[1])

		-- early stopping
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

	print('--- Finished ----------')
	print('Best MAE: ' .. best_mae)
	torch.save('sunspot_fcff.t7', best_model)
end

train(train_data, test_data, model, criterion)
-- model = torch.load('sunspot_fcff.t7')
-- test(test_data, model, criterion)
