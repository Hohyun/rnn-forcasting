require 'nn'
require 'csvigo'
require 'rnn'
local nninit = require 'nninit'

-- parameters
torch.manualSeed(1234)
local epochs = 1000
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

