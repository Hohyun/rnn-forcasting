require 'nn'
require 'rnn'
require 'csvigo'
require 'optim'
local nninit = require 'nninit'

torch.manualSeed(1234)

rnn1 = nn.Sequential()
rnn1:add(nn.Sequencer(nn.GRU(1, 200, 0.5)))
rnn1:add(nn.SelectTable(-1))
rnn1:add(nn.Linear(200,1):init('weight', nninit.xavier))

rnn2 = nn.Sequential()
rnn2:add(nn.Sequencer(nn.GRU(1, 200, 0.5)))
rnn2:add(nn.SelectTable(-1))
rnn2:add(nn.Linear(200,1):init('weight', nninit.xavier))

sub = nn.ParallelTable()
sub:add(rnn1)
sub:add(rnn2)

model = nn.Sequential()
model:add(sub)
model:add(nn.CAddTable())
print(model)

function prepare_data(step)
	local x = {}
	local x1 = {}
	local x2 = {}
	
	for i = 1,step do table.insert(x1, torch.randn(1)) end
	for i = 1,step do table.insert(x2, torch.randn(1)) end
	x = {x1, x2}
	return x
end

input = prepare_data(9)

pred = model:forward(input)

print(pred)
