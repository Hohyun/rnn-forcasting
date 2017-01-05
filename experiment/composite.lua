require 'nn'
require 'rnn'

model = nn.Sequential()

sub = nn.ParallelTable()
sub:add(nn.Linear(9,1))
sub:add(nn.Linear(9,1))

model:add(sub)
model:add(nn.CAddTable())
print(model)

x1 = torch.randn(9)
x2 = torch.randn(9)
input = {x1, x2}


pred_sub = sub:forward(input)
pred = model:forward(input)

for i,k in pairs(pred_sub) do print (i,k) end
print(pred)
