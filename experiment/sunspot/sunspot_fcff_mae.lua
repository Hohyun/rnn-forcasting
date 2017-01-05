require 'nn'
require 'csvigo'

-- parameters
loaded = csvigo.load('sunspot_data.csv')
src_data = torch.Tensor(loaded.x)
-- nomalize data
src_data_normalized = (src_data / (src_data:max() - src_data:min())) - 0.5

train_data = torch.Tensor(212, 10)
test_data = torch.Tensor(67, 10)
      
for i = 1, 212 do
   train_data[{ i, {} }] = src_data_normalized[{ {i,i+9} }]
end

for i = 214, 280 do
   test_data[{ i-213, {} }] = src_data_normalized[{ {i,i+9} }]
end

model = torch.load('sunspot_fcff_12.84.t7')
--model = torch.load('sunspot_ann_919.t7')

mae = 0
mse = 0
mape = 0
range = src_data:max() - src_data:min()

pred_data = torch.Tensor((#test_data)[1], 3)

for i = 1, (#test_data)[1] do
   local pred = model:forward(test_data[i][{ {1,9} }])
   local target = test_data[i][{ {10} }]
	local pred_org = (pred[1] + 0.5) * range 
	local target_org = (target[1] + 0.5) * range 
	pred_data[{ i, 1 }] = target_org
	pred_data[{ i, 2 }] = pred_org
	pred_data[{ i, 3 }] = torch.abs(pred_org - target_org)
	mae = mae + torch.abs(pred - target) * range
   mse = mse + torch.pow((pred * range - target * range),2)
  	--mape = mape + (torch.abs(pred - target) / (torch.abs(target)[1] + 0.5))
	mape = mape + (torch.abs(pred_org - target_org) / torch.abs(target_org))
end

--mae = (mae / (#test_data)[1]) * (src_data:max() - src_data:min())
mae = mae / (#test_data)[1]
mse = mse / (#test_data)[1]
mape = mape / (#test_data)[1]

print(pred_data)
print(' MAE = ' .. mae[1])
print(' MSE = ' .. mse[1])
print(' MAPE = ' .. mape)

