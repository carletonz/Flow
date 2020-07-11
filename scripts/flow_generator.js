/*
Date					Name					Issue			Comment
10-July-2020			Carleton Zhao							Initialized file
*/

/*********Layers*********/

Blockly.Python['module'] = function(block) {
  var text_module_name = block.getFieldValue('module_name');
  var class_def = {
			'init':'super('+text_module_name+', self).__init__()\n',
			'forward':'',
			'length': 0,
			'module':true};
  
  var first_layer = block.getInputTargetBlock('layers');
  if(first_layer){
    first_layer.data = class_def;
    Blockly.Python.blockToCode(first_layer, true);
  }
  else{
    print('Warning: module does not seem to have any layers');
  }
  
  var init_func = 'def __init__(self):\n'+Flow.utils.addIndents(class_def.init);
  var forward_func = 'def forward(self, y):\n'+Flow.utils.addIndents(class_def.forward+'return y\n');
  var code = 'class '+text_module_name+'(nn.Module):\n'+Flow.utils.addIndents(init_func+forward_func);
  return code;
};

Blockly.Python['linear_layer'] = function(block) {
  var number_in_dim = block.getFieldValue('in_dim');
  var number_out_dim = block.getFieldValue('out_dim');
  
  if(!block.data){
    print('Error: no class def');
    return '';
  }
  
  block.data.init += (block.data.module ? 'self.' : '')+'layer'+String(block.data.length)+' = nn.Linear('+String(number_in_dim)+','+String(number_out_dim)+')\n';
  block.data.forward += 'y = '+(block.data.module ? 'self.' : '')+'layer'+String(block.data.length)+'(y)\n';
  block.data.length += 1;
  
  var next_block = block.getNextBlock();
  if(next_block){
    next_block.data = block.data;
    Blockly.Python.blockToCode(next_block, true);
  }
  
  var code = '...;\n';
  return code;
};

Blockly.Python['conv_2d'] = function(block) {
  var number_in_channels = block.getFieldValue('in_channels');
  var number_out_channels = block.getFieldValue('out_channels');
  var number_kernel_size = block.getFieldValue('kernel_size');
  
  if(!block.data){
    print('Error: no class def');
    return '';
  }
  
  block.data.init += (block.data.module ? 'self.' : '')+'layer'+String(block.data.length)+' = nn.Conv2d('+String(number_in_channels)+','+String(number_out_channels)+','+String(number_kernel_size)+')\n';
  block.data.forward += 'y = '+(block.data.module ? 'self.' : '')+'layer'+String(block.data.length)+'(y)\n';
  block.data.length += 1;
  
  var next_block = block.getNextBlock();
  if(next_block){
    next_block.data = block.data;
    Blockly.Python.blockToCode(next_block, true);
  }
  
  // TODO: Assemble Python into code variable.
  var code = '...;\n';
  return code;
};

/*********Operations*********/

Blockly.Python['view'] = function(block) {
  var value_output_shape = Blockly.Python.valueToCode(block, 'output_shape', Blockly.Python.ORDER_NONE);
  
  if(!block.data){
    print('Error: no class def');
    return '';
  }
  
  block.data.forward += 'y = y.view('+value_output_shape+')\n';
  
  var next_block = block.getNextBlock();
  if(next_block){
    next_block.data = block.data;
    Blockly.Python.blockToCode(next_block, true);
  }
  
  // TODO: Assemble Python into code variable.
  var code = '...;\n';
  return code;
};

Blockly.Python['dimension'] = function(block) {
  var number_dim = block.getFieldValue('dim');
  var value_dim = Blockly.Python.valueToCode(block, 'dim', Blockly.Python.ORDER_NONE);
  
  var code = String(number_dim) + "," + value_dim;
  
  return [code, Blockly.Python.ORDER_NONE];
};

Blockly.Python['relu'] = function(block) {
  if(!block.data){
    print('Error: no class def');
    return '';
  }
  
  block.data.forward += 'y = F.relu(y)\n';
  
  var next_block = block.getNextBlock();
  if(next_block){
    next_block.data = block.data;
    Blockly.Python.blockToCode(next_block, true);
  }
  
  var code = '...;\n';
  return code;
};

Blockly.Python['max_pool_2d'] = function(block) {
  var number_kernel_size = block.getFieldValue('kernel_size');
  var number_stride = block.getFieldValue('stride');
  
  if(!block.data){
    print('Error: no class def');
    return '';
  }
  
  // check if module already has a max pooling object with the same kernel size and stride
  if(!block.data['pool'+String(number_kernel_size)+String(number_stride)])
  {
    block.data.init += (block.data.module ? 'self.' : '')+'pool'+String(number_kernel_size)+String(number_stride)+' = nn.MaxPool2d('+String(number_kernel_size)+','+String(number_stride)+')\n';
    block.data['pool'+String(number_kernel_size)+String(number_stride)] = true;
  }
  block.data.forward += 'y = '+(block.data.module ? 'self.' : '')+'pool'+String(number_kernel_size)+String(number_stride)+'(y)\n';
  
  
  var next_block = block.getNextBlock();
  if(next_block){
    next_block.data = block.data;
    Blockly.Python.blockToCode(next_block, true);
  }
  
  // TODO: Assemble Python into code variable.
  var code = '...;\n';
  return code;
};

/*********Optimizer*********/

Blockly.Python['sgd_optimizer'] = function(block) {
  var number_lr = block.getFieldValue('lr');
  var number_momentum = block.getFieldValue('momentum');
  // TODO: should net be call net?
  var code = 'import torch.optim as optim\n';
  code += 'optimizer = optim.SGD(net.parameters(), lr='+String(number_lr)+', momentum='+String(number_momentum)+')\n';
  return code;
};

/*********Loss*********/

Blockly.Python['mse_loss'] = function(block) {
  var code = 'import torch.nn as nn\n';
  code += '_loss = nn.MSELoss()\n';
  return code;
};

Blockly.Python['cross_entropy_loss'] = function(block) {
  var code = 'import torch.nn as nn\n';
  code += '_loss = nn.CrossEntropyLoss()\n';
  return code;
};

/*********Data*********/

Blockly.Python['iris_data_set'] = function(block) {
  // TODO: Assemble Python into code variable.
  var code = '...';
  // TODO: Change ORDER_NONE to the correct strength.
  return [code, Blockly.Python.ORDER_NONE];
};

/*********Train*********/

Blockly.Python['train_holdout'] = function(block) {
  var value_training_data = Blockly.Python.valueToCode(block, 'training_data', Blockly.Python.ORDER_NONE);
  var number_epoch = block.getFieldValue('epoch');
  var statements_layers = Blockly.Python.statementToCode(block, 'layers');
  var first_layer = block.getInputTargetBlock('layers');
  
  if(first_layer){
    first_layer.data = train_def;
    Blockly.Python.blockToCode(first_layer, true);
  }
  else{
    print('Warning: module does not seem to have any layers');
  }
  
  var init_func = train_def.init;
  var forward_func = Flow.utils.addIndents(train_def.forward);
  var code = 'class '+text_module_name+'(nn.Module):\n'+Flow.utils.addIndents(init_func+forward_func);

  var code = '...;\n';
  return code;
};