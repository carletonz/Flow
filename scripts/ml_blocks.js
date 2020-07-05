function print(x){console.log(x)}

function addIndents(code){
  return Blockly.Python.prefixLines(code, Blockly.Python.INDENT);
}

/*********Iris Data Set*********/
Blockly.Blocks['iris_data_set'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Iris Data Set");
    this.setInputsInline(false);
    this.setNextStatement(true, "data");
    this.setColour(230);
 this.setTooltip("");
 this.setHelpUrl("");
  }
};

Blockly.Python['iris_data_set'] = function(block) {
  // TODO: Assemble Python into code variable.
  var code = '...\n';
  return code;
};

/*********Module*********/
Blockly.Blocks['module'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Module")
        .appendField(new Blockly.FieldTextInput("Net"), "module_name");
    this.appendStatementInput("layers")
        .setCheck("vector");
    this.setColour(230);
 this.setTooltip("");
 this.setHelpUrl("");
  }
};

Blockly.Python['module'] = function(block) {
  var text_module_name = block.getFieldValue('module_name');
  //var statements_layers = Blockly.Python.statementToCode(block, 'layers');
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
  
  var init_func = 'def __init__(self):\n'+addIndents(class_def.init);
  var forward_func = 'def forward(self, y):\n'+addIndents(class_def.forward+'return y\n');
  var code = 'class '+text_module_name+'(nn.Module):\n'+addIndents(init_func+forward_func);
  
  return code;
};

/*********Linear Layer*********/
Blockly.Blocks['linear_layer'] = {
  init: function() {
    this.appendDummyInput()
        .appendField('Linear Layer');
    this.appendDummyInput()
        .appendField(new Blockly.FieldNumber(0, 1, 128, 1), 'in_dim')
        .appendField('Input Dimension');
    this.appendDummyInput()
        .appendField(new Blockly.FieldNumber(0, 1, 128, 1), 'out_dim')
        .appendField('Output Dimention');
    this.setInputsInline(false);
    this.setPreviousStatement(true, ['vector', 'data']);
    this.setNextStatement(true, 'vector');
    this.setColour(230);
 this.setTooltip('');
 this.setHelpUrl('');
  }
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
  
  // TODO: Assemble Python into code variable.
  var code = '...\n';
  return code;
};

/*********Convolutional Layer*********/
Blockly.Blocks['conv_2d'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("2D Convolutional Layer");
    this.appendDummyInput()
        .appendField("Input Channels")
        .appendField(new Blockly.FieldNumber(0, 0, 24, 1), "in_channels");
    this.appendDummyInput()
        .appendField("Output Channels")
        .appendField(new Blockly.FieldNumber(0, 0, 24, 1), "out_channels");
    this.appendDummyInput()
        .appendField("Kernel Size")
        .appendField(new Blockly.FieldNumber(0, 0, Infinity, 1), "kernel_size");
    this.setPreviousStatement(true, ['vector', 'data']);
    this.setNextStatement(true, 'vector');
    this.setColour(230);
 this.setTooltip("");
 this.setHelpUrl("");
  }
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
  var code = '...\n';
  return code;
};

/*********View*********/
Blockly.Blocks['view'] = {
  init: function() {
    this.appendValueInput("output_shape")
        .setCheck("dimension")
        .appendField("View");
    this.setPreviousStatement(true, "vector");
    this.setNextStatement(true, "vector");
    this.setColour(230);
 this.setTooltip("");
 this.setHelpUrl("");
  }
};

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
  var code = '...\n';
  return code;
};

/*********Dimention*********/
Blockly.Blocks['dimension'] = {
  init: function() {
    this.appendValueInput("dim")
        .setCheck("dimension")
        .appendField("Dim")
        .appendField(new Blockly.FieldNumber(0, 0, Infinity, 1), "dim");
    this.setOutput(true, "dimension");
    this.setColour(230);
 this.setTooltip("");
 this.setHelpUrl("");
  }
};

Blockly.Python['dimension'] = function(block) {
  var number_dim = block.getFieldValue('dim');
  var value_dim = Blockly.Python.valueToCode(block, 'dim', Blockly.Python.ORDER_NONE);
  // TODO: Assemble Python into code variable.
  var code = String(number_dim) + "," + value_dim;
  // TODO: Change ORDER_NONE to the correct strength.
  return [code, Blockly.Python.ORDER_NONE];
};