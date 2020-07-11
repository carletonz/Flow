/*
Date					Name					Issue			Comment
10-July-2020			Carleton Zhao							Initialized file
*/

/*********Layers*********/

Blockly.Blocks['linear_layer'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Linear Layer");
    this.appendDummyInput()
        .appendField(new Blockly.FieldNumber(0, 1, 128, 1), "in_dim")
        .appendField("Input Dimension");
    this.appendDummyInput()
        .appendField(new Blockly.FieldNumber(0, 1, 128, 1), "out_dim")
        .appendField("Output Dimention");
    this.setInputsInline(false);
    this.setPreviousStatement(true, "vector");
    this.setNextStatement(true, "vector");
    this.setColour(0);
 this.setTooltip("");
 this.setHelpUrl("");
  }
};

Blockly.Blocks['module'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Module")
        .appendField(new Blockly.FieldTextInput("Net"), "module_name");
    this.appendStatementInput("layers")
        .setCheck("vector");
    this.setColour(0);
 this.setTooltip("");
 this.setHelpUrl("");
  }
};

Blockly.Blocks['conv_2d'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("2D Convolution Layer");
    this.appendDummyInput()
        .appendField("Input Channels")
        .appendField(new Blockly.FieldNumber(0, 0, 24, 1), "in_channels");
    this.appendDummyInput()
        .appendField("Output Channels")
        .appendField(new Blockly.FieldNumber(0, 0, 24, 1), "out_channels");
    this.appendDummyInput()
        .appendField("Kernel Size")
        .appendField(new Blockly.FieldNumber(0, 0, Infinity, 1), "kernel_size");
    this.setPreviousStatement(true, "vector");
    this.setNextStatement(true, "vector");
    this.setColour(0);
 this.setTooltip("");
 this.setHelpUrl("");
  }
};

/*********Operations*********/

Blockly.Blocks['view'] = {
  init: function() {
    this.appendValueInput("output_shape")
        .setCheck("dimension")
        .appendField("View");
    this.setPreviousStatement(true, "vector");
    this.setNextStatement(true, "vector");
    this.setColour(45);
 this.setTooltip("");
 this.setHelpUrl("");
  }
};

Blockly.Blocks['dimension'] = {
  init: function() {
    this.appendValueInput("dim")
        .setCheck("dimension")
        .appendField("Dim")
        .appendField(new Blockly.FieldNumber(0, -1, Infinity, 1), "dim");
    this.setOutput(true, "dimension");
    this.setColour(45);
 this.setTooltip("");
 this.setHelpUrl("");
  }
};

Blockly.Blocks['relu'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("ReLU");
    this.setPreviousStatement(true, "vector");
    this.setNextStatement(true, "vector");
    this.setColour(45);
 this.setTooltip("");
 this.setHelpUrl("");
  }
};

Blockly.Blocks['max_pool_2d'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Max Pool 2D");
    this.appendDummyInput()
        .appendField("Kernel Size")
        .appendField(new Blockly.FieldNumber(0, 0, Infinity, 1), "kernel_size");
    this.appendDummyInput()
        .appendField("Stride")
        .appendField(new Blockly.FieldNumber(0, 0, Infinity, 1), "stride");
    this.setPreviousStatement(true, "vector");
    this.setNextStatement(true, "vector");
    this.setColour(45);
 this.setTooltip("");
 this.setHelpUrl("");
  }
};

/*********Optimizer*********/

Blockly.Blocks['sgd_optimizer'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("SGD Optimizer");
    this.appendDummyInput()
        .appendField("Learning Rate")
        .appendField(new Blockly.FieldNumber(0), "lr");
    this.appendDummyInput()
        .appendField("Momentum")
        .appendField(new Blockly.FieldNumber(0), "momentum");
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(90);
 this.setTooltip("");
 this.setHelpUrl("");
  }
};

/*********Loss*********/

Blockly.Blocks['mse_loss'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("MSE Loss");
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(135);
 this.setTooltip("");
 this.setHelpUrl("");
  }
};

Blockly.Blocks['cross_entropy_loss'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Cross Entropy Loss");
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(135);
 this.setTooltip("");
 this.setHelpUrl("");
  }
};

/*********Data*********/

Blockly.Blocks['iris_data_set'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Iris Data Set");
    this.setInputsInline(false);
    this.setOutput(true, "data");
    this.setColour(180);
 this.setTooltip("");
 this.setHelpUrl("");
  }
};

/*********Train*********/

Blockly.Blocks['train_holdout'] = {
  init: function() {
    this.appendValueInput("training_data")
        .setCheck("data")
        .appendField("Train: Holdout");
    this.appendDummyInput()
        .appendField("Epochs")
        .appendField(new Blockly.FieldNumber(0, 0, 50, 1), "epoch");
    this.appendStatementInput("layers")
        .setCheck("vector");
    this.setInputsInline(false);
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(225);
 this.setTooltip("");
 this.setHelpUrl("");
  }
};