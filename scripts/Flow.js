function print(x){console.log(x)}

Flow = {};

Flow.utils = {};

//short cut to add intents to code
Flow.utils.addIndents = function(code){
  
  return Blockly.Python.prefixLines(code, Blockly.Python.INDENT);
};
