const solc = require('solc%s');
const input = %s;
var output = solc.compile(JSON.stringify(input));
if (output instanceof Object) {
  output = JSON.stringify(output);
}
console.log(output)
