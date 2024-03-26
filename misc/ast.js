const solc_ast = require('solc-typed-ast');
const ASTReader = solc_ast['ASTReader'];

const solc_ast_json_data = %s; // Important! Don't modify it!
var config;
if (solc_ast_json_data['children']) {
    const solc_ast_conf = require('solc-typed-ast/dist/ast/legacy/configuration');
    config = solc_ast_conf['LegacyConfiguration'];
} else {
    const solc_ast_conf = require('solc-typed-ast/dist/ast/modern/configuration');
    config = solc_ast_conf['ModernConfiguration'];
}

const reader = new ASTReader();
const root = reader.convert(solc_ast_json_data, config);
const attr_map = {
    'fullyImplemented': true,
    'isConstructor': true,
    'virtual': true,
    'stateVariable': true,
    'constant': true,
}
const allowed_node_types = {
    'Block': true,
    'Break': true,
    'Continue': true,
    'ContractDefinition': true,
    'DoWhileStatement': true,
    'ElementaryTypeNameExpression': true,
    'EmitStatement': true,
    'EnumDefinition': true,
    'ErrorDefinition': true,
    'EventDefinition': true,
    'ExpressionStatement': true,
    'ForStatement': true,
    'FunctionCall': true,
    'FunctionDefinition': true,
    'IdentifierPath': true,
    'IfStatement': true,
    'ImportDirective': true,
    'InheritanceSpecifier': true,
    'InlineAssembly': true,
    'Literal': true,
    'ModifierDefinition': true,
    'ModifierInvocation': true,
    'NewExpression': true,
    'OverrideSpecifier': true,
    'PlaceholderStatement': true,
    'PragmaDirective': true,
    'Return': true,
    'RevertStatement': true,
    'SourceUnit': true,
    'StructDefinition': true,
    'StructuredDocumentation': true,
    'Throw': true,
    'TryCatchClause': true,
    'TryStatement': true,
    'TupleExpression': true,
    'UncheckedBlock': true,
    'UserDefinedValueTypeDefinition': true,
    'UsingForDirective': true,
    'VariableDeclarationStatement': true,
    'WhileStatement': true,
}

let q = [];
let vis = new Set();
q.push(root);
while (q.length > 0) {
    let node = q.pop();
    if (vis.has(node.id)){
        continue
    }
    vis.add(node.id);

    let attr = {};
    for (const key of Object.keys(node)) {
        let val = node[key];
        if (attr_map[key]) { attr[key] = val; }
    }
    let item = {
        'is_node': true,
        'id': node.id,
        'src': node.src,
        'type': node.type,
    }
    if (Object.keys(attr).length > 0) { item['attr'] = attr; }
    console.log(JSON.stringify(item));
    let children = node.getChildren();
    for (let i = 0; i < children.length; i++) {
        let child = children[i];
        if (!allowed_node_types[child.type]) {
            continue
        }
        q.push(child);
        console.log(JSON.stringify({
            'is_node': false,
            'from': node.id,
            'to': child.id,
            'order': i,
        }))
    }
}
