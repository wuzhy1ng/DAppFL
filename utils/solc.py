import asyncio
import json
from asyncio import subprocess
from typing import Dict, List, Union


class SourceMappingItem:
    def __init__(
            self, begin: int, offset: int,
            filename: str, opcode: str, pc: int = -1,
    ):
        self.begin = begin
        self.offset = offset
        self.filename = filename
        self.opcode = opcode
        self.pc = pc

    def __str__(self):
        return "{}:{} {} {} {}".format(
            self.begin,
            self.offset,
            self.filename,
            self.opcode,
            self.pc,
        )


class CompileResult:
    def __init__(self, bytecode: str, ast: Dict[str, Dict], source_mapping: List[SourceMappingItem]):
        self.bytecode = bytecode
        self.ast = ast
        self.source_mapping = source_mapping


class Solc:
    """
    A solidity code compiler, based on `solc-bin`.
    """

    def __init__(self, path: str, timeout: float = 60.0):
        self.path = path
        self.timeout = timeout

    async def compile_json(
            self, standard_json_path: str, contract_name: str
    ) -> Union[CompileResult, None]:
        """
        Compile the source code by standard json.

        :param standard_json_path: The standard json path.
        :param contract_name: Name of target contract
        :return: a compacted result json.
        """
        cmd = [self.path, '--standard-json', standard_json_path]
        process = await subprocess.create_subprocess_shell(
            ' '.join(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            output, err_out = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout,
            )
            output_json = json.loads(output.decode())
        except:
            return None
        contracts = output_json["contracts"]
        sources = output_json["sources"]
        for path in contracts.keys():
            for _contract_name in contracts[path].keys():
                if _contract_name == contract_name:
                    target_contract = contracts[path]
                    break

        # extract the source info, e.g., ast, source id
        ast_dict = {k: v['ast'] for k, v in sources.items()}
        idx2path = {v['id']: k for k, v in sources.items()}

        # extract the compile info, e.g., bytecode, source mapping
        bytecode = target_contract[contract_name]["evm"]["deployedBytecode"]["object"]
        source_mapping_items = self._get_source_mappings(
            opcodes_str=target_contract[contract_name]["evm"]["deployedBytecode"]["opcodes"],
            source_map=target_contract[contract_name]["evm"]["deployedBytecode"]["sourceMap"],
            idx2path=idx2path,
        )

        return CompileResult(bytecode, ast_dict, source_mapping_items)

    async def compile_sol(
            self, source_path: str,
            contract_name: str, optimized: bool,
            optimize_runs: str = '200',
            libraries: Union[List[str], None] = None,
    ) -> Union[CompileResult, None]:
        """
        Compile the source code using the source file directly.

        :param source_path: The source file path.
        :param contract_name: Name of target contract
        :param optimized: whether do an optimization or not
        :param optimize_runs: optimization runs
        :return: a compacted result json.

        """
        cmd = [self.path, '--combined-json', 'bin,ast,opcodes,srcmap,compact-format']
        if optimized is True:
            cmd.extend(['--optimize', '--optimize-runs', optimize_runs])
        if libraries is not None:
            libraries = ','.join(['%s:%s' % (source_path, lib) for lib in libraries])
            cmd.append('--libraries %s' % libraries)
        cmd.append(source_path)
        process = await subprocess.create_subprocess_shell(
            ' '.join(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            output, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout
            )
            output_json = json.loads(output.decode())
        except:
            return None

        # pack and return the result
        # NOTE: use the empty string as the filename,
        # because the filename has not been assigned
        contracts = output_json["contracts"]
        sources = output_json["sources"]
        target = source_path + ':' + contract_name
        bytecode = contracts[target]['bin']
        source_mapping_items = self._get_source_mappings(
            opcodes_str=contracts[target]['opcodes'],
            source_map=contracts[target]['srcmap'],
            idx2path={0: sources},
        )
        for item in source_mapping_items:
            item.filename = ''

        return CompileResult(
            bytecode=bytecode,
            ast={'': sources[source_path]['AST']},
            source_mapping=source_mapping_items
        )

    def _get_source_mappings(
            self, opcodes_str: str, source_map: str,
            idx2path: Dict[int, str],
    ) -> List[SourceMappingItem]:
        # decompile the bytecode
        push2size = {'PUSH%d' % i: i for i in range(1, 32 + 1)}
        source_mapping_items, pc = list(), 0
        opcodes = opcodes_str.split(' ')
        for op in opcodes:
            if op.startswith('0x'):
                continue
            source_mapping_items.append(SourceMappingItem(
                begin=-1, offset=-1, filename='',
                opcode=op, pc=pc,
            ))
            pc += 1 if not op.startswith('PUSH') else 1 + push2size[op]

        # map the pc to source
        prev_s, prev_l, prev_f, prev_j = None, None, None, None
        source_map = source_map.split(';')
        for i, _source_map in enumerate(source_map):
            vals = [prev_s, prev_l, prev_f, prev_j]
            for j, val in enumerate(_source_map.split(':')):
                if val == '' or j >= len(vals) or not val.isdigit():
                    continue
                vals[j] = int(val)
            prev_s, prev_l, prev_f, prev_j = vals
            source_mapping_items[i].begin = vals[0]
            source_mapping_items[i].offset = vals[1]
            if 0 <= vals[2] < len(idx2path):
                source_mapping_items[i].filename = idx2path[vals[2]]

        # return the result
        source_mapping_items = source_mapping_items[: min(len(opcodes), len(source_map))]
        source_mapping_items = [item for item in source_mapping_items if item.filename != '']
        return source_mapping_items


class SolcJS(Solc):
    """
    A solidity code compiler, based on `solc-js`.
    """

    async def compile_json(
            self, standard_json_path: str, contract_name: str
    ) -> Union[CompileResult, None]:
        cmd = [self.path, '--stack-size=4096', standard_json_path]

        process = await subprocess.create_subprocess_shell(
            ' '.join(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            output, err_out = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout,
            )
            product = json.loads(output.decode())
        except:
            return None

        # parse data
        contracts = product.get("contracts", dict())
        sources = product.get("sources", dict())
        if len(contracts) == len(sources) == 0:
            return None

        for path in contracts.keys():
            for _contract_name in contracts[path].keys():
                if _contract_name == contract_name:
                    target_contract = contracts[path]
                    break

        # extract the source info, e.g., ast, source id
        ast_dict = {k: v['ast'] for k, v in sources.items()}
        idx2path = {v['id']: k for k, v in sources.items()}

        # extract the compile info, e.g., bytecode, source mapping
        bytecode = target_contract[contract_name]["evm"]["deployedBytecode"]["object"]
        source_mapping_items = self._get_source_mappings(
            opcodes_str=target_contract[contract_name]["evm"]["deployedBytecode"]["opcodes"],
            source_map=target_contract[contract_name]["evm"]["deployedBytecode"]["sourceMap"],
            idx2path=idx2path,
        )

        return CompileResult(bytecode, ast_dict, source_mapping_items)
