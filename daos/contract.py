import asyncio
import json
import os
import platform
import re
from json import JSONDecodeError
from typing import List, Dict, Union

from downloaders.defs import Downloader
from settings import PROJECT_PATH, NODE_PATH, SOLCJS_CODE
from utils.solc import SourceMappingItem, Solc, SolcJS
from utils.tmpfile import wrap_run4tmpfile


class ContractCompileItem:
    def __init__(
            self, contract_address: str, bytecode: str,
            ast: Dict, source_mapping: List[SourceMappingItem],
    ):
        self.contract_address = contract_address
        self.bytecode = bytecode
        self.ast = ast
        self.source_mapping = source_mapping


class ContractDao:
    def __init__(self, downloader: Downloader):
        self.downloader = downloader

    async def get_compile_item(self, contract_address: str) -> ContractCompileItem:
        """
        Compile the contract source code, which is fetched from etherscan.

        :param contract_address: the address of specific contract
        :return: the compiled result
        """
        # fetch source code and save to tmp file
        result = await self.downloader.download(contract_address=contract_address)
        if result.get('SourceCode') is None or result['SourceCode'] == '':
            return ContractCompileItem(contract_address, '', dict(), list())

        # use solc-bin to compile standard-json,
        # and use solc-js to compile one sol source code file
        try:
            json.loads(result['SourceCode'][1:-1])
            product = await self._get_compile_item_by_solc(
                contract_address=contract_address,
                result=result,
            )
            if product is not None:
                return product
        except:
            pass
        finally:
            product = await self._get_compile_item_by_solcjs(
                contract_address=contract_address,
                result=result,
            )
            return product if product is not None else ContractCompileItem(contract_address, '', dict(), list())

    async def _get_compile_item_by_solc(
            self, contract_address: str, result: Dict
    ) -> Union[ContractCompileItem, None]:
        """
        Compile the contract source code by `solc-bin`,
        and return the compiled item if available,
        otherwise return None.

        :param contract_address: the address of specific contract
        :param result: the result of the source code request
        :return: the compiled item or None
        """
        if platform.system().lower() == 'windows':
            version = "solc-windows-amd64-" + result["CompilerVersion"]
            solc_path = os.path.join(PROJECT_PATH, "compiler", version + ".exe")
        elif platform.system().lower() == 'linux':
            version = "solc-linux-amd64-" + result["CompilerVersion"]
            solc_path = os.path.join(PROJECT_PATH, "compiler", version)

        # if there is no source code
        if not os.path.exists(solc_path):
            return None

        contract_name = result["ContractName"]
        try:
            standard_json = json.loads(result['SourceCode'][1:-1])
            standard_json['settings']['outputSelection'] = {
                "*": {
                    "*": ["evm.deployedBytecode"],
                    "": ["ast"],
                },
            }
            product = await wrap_run4tmpfile(
                data=json.dumps(standard_json),
                async_func=lambda p: Solc(solc_path).compile_json(p, contract_name),
            )
        except JSONDecodeError:
            libraries = None
            if result['Library'] != '':
                libraries = result['Library'].split(',')
                libraries = ['{}:0x{}'.format(*lib.split(':')) for lib in libraries]
            product = await wrap_run4tmpfile(
                data=result['SourceCode'].replace('\r\n', '\n'),
                async_func=lambda p: Solc(solc_path).compile_sol(
                    source_path=p,
                    contract_name=contract_name,
                    optimized=result["OptimizationUsed"] == '1',
                    optimize_runs=result["Runs"],
                    libraries=libraries,
                ),

            )

        # return the compilation result
        return ContractCompileItem(
            contract_address=contract_address,
            bytecode=product.bytecode,
            ast=product.ast,
            source_mapping=product.source_mapping
        ) if product is not None else None

    async def _get_compile_item_by_solcjs(
            self, contract_address: str, result: Dict
    ) -> Union[ContractCompileItem, None]:
        """
        Compile the contract source code by `solc-js`,
        and return the compiled item if available,
        otherwise return None.

        :param contract_address: the address of specific contract
        :param result: the result of the source code request
        :return: the compiled item or None
        """

        solc_version = re.search('v(.*?)\+commit', result["CompilerVersion"])
        if solc_version is None:
            return None
        solc_version = 'v%s' % solc_version.group(1)
        contract_name = result["ContractName"]
        _tmp_filename = "this_is_a_tmp_filename.sol"
        try:
            standard_json = json.loads(result['SourceCode'][1:-1])
        except JSONDecodeError:
            standard_json = {
                "language": "Solidity",
                "settings": {
                    "optimizer": {
                        "enabled": result["OptimizationUsed"] == '1',
                        "runs": int(result["Runs"]),
                    },
                },
                "sources": {
                    _tmp_filename: {
                        "content": result['SourceCode'].replace('\r\n', '\n'),
                    }
                }
            }
            if result['Library'] != '':
                libraries = result['Library'].split(',')
                standard_json['settings']['libraries'] = {
                    lib.split(':')[0]: '0x%s' % lib.split(':')[1]
                    for lib in libraries
                }

        # return the compilation result
        standard_json['settings']['outputSelection'] = {
            "*": {
                "*": ["evm.deployedBytecode"],
                "": ["ast"],
            },
        }
        product = await wrap_run4tmpfile(
            data=SOLCJS_CODE % (solc_version, json.dumps(standard_json)),
            async_func=lambda p: SolcJS(NODE_PATH).compile_json(p, contract_name)
        )
        if product is None:
            return None
        for item in product.source_mapping:
            if item.filename == _tmp_filename:
                item.filename = ''
        return ContractCompileItem(
            contract_address=contract_address,
            bytecode=product.bytecode,
            ast={(k if k != _tmp_filename else ''): v for k, v in product.ast.items()},
            source_mapping=product.source_mapping
        ) if product is not None else None

    async def is_contract(self, contract_address: str) -> bool:
        result = await self.downloader.download(contract_address=contract_address)
        return result != '0x'


async def test():
    from downloaders.contract import ContractSourceDownloader, ContractBytecodeDownloader

    dao = ContractDao(
        ContractSourceDownloader('https://api.etherscan.com/api?apikey='))
    item = await dao.get_compile_item('0xd06527d5e56a3495252a528c4987003b712860ee')
    print(item)

    dao = ContractDao(ContractBytecodeDownloader('https://eth-mainnet.nodereal.io/v1/317f6d43dd4c4acea1fa00515cf02f90'))
    print(await dao.is_contract('0x4f6a43ad7cba042606decaca730d4ce0a57ac62e'))
    print(await dao.is_contract('0x6510438a7e273e71300892c6faf946ab3b04cbcb'))


if __name__ == '__main__':
    data = asyncio.run(test())
