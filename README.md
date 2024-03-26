# DAppFL

A basic implementation of DAppFL.

# Installation

Please ensure that your Python version is above 3.9 :)
And use following commands to install the dependency of DAppFL:

```shell
pip install -r requirements.txt
npm install
```

# Quickly start

Let's try to locate the [SushiSwap fault](https://rekt.news/badgers-digg-sushi/).
Perform fault localization with DAppFL using the following command:
```shell
$ python locate.py \
  --net=Ethereum \
  --fault_txhash=0x90fb0c9976361f537330a5617a404045ffb3fef5972cf67b531386014eeae7a9 \
  --faultless_txhash=0x7df39084b561ee2e7809e690f11e8e258dc65b6128399acbacf1f2433308de6a,0xddd734c1f3e097d3d1cdd7d4c0ffae166b39992a1d055008bf6660b8c0b7582e,0x5c1d151599bbacc19a09dfee888d3be2ccf3e2fa781679b9e0970e18b3300e44
```

> Params description
> 
> - `net`: indicate the blockchain network where dapp is deployed, supporting `Ethereum` and `BNBChain`
> - `fault_txhash`: transaction hash that results in the fault, please `,` for join multiple hash if available
> - `faultless_txhash`: transaction hash that has NOT resulting in the fault, please `,` for join multiple hash if available

You can get the output like this (where Top-0 is the true fault location):

```shell
loading model params...
collecting transaction executing data...
The following code snippets are most likely to cause faults, and the ranking is in descending order of faulty suspiciousness:
Top-0: fault function at contracts/SushiMaker.sol -> 0xe11fc0b43ab98eb91e9836129d1ee7c3bc95df50, offset is 1689:185
Top-1: fault function at contracts/SushiMaker.sol -> 0xe11fc0b43ab98eb91e9836129d1ee7c3bc95df50, offset is 3574:738
Top-2: fault function at contracts/SushiMaker.sol -> 0xe11fc0b43ab98eb91e9836129d1ee7c3bc95df50, offset is 4426:2288
Top-3: fault function at contracts/SushiMaker.sol -> 0xe11fc0b43ab98eb91e9836129d1ee7c3bc95df50, offset is 7944:173
Top-4: fault function at contracts/SushiMaker.sol -> 0xe11fc0b43ab98eb91e9836129d1ee7c3bc95df50, offset is 2988:109
```

# File description

- `algos`: Contains some simple algorithms, such as graph diffusion.
- `compiler`: Contains various versions of the Solc compiler.
- `daos`: The data access object, for data processing.
- `data`: The contributed dataset.
- `dataset`: The code for data modelling.
- `downloader`: Crawl data from the blockchain clients.
- `misc`: Necessary JavaScript files, such as scripts for injecting RPC interfaces.
- `models`: The Graph neural network model.
- `utils`: Some small tools used in the project.
- `main.py`: The main file for starting and evaluating our method, and we save the output to file named `main_output`.

# Train your own DAppFL

You can use `main.py` to train your DAppFL.
There are some parameters for `main.py`:

- `--data_path`: a path for the dataset.
- `--hidden_channels`: the number of neurons for each layer of HGT, default is `32`.
- `--num_layers`: the number of layers for HGT, default is `4`.
- `--num_heads`: the number of attention heads for HGT, default is `1`.
- `--gpu`: Enable GPU for training, default is `False`.
- `--k_folds`: Number of repeated experiments, default is `5`.
- `--epoch`: Number of training rounds, default is `10`.
- `--batch_size`: Batch size during training, default is `4`.
- `--lr`: Learning rate of optimizer, default is `0.001`.
- `--weight_decay`: Weight decay of optimizer, default is `5e-4`.
- `--p_norm`: The Power of Graph Diffusion, default is `6`.