# Implementations of Advanced RNNs

Note: Most of these implementations are sourced from GitHub. We have standardized all `forward` functions in these models to conform to the standard RNN API.

- **conv1d**: We wrap the 1D-convolutional layer as a special kind of RNN.
- **gilr**: A linear RNN implemented with Triton.
- **gilr_lstm**: An LSTM architecture simulated using `gilr`.
- **lru**: Linear Recurrent Unit, implemented with Triton.
- **s6**: Mamba, implemented with Triton.
- **smamba**: Mamba, implemented with officially released CUDA code. An additional `start` variable is introduced as an input, resetting the hidden state to 0 when the `start` flag is true.
- **flash_attention**: GPT implemented with flash_attention.
