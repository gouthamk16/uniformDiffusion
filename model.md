## TON-v1 - Score Entropy Discrete Diffusion Model for Text Generation

Usual transformer models are autoregressive, i.e., it generates one output tokens given a set of input tokens. They make use of causal attention, which masks the sucessive tokens and only considers the tokens (context) until the current token being processed. A text diffusion model works differently, it is not an autoregressive model, it generates a set of output tokens given a set of input tokens. We add an amount of noise to the input tokens and the model predicts the original tokens in that position (similar to how ddpm works). 

