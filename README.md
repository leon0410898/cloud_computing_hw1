### You can choose jupyter notebook file or python file both are avaliabe.
* For jupyter notebook just simplely run hw1.ipynb
* For python file run main.py file: 
  ```
  python main.py --epochs 1000 \
                 --input_len 32 \
                 --input_dim 313 \
                 --hidden_dim 32 \
                 --sr 16000 \
                 --mode train \
  ```     
  argument: hyper-parameter settings
  * epochs : training epochs
  * input_len : sequence length
  * input_dim : input dimension
  * hidden_dim : hidden dimension
  * sr : sample rate
  * mode : choose train or test mode
  
