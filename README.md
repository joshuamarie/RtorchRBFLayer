
<!-- README.md is generated from README.Rmd. Please edit that file -->

# RtorchRBFLayer

<!-- badges: start -->

[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- badges: end -->

An R torch implementation of Radial Basis Function (RBF) Layers. This
project is a port of
[PyTorchRBFLayer](https://github.com/rssalessio/PytorchRBFLayer) adapted
for the R ecosystem.

Radial Basis Function (RBF) networks can be used to approximate
nonlinear functions and can be combined with other R torch layers to
build hybrid neural architectures.

## What is an RBF Layer?

An RBF layer is defined by five elements:

1.  A radial kernel $\phi: [0,\infty) \to \mathbb{R}$
2.  The number of kernels $N$, and their centers $\{c_i\}_{i=1}^N$
3.  Positive shape parameters $\{\sigma_i\}_{i=1}^N, \sigma_i > 0$
4.  A chosen norm $\|\cdot\|$
5.  A set of learnable weights $\{w_i\}_{i=1}^N$

The output of an RBF layer is computed as:

$$y(x) = \sum_{i=1}^N w_i \cdot \phi(\sigma_i \cdot \|x - c_i\|)$$

where $x$ is the input.

For more background, refer to: - [Radial basis
function](https://en.wikipedia.org/wiki/Radial_basis_function) - [Radial
basis function
network](https://en.wikipedia.org/wiki/Radial_basis_function_network)

## Installation

This project is currently implemented as an independent development
module rather than a standalone R package. You can clone and load it
directly using `box`:

``` bash
git clone https://github.com/joshuamarie/RtorchRBFLayer.git
```

Then, in R, you can load it depending on your working context:

1.  *Within the project directory (`RtorchRBFLayer/`)*

    ``` r
    box::use(rbf = ./RBFLayer)
    ```

2.  *After forking or using it as a submodule*

    ``` r
    box::use(rbf = ./RtorchRBFLayer/RBFLayer)
    ```

A higher-level R package that builds upon this module and extends the
`{torch}` API is currently under development.

## Requirements

The following dependencies are required:

- **R ≥ 4.0**
- **torch** – Core tensor and neural network backend  
- **box** – Modular import and namespace management  
- **caret** *(optional)* – For model evaluation (e.g., confusion
  matrices)

You can install these using:

``` r
install.packages(c("box", "coro", "caret"))
torch::install_torch()  
```

## Usage

Load the module using `box`:

``` r
box::use(
    # rbf = ./RtorchRBFLayer/RBFLayer
    rbf = ./RBFLayer
)
```

Initialize the data:

``` r
dummy_data = data.frame(
    feature1 = rnorm(100),
    feature2 = rnorm(100),
    feature3 = rnorm(100),
    target = sample(1:3, 100, replace = TRUE)
)

dl = rbf$dataset$dataloader(
    x = dummy_data,
    target_col = target,
    feature_cols = c("feature1", "feature2", "feature3"),
    batch_size = 32,
    shuffle = TRUE
)
```

Initialize the RBF model:

``` r
model = rbf$RBFNetwork(
    in_features = 3,      
    num_classes = 3,      
    num_rbf = 10,         
    basis_func = 'gaussian'  
)
```

### Training Example

Specify the loss function and the optimizer:

``` r
box::use(
    torch[nn_cross_entropy_loss, optim_adam]
)

loss_fn = nn_cross_entropy_loss()
optimizer = optim_adam(model$parameters, lr = 0.01)
```

Train RBF network model:

``` r
num_epochs = 50
for (epoch in 1:num_epochs) {
    total_loss = 0
    batches = 0
    
    coro::loop(for (batch in dl) {
        # Forward pass
        predictions = model(batch$x)
        loss = loss_fn(predictions, batch$y$squeeze()$to(torch::torch_long()))
        
        # Backward pass
        optimizer$zero_grad()
        loss$backward()
        optimizer$step()
        
        total_loss = total_loss + loss$item()
        batches = batches + 1
    })
    
    if (epoch %% 10 == 0) {
        cat(sprintf("Epoch %d, Loss: %.4f\n", epoch, total_loss / batches))
    }
}
#> Epoch 10, Loss: 1.0991
#> Epoch 20, Loss: 1.0185
#> Epoch 30, Loss: 1.0041
#> Epoch 40, Loss: 1.0230
#> Epoch 50, Loss: 0.9698
```

### Model Evaluation

Extract the predictions

``` r
X_train = torch::torch_tensor(as.matrix(dummy_data[, c("feature1", "feature2", "feature3")]))
y_train = dummy_data$target

predictions = model(X_train)
predicted_classes = torch::torch_argmax(predictions, dim = 2)$squeeze()
```

Convert them to R vectors:

``` r
pred_vector = as.integer(predicted_classes$cpu())
true_vector = as.integer(y_train)
```

Run confusion matrix with `{caret}` package:

``` r
cm = caret::confusionMatrix(
    factor(pred_vector, levels = 1:3),
    factor(true_vector, levels = 1:3)
)

cm 
#> Confusion Matrix and Statistics
#> 
#>           Reference
#> Prediction  1  2  3
#>          1 18 10 14
#>          2  4 14  5
#>          3 11  7 17
#> 
#> Overall Statistics
#>                                          
#>                Accuracy : 0.49           
#>                  95% CI : (0.3886, 0.592)
#>     No Information Rate : 0.36           
#>     P-Value [Acc > NIR] : 0.005196       
#>                                          
#>                   Kappa : 0.232          
#>                                          
#>  Mcnemar's Test P-Value : 0.352577       
#> 
#> Statistics by Class:
#> 
#>                      Class: 1 Class: 2 Class: 3
#> Sensitivity            0.5455   0.4516   0.4722
#> Specificity            0.6418   0.8696   0.7188
#> Pos Pred Value         0.4286   0.6087   0.4857
#> Neg Pred Value         0.7414   0.7792   0.7077
#> Prevalence             0.3300   0.3100   0.3600
#> Detection Rate         0.1800   0.1400   0.1700
#> Detection Prevalence   0.4200   0.2300   0.3500
#> Balanced Accuracy      0.5936   0.6606   0.5955
```

## Available Basis Functions

The following radial basis functions are implemented:

- `gaussian`: $\phi(\alpha) = \exp(-\alpha^2)$  
- `linear`: $\phi(\alpha) = \alpha$  
- `quadratic`: $\phi(\alpha) = \alpha^2$  
- `inverse_quadratic`: $\phi(\alpha) = \frac{1}{1 + \alpha^2}$  
- `multiquadric`: $\phi(\alpha) = \sqrt{1 + \alpha^2}$  
- `inverse_multiquadric`:
  $\phi(\alpha) = \frac{1}{\sqrt{1 + \alpha^2}}$  
- `spline`: $\phi(\alpha) = \alpha^2 \log(\alpha + 1)$  
- `poisson_one`: $\phi(\alpha) = (\alpha - 1) \exp(-\alpha)$  
- `poisson_two`:
  $\phi(\alpha) = \frac{(\alpha - 2)}{2} \alpha \exp(-\alpha)$  
- `matern32`:
  $\phi(\alpha) = (1 + \sqrt{3}\alpha) \exp(-\sqrt{3}\alpha)$  
- `matern52`:
  $\phi(\alpha) = (1 + \sqrt{5}\alpha + \frac{5}{3}\alpha^2) \exp(-\sqrt{5}\alpha)$

## Important Notes

- **R uses 1-based indexing**: Class indices must start at 1 (unlike
  PyTorch, which uses 0-based indexing).  
- The codebase uses **box** for modular imports to ensure a clean and
  minimal namespace design.

## License

This project is distributed under the MIT License. See the `LICENSE`
file for details.

## Original Author

Original Python Implementation: [Alessio
Russo](https://github.com/rssalessio) (KTH, Sweden)
