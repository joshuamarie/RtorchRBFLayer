box::use(
    torch[
        torch_randn, torch_zeros, torch_exp, 
        nn_module, nn_parameter, nn_linear
    ], 
    rlang[is_function], 
    ./rbf_utils[basis_funcs]
)

#' @export
RBFLayer = nn_module(
    "RBFLayer",
    initialize = function(in_features, out_features, basis_func = 'gaussian') {
        self$in_features = in_features
        self$out_features = out_features
        self$centres = nn_parameter(torch_randn(out_features, in_features))
        self$log_sigmas = nn_parameter(torch_zeros(out_features))
        self$basis_func = if (is_function(basis_func)) {
            basis_func
        } else {
            basis_funcs()[[basis_func]]
        }
    },
    
    forward = function(input) {
        size = c(input$size(1), self$out_features, self$in_features)
        x = input$unsqueeze(2)$expand(size)
        c = self$centres$unsqueeze(1)$expand(size)
        distances = (x - c)$pow(2)$sum(3)$sqrt() / torch_exp(self$log_sigmas)$unsqueeze(1)
        self$basis_func(distances)
    }
)

#' @export
RBFNetwork = nn_module(
    "RBFNetwork",
    initialize = function(in_features, num_classes, num_rbf, basis_func = 'gaussian') {
        self$rbf_layer = RBFLayer(in_features, num_rbf, basis_func)
        self$linear = nn_linear(num_rbf, num_classes)
    },
    
    forward = function(x) {
        out = self$rbf_layer(x)
        self$linear(out)
    }
)
