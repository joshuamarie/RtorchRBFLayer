box::use(
    torch[torch_tensor, dataset, torch_dataloader = dataloader, torch_float],
    dplyr[select],
    rlang[enquo, quo_is_null]
)

#' @export
data = function(df, target_col, feature_cols = NULL, dtype = torch_float()) {
    target_quo = enquo(target_col)
    
    y = df |> 
        select(!!target_quo) |> 
        as.matrix()
    
    if (is.null(feature_cols)) {
        X = df |> select(-!!target_quo) |> as.matrix()
    } else {
        feature_quo = enquo(feature_cols)
        X = df |> select(!!feature_quo) |> as.matrix()
    }
    
    X_tensor = torch_tensor(X, dtype = dtype)
    y_tensor = torch_tensor(y, dtype = dtype)
    
    ds = dataset(
        initialize = function() {
            self$X = X_tensor
            self$y = y_tensor
        },
        .getitem = function(i) {
            list(x = self$X[i, ], y = self$y[i, ])
        },
        .length = function() {
            self$X$size(1)
        }
    )
    
    ds()
}

#' @export
dataloader = function(
    x, target_col, feature_cols = NULL, 
    batch_size = 32, shuffle = TRUE, 
    dtype = torch_float()    
) {
    ds = data(x, {{target_col}}, {{feature_cols}}, dtype)
    torch_dataloader(ds, batch_size = batch_size, shuffle = shuffle)
}
