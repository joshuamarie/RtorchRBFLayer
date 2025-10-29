box::use(
    torch[
        torch_ones_like, torch_exp, torch_log
    ]
)

basis_funcs = function() {
    list(
        gaussian = gaussian,
        linear = linear,
        quadratic = quadratic,
        inverse_quadratic = inverse_quadratic,
        multiquadric = multiquadric,
        inverse_multiquadric = inverse_multiquadric,
        spline = spline,
        poisson_one = poisson_one,
        poisson_two = poisson_two,
        matern32 = matern32,
        matern52 = matern52
    )
}

gaussian = function(alpha)
    torch_exp(-1 * alpha$pow(2))

linear = function(alpha) 
    alpha

quadratic = function(alpha) 
    alpha$pow(2)


inverse_quadratic = function(alpha) 
    torch_ones_like(alpha) / (torch_ones_like(alpha) + alpha$pow(2))

multiquadric = function(alpha) 
    (torch_ones_like(alpha) + alpha$pow(2))$pow(0.5)

inverse_multiquadric = function(alpha) 
    torch_ones_like(alpha) / (torch_ones_like(alpha) + alpha$pow(2))$pow(0.5)

spline = function(alpha) 
    alpha$pow(2) * torch_log(alpha + torch_ones_like(alpha))

poisson_one = function(alpha)
    (alpha - torch_ones_like(alpha)) * torch_exp(-alpha)

poisson_two = function(alpha) 
    ((alpha - 2 * torch_ones_like(alpha)) / (2 * torch_ones_like(alpha))) * alpha * torch_exp(-alpha)

matern32 = function(alpha)
    (torch_ones_like(alpha) + 3^0.5 * alpha) * torch_exp(-3^0.5 * alpha)


matern52 = function(alpha) 
    (torch_ones_like(alpha) + 5^0.5 * alpha + (5/3) * alpha$pow(2)) * torch_exp(-5^0.5 * alpha)
