# solve u''(t) + a u'(t) + b u(t) = f(t)
# where t = [0,1], u(0) = 1, u'(0) = 0

using PyPlot

"""
From the class Jupyter notebook.
"""
function vander_chebyshev(x :: Vector, n :: Int)
    T = ones((length(x), n))
    if n > 1
        T[:,2] = x
    end
    for k in 3:n
        T[:,k] = 2 * x .* T[:,k-1] - T[:,k-2]
    end
    T
end

"""
Create Chebyshev polynomials at points `x` that evaluate u(t), u'(t) and u''(t).
Ported from the class Jupyter notebook.
"""
function chebeval(x :: Vector, n :: Int)
    Tz = vander_chebyshev(x, n)
    dTz = zeros(Tz)
    dTz[:,2] = 1
    dTz[:,3] = 4*x
    ddTz = zeros(Tz)
    ddTz[:,3] = 4
    for k in 4:n
        dTz[:,k]  = k * (2*Tz[:,k-1] + dTz[:,k-2]/(k-2))
        ddTz[:,k] = k * (2*dTz[:,k-1] + ddTz[:,k-2]/(k-2))
    end
    Tz, dTz, ddTz
end

"""
Also from the Jupyter notebook.
"""
cosspace(a, b, n) = (a + b)/2 + (b - a)/2 * (cos.(linspace(-pi, 0, n)))

"""
Construct the finite element matrix for the problem using the Chebyshev method.
"""
function cheb_problem(n :: Int, a, b, f)
    x = cosspace(-1,1,n+1)
    T = chebeval(x, n+1) # u(t), u'(t), u''(t)
    L = T[3] + a * T[2] + b * T[1] # construct u''(t) + a u'(t) + b u(t)
    rhs = f.(x) # f at points x

    # u(0) = 1
    L = vcat(reshape(T[1][1,:],1,:), L)
    rhs = vcat(1, rhs)

    # u'(0) = 0
    L = vcat(reshape(T[2][1,:],1,:), L)
    rhs = vcat(0, rhs)

    x, L * inv(T[1]), rhs
end

f(x) = tanh(x+1) - (x+1) + 1 # f(-1) = 1, f'(-1) = 0
diff(x) = sech(x+1) ^ 2 - 1
diff2(x) = -2 * tanh(x+1) * sech(x+1) ^ 2

a = 0
b = 0

for i in reverse([10,30,50])
    x, L, rhs = cheb_problem(i, a, b, x -> diff2.(x) + a * diff.(x) + b * f.(x))
    figure()
    plot(x, L\rhs, ".")
    xx = linspace(-1,1,100)
    plot(xx, f.(xx))
end