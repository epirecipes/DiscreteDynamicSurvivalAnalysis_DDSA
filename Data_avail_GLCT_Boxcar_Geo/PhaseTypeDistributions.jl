"""
    PhaseTypeDistributions.jl

A focused Julia package for specific Phase-Type distributions.

## Overview

This package provides implementations of:
- **Continuous Phase-Type Distributions**: Time to absorption in continuous-time Markov chains
- **Discrete Phase-Type Distributions**: Number of steps to absorption in discrete-time Markov chains  
- **Smooth Interpolations**: For continuous parameter support in discrete distributions

## Quick Start

```julia
using PhaseTypeDistributions

# Create continuous phase-type distributions
exp_dist = ph_exponential(2.0)           # Exponential with rate 2
erlang_dist = ph_erlang(3, 1.5)          # Erlang with 3 phases, rate 1.5
erlang_mean = ph_erlang_by_mean(4, 2.0)  # Erlang with 4 phases, mean 2.0

# Create discrete phase-type distributions
geom_dist = dph_geometric(0.3)           # Geometric distribution
nb_dist = dph_negative_binomial(5, 0.4)  # Negative binomial

# Evaluate distributions
pdf(exp_dist, 1.0)     # Probability density
cdf(exp_dist, 1.0)     # Cumulative distribution
mean(exp_dist)         # Expected value
var(exp_dist)          # Variance

# Use in Turing models
@model function my_model()
    ti ~ exp_dist
    # ... rest of model
end
```

### Phase-Type Distributions

A phase-type distribution models the time to absorption in a Markov chain with k transient states
and one absorbing state. It is characterised by:
- **Initial distribution α**: Probability of starting in each transient state
- **Generator matrix A**: Transition rates between transient states (continuous) or transition probabilities (discrete)
- **Absorption rates/probabilities**: Rates of absorption from each transient state

The probability density function for continuous phase-type distributions is:
```
f(t) = α' * exp(A*t) * q
```

## References

- Hurtado, P. J. & Richards, D. (2021). The generalized linear chain trick: a computational method for complex infectious period distributions. Journal of Mathematical Biology, 82(1-2), 1-30. https://doi.org/10.1007/s00285-021-01574-6
"""

module PhaseTypeDistributions

# =============================================================================
# DEPENDENCIES
# =============================================================================

using LinearAlgebra
using Distributions
using ForwardDiff
using SpecialFunctions
using Random

# =============================================================================
# TYPE ALIASES AND CONSTANTS
# =============================================================================

# Type aliases for readability
const RealType = Real
const VectorType{T} = Vector{T}
const MatrixType{T} = Matrix{T}

# Numerical constants
const DEFAULT_TOLERANCE = 1e-10
const MAX_PHASES = 50  # Upper bound for phase-type distributions

# =============================================================================
# CONTINUOUS PHASE-TYPE DISTRIBUTIONS
# =============================================================================

"""
    PhaseType{T<:Real} <: ContinuousUnivariateDistribution

A continuous phase-type distribution representing the time to absorption in a 
finite-state continuous-time Markov chain.

# Fields
- `α::Vector{T}`: Initial probability distribution over transient states (must sum to 1)
- `A::Matrix{T}`: k×k generator matrix for transient states (A[i,j] ≥ 0 for i ≠ j, A[i,i] ≤ 0)
- `q::Vector{T}`: Absorption rates q = -A*1 (computed automatically, all elements ≥ 0)

# Type Parameters
- `T<:Real`: Supports automatic differentiation (e.g., Dual numbers)

# Example
```julia
# Create a 3-phase Erlang distribution
α = [1.0, 0.0, 0.0]  # Start in first phase
A = [-2.0  2.0  0.0;
      0.0 -2.0  2.0;
      0.0  0.0 -2.0]
ph = PhaseType(α, A)
```
"""
struct PhaseType{T<:RealType} <: ContinuousUnivariateDistribution
    α::VectorType{T}  # Initial distribution over transient states
    A::MatrixType{T}  # Generator matrix for transient states
    q::VectorType{T}  # Absorption rates (computed automatically)
end

"""
    PhaseType(α, A)

Construct a continuous phase-type distribution from initial distribution α and generator matrix A.

# Arguments
- `α::AbstractVector{Ta}`: Initial probability distribution over transient states
- `A::AbstractMatrix{Tb}`: Generator matrix for transient states

# Returns
- `PhaseType{Tc}`: Phase-type distribution with promoted element type Tc

# Example
```julia
α = [0.5, 0.3, 0.2]
A = [-1.0  0.5  0.0;
      0.0 -2.0  1.0;
      0.0  0.0 -1.5]
ph = PhaseType(α, A)
```
"""
function PhaseType(α::AbstractVector{Ta}, A::AbstractMatrix{Tb}) where {Ta<:RealType, Tb<:RealType}
    # Type promotion for consistency
    Tc = promote_type(Ta, Tb)
    αp = VectorType{Tc}(α)
    Ap = MatrixType{Tc}(A)
    k = length(αp)
    
    # Validate dimensions
    @assert size(Ap, 1) == k == size(Ap, 2) "Generator matrix A must be k×k where k=length(α)"
    
    # Normalize initial distribution
    @assert all(αp .>= zero(Tc)) "Initial distribution α must be elementwise ≥ 0"
    s = sum(αp)
    @assert s > zero(Tc) "Initial distribution α must have positive sum"
    αp ./= s
    
    # Compute absorption rates
    qp = -Ap * ones(Tc, k)
    @assert all(qp .>= zero(Tc)) "Absorption rates q = -A*1 must be ≥ 0"
    
    return PhaseType{Tc}(αp, Ap, qp)
end

# =============================================================================
# CONTINUOUS PHASE-TYPE: CORE COMPUTATIONS
# =============================================================================

"""
    _ph_pdf(t, α, A, q)

Compute the probability density function for a continuous phase-type distribution.

# Formula
f(t) = α' * exp(A*t) * q

# Arguments
- `t`: Time point
- `α`: Initial distribution
- `A`: Generator matrix
- `q`: Absorption rates

# Returns
- Probability density at time t
"""
function _ph_pdf(t, α, A, q)
    try
        # Special case: Erlang distributions have exact formulas
        if _is_erlang_matrix(A)
            return _erlang_pdf_exact(t, α, A)
        else
            # General case: use matrix exponential
            exp_At = exp(A * t)
            result = (α' * exp_At * q)[]
            
            # Ensure non-negative result
            return max(result, 0.0)
        end
    catch e
        # Return 0 for any numerical errors
        return 0.0
    end
end

"""
    _is_erlang_matrix(A; atol=DEFAULT_TOLERANCE)

Check if a generator matrix A represents an Erlang distribution.

An Erlang matrix has the form:
```
A = [-r  r   0  ...  0;
      0 -r   r  ...  0;
      ... ... ... ... 0;
      0   0   0  ... -r]
```

# Arguments
- `A`: Generator matrix to check
- `atol`: Absolute tolerance for numerical comparisons

# Returns
- `true` if A is an Erlang matrix, `false` otherwise
"""
function _is_erlang_matrix(A; atol=DEFAULT_TOLERANCE)
    k = size(A, 1)
    r = -A[1, 1]
    r > 0 || return false
    
    for i in 1:k
        # Check diagonal elements
        abs(A[i, i] + r) ≤ atol || return false
        
        # Check superdiagonal elements
        if i < k
            abs(A[i, i+1] - r) ≤ atol || return false
        end
        
        # Check that all other elements are zero
        for j in 1:k
            if j != i && j != i+1
                abs(A[i, j]) ≤ atol || return false
            end
        end
    end
    
    return true
end

"""
    _erlang_pdf_exact(t, α, A; atol=DEFAULT_TOLERANCE)

Compute the exact PDF for an Erlang distribution using the analytical formula.

# Formula
f(t) = (r^k * t^(k-1) * exp(-r*t)) / (k-1)!

# Arguments
- `t`: Time point
- `α`: Initial distribution (must be [1,0,...,0] for Erlang)
- `A`: Erlang generator matrix
- `atol`: Absolute tolerance for validation

# Returns
- Exact probability density for Erlang distribution
"""
function _erlang_pdf_exact(t, α, A; atol=DEFAULT_TOLERANCE)
    k = length(α)
    r = -A[1, 1]
    
    # Validate that α is the standard Erlang initial distribution
    (abs(α[1] - 1) ≤ atol && all(abs.(α[2:end]) .≤ atol)) || return 0.0
    t > 0 || return 0.0
    
    # Use log-space computation for numerical stability
    log_pdf = k * log(r) + (k-1) * log(t) - r * t - lgamma(k)
    return exp(log_pdf)
end

"""
    _ph_surv(t, α, A, k)

Compute the survival function for a continuous phase-type distribution.

# Formula
S(t) = α' * exp(A*t) * 1

# Arguments
- `t`: Time point
- `α`: Initial distribution
- `A`: Generator matrix
- `k`: Number of phases

# Returns
- Survival probability at time t
"""
_ph_surv(t, α, A, k) = (α' * exp(A * t) * ones(eltype(α), k))[]

# =============================================================================
# CONTINUOUS PHASE-TYPE: DISTRIBUTIONS.JL INTERFACE
# =============================================================================

# Support function
Distributions.insupport(d::PhaseType, x::RealType) = isfinite(x) && x > 0

# Probability density function
function Distributions.pdf(d::PhaseType, x::RealType)
    x ≤ 0 && return zero(eltype(d.α))
    return _ph_pdf(eltype(d.α)(x), d.α, d.A, d.q)
end

# Log probability density function
function Distributions.logpdf(d::PhaseType, x::RealType)
    x ≤ 0 && return -Inf
    v = _ph_pdf(eltype(d.α)(x), d.α, d.A, d.q)
    return (isfinite(v) && v > 0) ? log(v) : -Inf
end

# Cumulative distribution function
function Distributions.cdf(d::PhaseType, x::RealType)
    x ≤ 0 && return zero(eltype(d.α))
    k = length(d.α)
    S = _ph_surv(eltype(d.α)(x), d.α, d.A, k)
    return one(eltype(d.α)) - S
end

# Log cumulative distribution function
function Distributions.logcdf(d::PhaseType, x::RealType)
    x ≤ 0 && return -Inf
    k = length(d.α)
    S = _ph_surv(eltype(d.α)(x), d.α, d.A, k)
    T = one(eltype(d.α)) - S
    return log(T > 0 ? T : eps(eltype(d.α)))
end

# Mean
function Distributions.mean(d::PhaseType)
    k = length(d.α)
    m = -d.A \ ones(eltype(d.α), k)
    return dot(d.α, m)
end

# Variance
function Distributions.var(d::PhaseType)
    k = length(d.α)
    T1 = -d.A \ ones(eltype(d.α), k)
    T2 = -d.A \ (2 .* T1)  # E[T^2] = 2 α^T (-A)^{-2} 1
    μ = dot(d.α, T1)
    eT2 = dot(d.α, T2)
    return eT2 - μ^2
end

# Convenience functions (also exported)
pdf(d::PhaseType, x::RealType) = Distributions.pdf(d, x)
cdf(d::PhaseType, x::RealType) = Distributions.cdf(d, x)
mean(d::PhaseType) = Distributions.mean(d)
var(d::PhaseType) = Distributions.var(d)

# =============================================================================
# CONTINUOUS PHASE-TYPE: CONSTRUCTOR FUNCTIONS
# =============================================================================

"""
    ph_exponential(λ::Real)

Create an exponential distribution as a 1-phase phase-type distribution.

# Arguments
- `λ::Real`: Rate parameter (must be > 0)

# Returns
- `PhaseType{T}`: Exponential distribution

# Example
```julia
exp_dist = ph_exponential(2.0)  # Rate = 2
mean(exp_dist)  # 0.5
```
"""
function ph_exponential(λ::RealType)
    T = float(λ)
    @assert T > 0 "Rate λ must be > 0"
    
    α = [one(T)]
    A = reshape([-T], 1, 1)
    return PhaseType(α, A)
end

"""
    ph_erlang(k::Integer, r::Real)

Create an Erlang distribution as a k-phase phase-type distribution.

# Arguments
- `k::Integer`: Number of phases (must be ≥ 1)
- `r::Real`: Rate parameter (must be > 0)

# Returns
- `PhaseType{T}`: Erlang distribution

# Example
```julia
erlang_dist = ph_erlang(3, 1.5)  # 3 phases, rate 1.5
mean(erlang_dist)  # 2.0
```
"""
function ph_erlang(k::Integer, r::RealType)
    @assert k ≥ 1 "Number of phases k must be ≥ 1"
    T = float(r)
    @assert T > 0 "Rate r must be > 0"
    
    # Initial distribution: start in first phase
    α = vcat(one(T), zeros(typeof(T), k-1))
    
    # Generator matrix: sequential phases
    A = zeros(typeof(T), k, k)
    @inbounds for i in 1:k
        A[i, i] = -T
        if i < k
            A[i, i+1] = T
        end
    end
    
    return PhaseType(α, A)
end

"""
    ph_erlang_by_mean(k::Integer, μ::Real)

Create an Erlang distribution with specified mean.

# Arguments
- `k::Integer`: Number of phases (must be ≥ 1)
- `μ::Real`: Mean of the distribution (must be > 0)

# Returns
- `PhaseType{T}`: Erlang distribution with mean μ

# Example
```julia
erlang_dist = ph_erlang_by_mean(4, 2.0)  # 4 phases, mean 2.0
mean(erlang_dist)  # 2.0
```
"""
function ph_erlang_by_mean(k::Integer, μ::RealType)
    T = float(μ)
    @assert k ≥ 1 && T > 0 "k must be ≥ 1 and mean μ must be > 0"
    
    r = k / T
    return ph_erlang(k, r)
end

# =============================================================================
# DISCRETE PHASE-TYPE DISTRIBUTIONS
# =============================================================================

# Make DPH work with Turing.jl by implementing Distribution interface
import Distributions: Distribution, Univariate, Discrete

"""
    DPH{T<:Real} <: Distribution{Univariate, Discrete}

A discrete phase-type distribution representing the number of steps to absorption 
in a finite-state discrete-time Markov chain.

# Fields
- `α::Vector{T}`: Initial probability distribution over transient states (must sum to 1)
- `S::Matrix{T}`: k×k sub-stochastic transition matrix (rows sum ≤ 1)
- `s::Vector{T}`: Absorption probabilities per state, s = 1 - S*1

# Type Parameters
- `T<:Real`: Supports automatic differentiation (e.g., Dual numbers)

# Example
```julia
# Create a geometric distribution
α = [1.0]
S = reshape([0.7], 1, 1)  # Stay with probability 0.7
dph = DPH(α, S)
```
"""
struct DPH{T<:RealType} <: Distribution{Univariate, Discrete}
    α::VectorType{T}  # Initial distribution over transient states
    S::MatrixType{T}  # Sub-stochastic transition matrix
    s::VectorType{T}  # Absorption probabilities per state
    
    function DPH(α::VectorType{T}, S::MatrixType{T}) where {T<:RealType}
        k = length(α)
        
        # Validate dimensions
        @assert size(S, 1) == k == size(S, 2) "Transition matrix S must be k×k where k=length(α)"
        
        # Validate initial distribution
        @assert abs(sum(α) - one(T)) < DEFAULT_TOLERANCE "Initial distribution α must sum to 1"
        @assert all(α .>= 0) "Initial distribution α must be non-negative"
        
        # Validate transition matrix
        @assert all(S .>= 0) "Transition matrix S must be non-negative"
        row_sums = sum(S, dims=2)
        @assert all(row_sums .<= 1 .+ eps(T)) "Rows of S must sum ≤ 1"
        
        # Compute absorption probabilities
        s = ones(T, k) .- vec(S * ones(T, k))
        @assert all(s .>= 0) "Absorption probabilities must be non-negative"
        
        new{T}(α, S, s)
    end
end

# =============================================================================
# DISCRETE PHASE-TYPE: CORE COMPUTATIONS
# =============================================================================

"""
    _fundamental(dph::DPH)

Compute the fundamental matrix N = (I - S)^{-1} for a discrete phase-type distribution.

The fundamental matrix gives the expected number of visits to each transient state
before absorption.

# Arguments
- `dph::DPH`: Discrete phase-type distribution

# Returns
- Fundamental matrix N
"""
function _fundamental(dph::DPH{T}) where {T}
    k = length(dph.α)
    I_matrix = MatrixType{eltype(dph.α)}(LinearAlgebra.I, k, k)
    return inv(I_matrix - dph.S)
end

"""
    mean(dph::DPH)

Compute the mean of a discrete phase-type distribution.

# Formula
μ = α' * N * 1

# Arguments
- `dph::DPH`: Discrete phase-type distribution

# Returns
- Mean of the distribution
"""
function mean(dph::DPH{T}) where {T}
    k = length(dph.α)
    N = _fundamental(dph)
    μ = (dph.α' * (N * ones(eltype(dph.α), k)))[1]
    return μ
end

"""
    var(dph::DPH)

Compute the variance of a discrete phase-type distribution.

# Formula
`Var(X) = 2 α' N^2 S 1 + μ - μ^2` with `N = (I-S)^{-1}`, i.e. `E[X(X-1)] = 2 α' N^2 S 1` (second factorial moment) plus `μ - μ^2`.

# Arguments
- `dph::DPH`: Discrete phase-type distribution

# Returns
- Variance of the distribution
"""
function var(dph::DPH{T}) where {T}
    k = length(dph.α)
    N = _fundamental(dph)
    onev = ones(eltype(dph.α), k)
    μ = (dph.α' * (N * onev))[1]
    term = (dph.α' * (N * (N * (dph.S * onev))))[1]  # α' N^2 S 1
    return 2 * term + μ - μ^2
end

"""
    pmf(dph::DPH, n::Integer)

Compute the probability mass function for a discrete phase-type distribution.

# Formula
pmf(n) = α' * S^(n-1) * s, n ≥ 1

# Arguments
- `dph::DPH`: Discrete phase-type distribution
- `n::Integer`: Number of steps (must be ≥ 1)

# Returns
- Probability mass at n
"""
function pmf(dph::DPH{T}, n::Integer) where {T}
    @assert n ≥ 1 "DPH support starts at n = 1"
    
    if n == 1
        return (dph.α' * dph.s)[1]
    else
        return (dph.α' * (dph.S^(n-1) * dph.s))[1]
    end
end

"""
    cdf(dph::DPH, n::Integer)

Compute the cumulative distribution function for a discrete phase-type distribution.

# Formula
cdf(n) = 1 - α' * S^n * 1, n ≥ 1

# Arguments
- `dph::DPH`: Discrete phase-type distribution
- `n::Integer`: Number of steps (must be ≥ 1)

# Returns
- Cumulative probability at n
"""
function cdf(dph::DPH{T}, n::Integer) where {T}
    @assert n ≥ 1 "DPH support starts at n = 1"
    k = length(dph.α)
    return 1 - (dph.α' * (dph.S^n * ones(eltype(dph.α), k)))[1]
end

"""
    Distributions.pdf(dph::DPH{T}, x::Integer) where {T}

Compute the probability mass function.

# Arguments
- `dph::DPH{T}`: Discrete phase-type distribution
- `x::Integer`: Value to evaluate (must be ≥ 1)

# Returns
- Probability mass at x
"""
function Distributions.pdf(dph::DPH{T}, x::Integer) where {T}
    return pmf(dph, x)
end

"""
    Distributions.logpdf(dph::DPH{T}, x::Integer) where {T}

Compute the log probability mass function for Turing.jl compatibility.

# Arguments
- `dph::DPH{T}`: Discrete phase-type distribution
- `x::Integer`: Value to evaluate (must be ≥ 1)

# Returns
- Log probability mass at x
"""
function Distributions.logpdf(dph::DPH{T}, x::Integer) where {T}
    k = length(dph.s)
    # NegBin-chain detection: only the last transient state has positive absorption
    # probability (s[1:k-1] == 0 exactly as constructed). Covers geometric (k=1)
    # and NegBin (k=r) DPH. Uses scalar arithmetic — ForwardDiff-safe, no matrix power.
    if all(i -> iszero(dph.s[i]), 1:k-1)
        return negbin_total_logpdf(x, k, dph.s[k])
    end
    pmf_val = pmf(dph, x)
    return log(max(pmf_val, 1e-12))
end

# Also define logpdf without Distributions prefix for convenience
function logpdf(dph::DPH{T}, x::Integer) where {T}
    k = length(dph.s)
    if all(i -> iszero(dph.s[i]), 1:k-1)
        return negbin_total_logpdf(x, k, dph.s[k])
    end
    pmf_val = pmf(dph, x)
    return log(max(pmf_val, 1e-12))
end

"""
    negbin_total_logpdf(k, r, p)

Analytical log-PMF for the NegBin DPH parameterisation counting TOTAL trials
(support {r, r+1, ...}): P(W=k) = C(k-1,r-1) * p^r * (1-p)^(k-r).

Uses scalar arithmetic only — does NOT construct or power the DPH transition
matrix, making it safe for ForwardDiff AD when `p` is a Dual number.
"""
function negbin_total_logpdf(k::Integer, r::Integer, p::Real)
    k < r && return oftype(float(p), -Inf)
    T  = typeof(float(p))
    lc = SpecialFunctions.logabsbinomial(k - 1, r - 1)[1]
    return T(lc) + r * log(T(p)) + (k - r) * log(one(T) - T(p))
end

"""
    negbin_total_logpdf(k, r::Real, p)

Real-valued `r` overload.  Generalises the NegBin log-PMF (total-trials
parameterisation, support {ceil(r), ceil(r)+1, ...}) to continuous r > 0 using
`loggamma` instead of `logabsbinomial`.  The integer overload uses the exact
binomial coefficient; this version enables NUTS to differentiate through `r`
via ForwardDiff — `SpecialFunctions.loggamma` carries Dual partials correctly.

  log P(W=k) = loggamma(k) - loggamma(r) - loggamma(k-r+1) + r*log(p) + (k-r)*log(1-p)
"""
function negbin_total_logpdf(k::Integer, r::Real, p::Real)
    T  = promote_type(typeof(float(r)), typeof(float(p)))
    kr = T(k) - T(r)
    kr < 0 && return T(-Inf)
    lc = SpecialFunctions.loggamma(T(k)) -
         SpecialFunctions.loggamma(T(r)) -
         SpecialFunctions.loggamma(kr + one(T))
    return lc + T(r) * log(T(p)) + kr * log(one(T) - T(p))
end

# =============================================================================
# DISCRETE PHASE-TYPE: DISTRIBUTIONS.JL INTERFACE
# =============================================================================

# Support: DPH distributions have support on positive integers ≥ 1
function Distributions.support(dph::DPH{T}) where {T}
    return 1:typemax(Int)
end

function Distributions.minimum(dph::DPH{T}) where {T}
    return 1
end

function Distributions.maximum(dph::DPH{T}) where {T}
    return Inf
end

function Distributions.insupport(dph::DPH{T}, x) where {T}
    return isa(x, Integer) && x >= 1
end

# Convenience functions without Distributions prefix
function support(dph::DPH{T}) where {T}
    return 1:typemax(Int)
end

function minimum(dph::DPH{T}) where {T}
    return 1
end

function maximum(dph::DPH{T}) where {T}
    return Inf
end

function insupport(dph::DPH{T}, x) where {T}
    return isa(x, Integer) && x >= 1
end

# Element type for sampling
Base.eltype(::Type{DPH{T}}) where {T} = Int
Base.eltype(::DPH{T}) where {T} = Int

"""
    rand(rng::AbstractRNG, dph::DPH{T}) where {T}

Generate a random sample from a discrete phase-type distribution using inverse CDF.

# Arguments
- `rng::AbstractRNG`: Random number generator
- `dph::DPH{T}`: Discrete phase-type distribution

# Returns
- Number of steps until absorption (Integer ≥ 1)
"""
function Random.rand(rng::AbstractRNG, dph::DPH{T}) where {T}
    u = Random.rand(rng)
    cum_prob = 0.0
    n = 1
    
    # Find the sample using inverse CDF
    while cum_prob < u
        pmf_val = pmf(dph, n)
        cum_prob += pmf_val
        if cum_prob >= u
            break
        end
        n += 1
    end
    
    return n
end

# Base.rand method for compatibility
function Base.rand(rng::AbstractRNG, dph::DPH{T}) where {T}
    u = Random.rand(rng)
    cum_prob = 0.0
    n = 1
    
    # Find the sample using inverse CDF
    while cum_prob < u
        pmf_val = pmf(dph, n)
        cum_prob += pmf_val
        if cum_prob >= u
            break
        end
        n += 1
    end
    
    return n
end

# Define rand without rng parameter (uses default RNG)
function Base.rand(dph::DPH{T}) where {T}
    u = Random.rand(Random.GLOBAL_RNG)
    cum_prob = 0.0
    n = 1
    
    # Find the sample using inverse CDF
    while cum_prob < u
        pmf_val = pmf(dph, n)
        cum_prob += pmf_val
        if cum_prob >= u
            break
        end
        n += 1
    end
    
    return n
end

# Distributions.jl interface for rand (Turing.jl compatibility)
# NOTE: must NOT delegate to Base.rand/Random.rand — Distributions.jl intercepts
# all Random.rand(rng, ::Sampleable) and redirects to Distributions.rand, causing
# infinite recursion. Inline the CDF sampling directly instead.
function Distributions.rand(rng::AbstractRNG, dph::DPH{T}) where {T}
    u = Random.rand(rng)
    cum_prob = 0.0
    n = 1
    while cum_prob < u
        pmf_val = pmf(dph, n)
        cum_prob += pmf_val
        cum_prob >= u && break
        n += 1
    end
    return n
end

# =============================================================================
# DISCRETE PHASE-TYPE: CONSTRUCTOR FUNCTIONS
# =============================================================================

"""
    dph_geometric(p::Real)

Create a geometric distribution as a 1-phase discrete phase-type distribution.

# Arguments
- `p::Real`: Success probability (must be in (0,1])

# Returns
- `DPH{T}`: Geometric distribution

# Example
```julia
geom_dist = dph_geometric(0.3)  # Success probability 0.3
mean(geom_dist)  # 1/0.3 ≈ 3.33
```
"""
function dph_geometric(p::RealType)
    @assert 0 < p ≤ 1 "Success probability p must be in (0,1]"
    
    α = [one(p)]
    S = reshape([one(p) - p], 1, 1)
    return DPH(α, S)
end

"""
    dph_negative_binomial(r::Integer, p::Real)

Create a negative binomial distribution as a discrete phase-type distribution.

# Arguments
- `r::Integer`: Number of successes required (must be ≥ 1)
- `p::Real`: Success probability per trial (must be in (0,1])

# Returns
- `DPH{T}`: Negative binomial distribution

# Example
```julia
nb_dist = dph_negative_binomial(5, 0.4)  # 5 successes, prob 0.4
mean(nb_dist)  # r/p = 5/0.4 = 12.5  (total-trials convention: mean = r/p)
```
"""
function dph_negative_binomial(r::Integer, p::RealType)
    @assert r ≥ 1 "Number of successes r must be ≥ 1"
    @assert 0 < p ≤ 1 "Success probability p must be in (0,1]"
    
    k = r
    # Start in state 1 (0 successes), but DPH indexing starts at 1
    α = vcat(one(p), zeros(eltype(p), k-1))
    S = zeros(eltype(p), k, k)
    
    for i in 1:k-1
        S[i, i] = 1 - p      # Stay in state i (failure)
        S[i, i+1] = p        # Move to state i+1 (success)
    end
    S[k, k] = 1 - p          # Final state: stay with prob 1-p, absorb with prob p
    
    return DPH(α, S)
end

# Support for ForwardDiff.Dual r — extract integer value, keep p as-is so gradients flow
function dph_negative_binomial(r::T, p::RealType) where T <: ForwardDiff.Dual
    r_int = round(Int, ForwardDiff.value(r))
    @assert r_int ≥ 1 "Number of successes r must be ≥ 1"
    @assert 0 < p ≤ 1 "Success probability p must be in (0,1]"

    k = r_int
    α = vcat(one(p), zeros(eltype(p), k-1))
    S = zeros(eltype(p), k, k)

    for i in 1:k-1
        S[i, i] = 1 - p
        S[i, i+1] = p
    end
    S[k, k] = 1 - p

    return DPH(α, S)
end

# Support for integer r with dual p — keep p as Dual so gradient flows through nb_p
function dph_negative_binomial(r::Integer, p::T) where T <: Union{ForwardDiff.Dual, RealType}
    @assert r ≥ 1 "Number of successes r must be ≥ 1"

    p_val = ForwardDiff.value(p)
    @assert 0 < p_val ≤ 1 "Success probability p must be in (0,1]"

    k = r
    α = vcat(one(p), zeros(eltype(p), k-1))
    S = zeros(eltype(p), k, k)

    for i in 1:k-1
        S[i, i] = 1 - p      # keep Dual: gradient of DPH w.r.t. nb_p flows through
        S[i, i+1] = p
    end
    S[k, k] = 1 - p

    return DPH(α, S)
end

"""
    dph_prepend_deterministic_delay(dph, c)

Return a new `DPH` that prepends `c` **deterministic** transient states in front of the
core chain described by `dph`. Each prepended state advances to the next with probability
1 on the next time step (no absorption from those states); after `c` steps the process
enters the original phase-type states according to `dph.α` and then evolves with `dph.S`.

Total time to absorption increases by exactly `c` steps relative to `dph`:
``\\mathbb{E}[T_{\\mathrm{new}}] = c + \\mathbb{E}[T_{\\mathrm{core}}]``.

Use this when the line list defines a **calendar** dwell time with a **known minimum**
(e.g. fixed clinical offsets) while the stochastic tail remains phase-type, or for any
**delayed start** before a generic DPH clock runs. For any other minimum-support adjustment,
compose `dph_prepend_deterministic_delay` with your chosen core `DPH` (Erlang, fitted PH, …).

# Arguments
- `dph::DPH`: Core discrete phase-type distribution
- `c::Integer`: Non-negative number of one-step delays (`c = 0` returns `dph` unchanged)

# Example
```julia
core = dph_negative_binomial(3, 0.25)   # core minimum index r = 3
delayed = dph_prepend_deterministic_delay(core, 2)  # minimum time index 3+2 = 5
```
"""
function dph_prepend_deterministic_delay(dph::DPH{T}, c::Integer) where {T}
    @assert c ≥ 0 "c (deterministic delay steps) must be ≥ 0, got $c"
    if iszero(c)
        return dph
    end
    kr = length(dph.α)
    Te = promote_type(eltype(dph.α), eltype(dph.S))
    k = c + kr
    S = zeros(Te, k, k)
    for i in 1:(c - 1)
        S[i, i + 1] = one(Te)
    end
    for j in 1:kr
        S[c, c + j] = Te(dph.α[j])
    end
    for i in 1:kr, j in 1:kr
        S[c + i, c + j] = Te(dph.S[i, j])
    end
    α = zeros(Te, k)
    α[1] = one(Te)
    return DPH(α, S)
end

"""
    dph_negative_binomial_min_support(r, p, δ_min)

Convenience for **integer** dwell times with a known minimum `δ_min` (e.g. Neal & Roberts
infectious period with a fixed offset so all observed `δ ≥ δ_min`).

`dph_negative_binomial(r, p)` has support `\\{r, r+1, \\ldots\\}`. If `δ_min ≥ r`, this returns
`dph_prepend_deterministic_delay(dph_negative_binomial(r, p), δ_min - r)`, so the minimum
time index is `δ_min` and the mean is `(δ_min - r) + r/p`.

Requires `δ_min ≥ r` (equivalently `δ_min - r` non-negative).

See also `dph_prepend_deterministic_delay`.
"""
function dph_negative_binomial_min_support(r::Integer, p, δ_min::Integer)
    @assert δ_min ≥ r "δ_min ($δ_min) must be ≥ r ($r)"
    dph_prepend_deterministic_delay(dph_negative_binomial(r, p), δ_min - r)
end

function dph_negative_binomial_min_support(r::Tr, p::RealType, δ_min::Integer) where Tr <: ForwardDiff.Dual
    r_int = round(Int, ForwardDiff.value(r))
    @assert δ_min ≥ r_int "δ_min ($δ_min) must be ≥ r ($r_int)"
    dph_prepend_deterministic_delay(dph_negative_binomial(r, p), δ_min - r_int)
end

"""
    dph_erlang(k::Integer, μ::Real, Δt::Real=1.0)

Create a discrete Erlang distribution by discretizing a continuous Erlang distribution.

# Arguments
- `k::Integer`: Number of phases (must be ≥ 1)
- `μ::Real`: Mean of the continuous distribution (must be > 0)
- `Δt::Real`: Time step size for discretization (must be > 0)

# Returns
- `DPH{T}`: Discrete Erlang distribution

# Example
```julia
dph_erlang_dist = dph_erlang(3, 2.0, 0.1)  # 3 phases, mean 2.0, step 0.1
```
"""
function dph_erlang(k::Integer, μ::RealType, Δt::RealType=1.0)
    @assert k ≥ 1 "Number of phases k must be ≥ 1"
    @assert μ > 0 "Mean μ must be > 0"
    @assert Δt > 0 "Time step Δt must be > 0"

    # Delegate to dph_from_rates which uses scalar per-entry discretization
    # (avoids exp(A*Δt) on a matrix of Dual numbers, which is not ForwardDiff-safe)
    r = k / μ
    return dph_erlang_from_rates(k, r, Δt)
end

# =============================================================================
# DISCRETE PHASE-TYPE: CONVERSION FROM CONTINUOUS RATES
# =============================================================================

"""
    rate_to_proportion(r, t=1.0)

Convert a continuous rate to a transition probability for discretisation.

# Formula
p = 1 - exp(-r * t)

# Arguments
- `r`: Continuous rate
- `t`: Time step size

# Returns
- Transition probability
"""
@inline function rate_to_proportion(r, t=1.0)
    if r < 0 || isnan(r)
        return 0.0
    elseif isinf(r)
        return 1.0
    else
        result = 1 - exp(-r * t)
        # Ensure result is valid
        return isnan(result) ? 0.0 : max(min(result, 1.0), 0.0)
    end
end

"""
    dph_from_rates(α, A, Δt=1.0)

Create a discrete phase-type distribution from continuous-time rates by discretization.

# Arguments
- `α::AbstractVector{Ta}`: Initial probability distribution over transient states
- `A::AbstractMatrix{Tb}`: Continuous-time generator matrix
- `Δt::Real`: Time step size for discretization

# Returns
- `DPH{Tc}`: Discrete phase-type distribution

# Example
```julia
α = [1.0, 0.0]
A = [-2.0  1.0;
      0.0 -1.0]
dph = dph_from_rates(α, A, 0.1)
```
"""
function dph_from_rates(α::AbstractVector{Ta}, A::AbstractMatrix{Tb}, Δt::RealType=1.0) where {Ta<:RealType, Tb<:RealType}
    Tc = promote_type(Ta, Tb, typeof(Δt))
    αp = VectorType{Tc}(α)
    Ap = MatrixType{Tc}(A)
    k = length(αp)
    
    # Validate dimensions
    @assert size(Ap, 1) == k == size(Ap, 2) "Generator matrix A must be k×k where k=length(α)"
    @assert all(αp .>= zero(Tc)) "Initial distribution α must be elementwise ≥ 0"
    
    # Normalize initial distribution
    s = sum(αp)
    @assert s > zero(Tc) "Initial distribution α must have positive sum"
    αp ./= s
    
    # Convert generator matrix A to transition matrix S
    S = zeros(Tc, k, k)
    for i in 1:k
        # Calculate total exit rate from state i
        total_exit_rate = -A[i, i]
        
        # Probability of staying in state i (no transition)
        S[i, i] = exp(-total_exit_rate * Δt)
        
        # Probabilities of transitioning to other states
        for j in 1:k
            if i != j && A[i, j] > 0
                # Probability of transitioning from i to j
                S[i, j] = (A[i, j] / total_exit_rate) * (1 - exp(-total_exit_rate * Δt))
            end
        end
    end
    
    return DPH(αp, S)
end

"""
    dph_exponential(λ, Δt=1.0)

Create a discrete geometric distribution from exponential rate λ.

# Arguments
- `λ::Real`: Rate parameter of the exponential distribution
- `Δt::Real`: Time step size for discretization

# Returns
- `DPH{T}`: Discrete phase-type distribution (geometric)

# Example
```julia
dph_exp = dph_exponential(2.0, 0.1)  # Rate 2, step 0.1
```
"""
function dph_exponential(λ::RealType, Δt::RealType=1.0)
    @assert λ > 0 "Rate λ must be > 0"
    T = promote_type(typeof(λ), typeof(Δt))
    α = [one(T)]
    A = reshape([-T(λ)], 1, 1)
    return dph_from_rates(α, A, Δt)
end

"""
    dph_erlang_from_rates(k, r, Δt=1.0)

Create a discrete Erlang distribution from continuous Erlang parameters.

# Arguments
- `k::Integer`: Number of phases
- `r::Real`: Rate parameter
- `Δt::Real`: Time step size for discretization

# Returns
- `DPH{T}`: Discrete phase-type distribution

# Example
```julia
dph_erlang = dph_erlang_from_rates(3, 1.5, 0.1)
```
"""
function dph_erlang_from_rates(k::Integer, r::RealType, Δt::RealType=1.0)
    @assert k ≥ 1 "Number of phases k must be ≥ 1"
    T = promote_type(typeof(r), typeof(Δt))
    @assert T(r) > 0 "Rate r must be > 0"
    
    α = vcat(one(T), zeros(T, k-1))
    A = zeros(T, k, k)
    
    for i in 1:k
        A[i, i] = -T(r)
        if i < k
            A[i, i+1] = T(r)
        end
    end
    
    return dph_from_rates(α, A, Δt)
end

"""
    dph_erlang_from_mean(k, μ, Δt=1.0)

Create a discrete Erlang distribution with specified mean.

# Arguments
- `k::Integer`: Number of phases
- `μ::Real`: Mean of the distribution
- `Δt::Real`: Time step size for discretization

# Returns
- `DPH{T}`: Discrete phase-type distribution

# Example
```julia
dph_erlang = dph_erlang_from_mean(4, 2.0, 0.1)
```
"""
function dph_erlang_from_mean(k::Integer, μ::RealType, Δt::RealType=1.0)
    @assert k ≥ 1 "Number of phases k must be ≥ 1"
    T = promote_type(typeof(μ), typeof(Δt))
    @assert T(μ) > 0 "Mean μ must be > 0"
    
    r = k / μ
    return dph_erlang_from_rates(k, r, Δt)
end

"""
    dph_negative_binomial_from_rates(r, p, Δt=1.0)

Create a discrete negative binomial distribution from continuous-time parameters.

# Arguments
- `r::Integer`: Number of successes required
- `p::Real`: Success probability per trial
- `Δt::Real`: Time step size for discretization

# Returns
- `DPH{T}`: Discrete phase-type distribution (negative binomial)

# Example
```julia
dph_nb = dph_negative_binomial_from_rates(5, 0.4, 0.1)
```
"""
function dph_negative_binomial_from_rates(r::Integer, p::RealType, Δt::RealType=1.0)
    @assert r ≥ 1 "Number of successes r must be ≥ 1"
    T = promote_type(typeof(p), typeof(Δt))
    @assert T(p) > 0 && T(p) < 1 "Success probability p must be in (0,1)"
    
    # Model as sequential chain of r exponential phases, each with rate λ = -log(1-p)
    λ = -log(1 - T(p))

    # Create generator matrix for sequential r-phase chain (hypoexponential / Erlang)
    A = zeros(T, r, r)
    for i in 1:r
        A[i, i] = -λ
        if i < r
            A[i, i+1] = λ   # advance to next phase
        end
    end

    # Initial distribution: start in first phase
    α = vcat(one(T), zeros(T, r-1))

    return dph_from_rates(α, A, Δt)
end

"""
    dph_negative_binomial_from_mean(r, μ)

Create a discrete negative binomial DPH with specified mean (total-trials convention,
same construction as `dph_negative_binomial`: no time step).

For a mean-parameterised distribution built by discretising a continuous-time
Erlang / success-counting chain, use `dph_negative_binomial_from_rates` with
the desired `Δt`.

# Arguments
- `r::Integer`: Number of successes required
- `μ::Real`: Mean of the distribution

# Returns
- `DPH{T}`: Discrete phase-type distribution

# Example
```julia
dph_nb = dph_negative_binomial_from_mean(5, 12.5)  # r=5, mean=12.5 → p = 5/12.5 = 0.4
```
"""
function dph_negative_binomial_from_mean(r::Integer, μ::RealType)
    @assert r ≥ 1 "Number of successes r must be ≥ 1"
    T = typeof(float(μ))
    @assert T(μ) > 0 "Mean μ must be > 0"

    # Total-trials NegBin: μ = r/p, so p = r/μ
    p = T(r) / T(μ)
    @assert 0 < p ≤ 1 "Mean μ must be ≥ r; got μ=$μ with r=$r"
    return dph_negative_binomial(r, p)
end


# =============================================================================
# SMOOTH INTERPOLATION FOR DISCRETE PHASE-TYPE DISTRIBUTIONS
# =============================================================================

"""
    SmoothDPH{T, W} <: Distribution{Univariate, Discrete}

Structure to hold two neighboring discrete phase-type distributions and their
interpolation weight for smooth parameter support.

# Fields
- `dph_k::DPH{T}`: Discrete phase-type distribution with k phases
- `dph_k1::DPH{T}`: Discrete phase-type distribution with k+1 phases
- `w::W`: Interpolation weight

# Type Parameters
- `T`: Element type of the DPH distributions
- `W`: Type of the interpolation weight
"""
struct SmoothDPH{T, W} <: Distribution{Univariate, Discrete}
    dph_k::DPH{T}   # DPH with k phases
    dph_k1::DPH{T}  # DPH with k+1 phases
    w::W            # Interpolation weight
end

"""
    dph_negative_binomial_smooth(r::Real, p::Real)

Create a smooth interpolation of discrete phase-type negative binomial distributions.
This allows `r` to be any positive real number, similar to how Distributions.jl handles
the negative binomial distribution.

The function interpolates between neighboring integer phase-type distributions:
- For r = k + w where k is integer and w ∈ [0,1)
- Uses linear interpolation: (1-w) * DPH_k + w * DPH_{k+1}

This maintains smooth gradients for automatic differentiation while preserving
the discrete phase-type structure.

# Arguments
- `r::Real`: Number of successes (can be non-integer)
- `p::Real`: Success probability (must be in (0,1])

# Returns
- `SmoothDPH{T, W}`: Smoothly interpolated distribution

# Example
```julia
smooth_nb = dph_negative_binomial_smooth(3.7, 0.4)  # Non-integer r
mean(smooth_nb)  # Interpolated mean
```
"""
function dph_negative_binomial_smooth(r::RealType, p::RealType)
    @assert r > 0 "Number of successes r must be positive"
    @assert 0 < p ≤ 1 "Success probability p must be in (0,1]"
    
    # Clamp r to prevent extreme values (use clamped value for k and w so w ∈ [0,1))
    r_clamped = clamp(r, 1.0, MAX_PHASES)
    
    # Interpolation: r_clamped = k + w with k = ⌊r_clamped⌋, w ∈ [0,1)
    k = floor(Int, r_clamped)
    w = r_clamped - k
    
    # Ensure k >= 1 and reasonable upper bound
    k = max(1, k)
    if k >= MAX_PHASES
        k = MAX_PHASES
        w = 0.0
    end
    
    # Create neighboring phase-type distributions
    dph_k = dph_negative_binomial(k, p)
    dph_k1 = dph_negative_binomial(k+1, p)
    
    # Ensure w has the correct type to match the DPH elements
    T_eltype = eltype(dph_k.α)
    w_typed = convert(promote_type(T_eltype, eltype(w)), w)
    
    return SmoothDPH(dph_k, dph_k1, w_typed)
end

# Support for ForwardDiff.Dual in smooth DPH
function dph_negative_binomial_smooth(r::T, p::RealType) where T <: Union{ForwardDiff.Dual, RealType}
    # Extract primal values for bounds checking
    r_val = ForwardDiff.value(r)
    p_val = ForwardDiff.value(p)
    
    @assert r_val > 0 "Number of successes r must be positive"
    @assert 0 < p_val ≤ 1 "Success probability p must be in (0,1]"
    
    # Clamp primal r to prevent extreme values during ForwardDiff exploration
    r_val_clamped = clamp(r_val, 1.0, MAX_PHASES)
    
    k = floor(Int, r_val_clamped)
    # w uses clamp(r,...) so w ≥ 0 when r was raised to 1; ∂w/∂r follows clamp at bounds
    w = clamp(r, 1, MAX_PHASES) - k
    
    # Ensure k >= 1 and reasonable upper bound
    k = max(1, k)
    if k >= MAX_PHASES
        k = MAX_PHASES
        w = zero(w)  # Use zero with same type as w
    end
    
    # Create neighboring phase-type distributions
    dph_k = dph_negative_binomial(k, p)
    dph_k1 = dph_negative_binomial(k+1, p)
    
    # Ensure w has the correct type to match the DPH elements
    T_eltype = eltype(dph_k.α)
    w_typed = convert(promote_type(T_eltype, eltype(w)), w)
    
    return SmoothDPH(dph_k, dph_k1, w_typed)
end

# =============================================================================
# SMOOTH DPH: DISTRIBUTION INTERFACE
# =============================================================================

"""
    pmf(dph::SmoothDPH, x::Int)

Compute the probability mass function using smooth interpolation:
pmf(x) = (1-w) * pmf_k(x) + w * pmf_{k+1}(x)

# Arguments
- `dph::SmoothDPH`: Smooth discrete phase-type distribution
- `x::Int`: Value to evaluate

# Returns
- Interpolated probability mass
"""
function pmf(dph::SmoothDPH{T, W}, x::Int) where {T, W}
    pmf_k = PhaseTypeDistributions.pmf(dph.dph_k, x)
    pmf_k1 = PhaseTypeDistributions.pmf(dph.dph_k1, x)
    return (1 - dph.w) * pmf_k + dph.w * pmf_k1
end

"""
    mean(dph::SmoothDPH)

Compute the mean of the smoothly interpolated distribution.

# Arguments
- `dph::SmoothDPH`: Smooth discrete phase-type distribution

# Returns
- Interpolated mean
"""
function mean(dph::SmoothDPH{T, W}) where {T, W}
    mean_k = PhaseTypeDistributions.mean(dph.dph_k)
    mean_k1 = PhaseTypeDistributions.mean(dph.dph_k1)
    return (1 - dph.w) * mean_k + dph.w * mean_k1
end

"""
    var(dph::SmoothDPH)

Compute the variance of the smoothly interpolated distribution.

# Arguments
- `dph::SmoothDPH`: Smooth discrete phase-type distribution

# Returns
- Interpolated variance
"""
function var(dph::SmoothDPH{T, W}) where {T, W}
    var_k = PhaseTypeDistributions.var(dph.dph_k)
    var_k1 = PhaseTypeDistributions.var(dph.dph_k1)
    mean_k = PhaseTypeDistributions.mean(dph.dph_k)
    mean_k1 = PhaseTypeDistributions.mean(dph.dph_k1)
    
    # For mixture: Var = E[X²] - E[X]²
    # E[X²] = (1-w) * E[X²_k] + w * E[X²_{k+1}]
    # E[X] = (1-w) * E[X_k] + w * E[X_{k+1}]
    mean_mix = (1 - dph.w) * mean_k + dph.w * mean_k1
    second_moment_k = var_k + mean_k^2
    second_moment_k1 = var_k1 + mean_k1^2
    second_moment_mix = (1 - dph.w) * second_moment_k + dph.w * second_moment_k1
    
    return second_moment_mix - mean_mix^2
end

"""
    logpdf(dph::SmoothDPH, x::Int)

Compute the log probability density function for Turing.jl compatibility.

# Arguments
- `dph::SmoothDPH`: Smooth discrete phase-type distribution
- `x::Int`: Value to evaluate

# Returns
- Log probability density
"""
function logpdf(dph::SmoothDPH{T, W}, x::Int) where {T, W}
    pmf_val = pmf(dph, x)
    return log(max(pmf_val, 1e-12))  # Numerical safety
end

# Make logpdf available in global scope for Turing.jl
Distributions.logpdf(dph::SmoothDPH, x::Int) = logpdf(dph, x)

"""
    rand(rng::AbstractRNG, dph::SmoothDPH)

Generate a random sample from the smooth DPH distribution.
Uses the interpolation weight to decide which underlying DPH to sample from.

# Arguments
- `rng::AbstractRNG`: Random number generator
- `dph::SmoothDPH`: Smooth discrete phase-type distribution

# Returns
- Random sample
"""
function rand(rng::AbstractRNG, dph::SmoothDPH{T, W}) where {T, W}
    # Use the interpolation weight to decide which DPH to sample from
    if rand(rng) < dph.w
        return rand(rng, dph.dph_k1)
    else
        return rand(rng, dph.dph_k)
    end
end

# Make SmoothDPH work with Turing.jl and Distributions.jl
Base.eltype(::Type{SmoothDPH{T, W}}) where {T, W} = Int
Base.eltype(::SmoothDPH{T, W}) where {T, W} = Int

# Required methods for Distribution interface
Distributions.support(dph::SmoothDPH{T, W}) where {T, W} = Distributions.support(dph.dph_k)
Distributions.minimum(dph::SmoothDPH{T, W}) where {T, W} = Distributions.minimum(dph.dph_k)
Distributions.maximum(dph::SmoothDPH{T, W}) where {T, W} = Distributions.maximum(dph.dph_k1)
Distributions.insupport(dph::SmoothDPH{T, W}, x) where {T, W} = Distributions.insupport(dph.dph_k, x) || Distributions.insupport(dph.dph_k1, x)

# =============================================================================
# EXPORTS
# =============================================================================

# Continuous phase-type distributions
export PhaseType, ph_exponential, ph_erlang, ph_erlang_by_mean

# Discrete phase-type distributions  
export DPH, dph_geometric, dph_negative_binomial, dph_erlang
export dph_from_rates, dph_exponential, dph_erlang_from_rates, dph_erlang_from_mean
export dph_negative_binomial_from_rates, dph_negative_binomial_from_mean
export dph_prepend_deterministic_delay, dph_negative_binomial_min_support

# Smooth interpolations
export dph_negative_binomial_smooth, SmoothDPH

# Common methods
export mean, var, pmf, cdf, pdf

end # module
