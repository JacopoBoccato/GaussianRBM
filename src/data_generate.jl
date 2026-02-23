"""
    sample_gaussian_pair(directions::Vector{Vector{Float64}},
                         intensities::Vector{Float64},
                         n_samples::Int)

Generate two families of random samples:
  - family 1: N(0, I)
  - family 2: N(0, I + sum_k intensities[k] * directions[k] * directions[k]')

Inputs:
  - directions  : K vectors of the same length D (need not be normalized)
  - intensities : K positive scalars, one per direction
  - n_samples   : number of samples to draw
  -mean_shifts : optional D-dimensional vector to shift the mean of family 2 (default: zero vector)

Returns:
  - samples_iso   : D × n_samples matrix from N(0, I)
  - samples_lowrank : D × n_samples matrix from N(0, I + V * diag(intensities) * V')
"""
function sample_gaussian_pair(
    directions::Vector{Vector{Float64}},
    intensities::Vector{Float64},
    n_samples::Int;
    mean_shift::Union{Vector{Float64}, Nothing}=nothing
)
    @assert length(directions) == length(intensities) "Need one intensity per direction"
    @assert all(length(d) == length(directions[1]) for d in directions) "All directions must have the same length"
    @assert all(intensities .> 0) "Intensities must be positive"
    @assert iseven(n_samples) "n_samples must be even for balanced classes"

    D = length(directions[1])
    K = length(directions)
    half = n_samples ÷ 2

    if mean_shift !== nothing
        @assert length(mean_shift) == D "mean_shift must have the same length as directions"
    end

    dirs_normalized = [d / norm(d) for d in directions]

    # Family 1: N(0, I)  — half samples
    samples_iso = randn(D, half)

    # Family 2: N(mean_shift, I + Σ_k λ_k v_k v_k')  — half samples
    samples_lowrank = randn(D, half)
    for k in 1:K
        w_k = randn(1, half)
        samples_lowrank .+= sqrt(intensities[k]) .* dirs_normalized[k] .* w_k
    end
    if mean_shift !== nothing
        samples_lowrank .+= mean_shift
    end

    # Concatenate and create balanced labels
    samples = hcat(samples_iso, samples_lowrank)   # D × n_samples
    labels  = Int32[zeros(Int32, half); ones(Int32, half)]  # 0 = iso, 1 = low-rank

    # Shuffle so classes are interleaved
    perm    = randperm(n_samples)
    samples = samples[:, perm]
    labels  = labels[perm]

    return samples, labels
end