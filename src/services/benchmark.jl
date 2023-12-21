using StatsBase

"""
This can be ran in parallel for each instance-model, however it is not recommended to avoid resource competition.
"""
function benchmark!(instances::Vector{String}, models::Vector{String}; kwargs...)::Nothing
    for instance in instances
        for model in models
            timeout = get(kwargs, Symbol("$(model)_timeout"), 1 * 60 * 60)
            @info "Benchmarking $(model) on $(instance)"
            args = Dict(
                "filename" => instance, 
                "model" => model,
                "limit" => timeout,
                "benchmark" => true,
                "verbose" => false,
            )
            _ = execute(args)
        end
    end

    return nothing
end

function benchmark!(models::Vector{String}, sample_size::Int64=1; kwargs...)::Nothing
    path = joinpath(pwd(), "inputs", "instances")
    instances = [
        replace(file, ".dat" => "")
        for file in readdir(path) if isfile(joinpath(path, file))
    ]
    if sample_size > 1
        instances = sample(instances, sample_size, replace=false)
        @info "Sampling $(length(instances)) instances"
    end

    benchmark!(instances, models; kwargs...)
end