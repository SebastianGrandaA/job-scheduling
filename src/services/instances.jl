module Instances

using ..Models: Location, Job, Site, PMSLPData
using DelimitedFiles: readdlm

export load

function parse_array(raw::Vector{Any})::Vector{Int64}
    filtered = filter(x -> occursin(r"\d", x), raw)

    return map(x -> parse(Int, replace(x, r"[^\d]" => "")), filtered)
end

function parse_number(raw::Vector{Any})::Real
    return first(filter(x -> isa(x, Real), raw))
end

function parse_coordinates(raw::Vector{Any})::Vector{Vector{Int64}}
    joined = join(raw)
    matches = eachmatch(r"\d+", joined)
    numbers = [parse(Int, m.match) for m in matches]

    return [[numbers[i], numbers[i+1]] for i in 1:2:length(numbers)-1]
end

function build_parameters(; kwargs...)::Dict{Symbol, Any}
    weights = Dict{Symbol, Real}(
        :fixed_cost => 0.4,
        :travel_time => 0.3,
        :tardiness => 0.3,
    )
    @assert sum(values(weights)) == 1.0

    return Dict{Symbol, Any}(
        kwargs...,
        :velocity => 10.0, # km/h
        :weights => weights,
        :cost_per_km => 1.0, # euros/km
    )
end

function load(filename::String; kwargs...)::PMSLPData
    raw_data = readdlm(filename, comment_char='=')
    jobs = parse_number(raw_data[1, 1:end])
    sites = parse_number(raw_data[2, 1:end])
    machines = parse_number(raw_data[3, 1:end])
    tardiness_penalty = parse_number(raw_data[4, 1:end])
    processing = parse_array(raw_data[5, 1:end])
    job_coordinates = parse_coordinates(raw_data[6, 1:end])
    site_coordinates = parse_coordinates(raw_data[7, 1:end])
    fixed_costs = parse_array(raw_data[8, 1:end])
    due_dates = parse_array(raw_data[9, 1:end])

    @assert jobs == length(processing) == length(job_coordinates) == length(due_dates)
    @assert sites == length(fixed_costs) == length(site_coordinates)

    return PMSLPData(
        machines,
        [Job("J$i", processing[i], Location(job_coordinates[i]...), due_dates[i]) for i in 1:jobs],
        [Site("S$i", fixed_costs[i], Location(site_coordinates[i]...)) for i in 1:sites],
        build_parameters(
            tardiness_penalty=tardiness_penalty,
            timeout=Int64(kwargs[:timeout]),
        ),
    )
end

end