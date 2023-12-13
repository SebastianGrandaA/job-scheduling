using DelimitedFiles: readdlm
using Distances: euclidean

struct Location
    latitude::Float64
    longitude::Float64
end

struct Job
    id::String
    processing_time::Int64
    location::Location
    due_date::Int64
end

struct Site
    id::String
    fixed_cost::Int64
    location::Location
end

struct Estimator
    times::Matrix{Int64}
    costs::Matrix{Float64}
end

struct PMSLPData
    horizon::UnitRange{Int64}
    nb_machines::Int64
    jobs::Vector{Job}
    sites::Vector{Site}
    estimator::Estimator
    parameters::Dict{Symbol, Any}
end

get_idx(job::Job)::Int64 = parse(Int, replace(job.id, JOB_PREFIX => ""))
get_idx(site::Site)::Int64 = parse(Int, replace(site.id, SITE_PREFIX => ""))
get_idx(id::String)::Int64 = parse(Int, replace(id, JOB_PREFIX => "", SITE_PREFIX => ""))

function PMSLPData(filename::String; kwargs...)
    path = "inputs/instances/$(filename).dat"
    !isfile(path) && error("File does not exist: $path")

    raw_data = readdlm(path, comment_char='=')
    jobs = parse_number(raw_data[1, 1:end])
    sites = parse_number(raw_data[2, 1:end])
    nb_machines = parse_number(raw_data[3, 1:end])
    tardiness_penalty = parse_number(raw_data[4, 1:end])
    processing = parse_array(raw_data[5, 1:end])
    job_coordinates = parse_coordinates(raw_data[6, 1:end])
    site_coordinates = parse_coordinates(raw_data[7, 1:end])
    fixed_costs = parse_array(raw_data[8, 1:end])
    due_dates = parse_array(raw_data[9, 1:end])

    @assert jobs == length(processing) == length(job_coordinates) == length(due_dates)
    @assert sites == length(fixed_costs) == length(site_coordinates)

    jobs = [Job("$(JOB_PREFIX)$(i)", processing[i], Location(job_coordinates[i]...), due_dates[i]) for i in 1:jobs]
    sites = [Site("$(SITE_PREFIX)$(i)", fixed_costs[i], Location(site_coordinates[i]...)) for i in 1:sites]
    parameters = build_parameters(tardiness_penalty=tardiness_penalty, kwargs...)
    estimator = build_matrices(jobs, sites, parameters)
    horizon = build_horizon(jobs, estimator)

    return PMSLPData(horizon, nb_machines, jobs, sites, estimator, parameters)
end

"""
    build_horizon(instance::PMSLPData)

Builds the horizon of the problem assuming that all jobs are processed in sequence.
Reference: paper
"""
function build_horizon(jobs::Vector{Job}, estimator::Estimator)::UnitRange{Int64}
    max_travel_time = maximum(estimator.times)
    total_processing_time = sum(processing_time(job) for job in jobs)
    upper_bound = 2 * max_travel_time + total_processing_time

    return 1:ceil(Int64, upper_bound)
end

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
    λ = [
        0.4, # fixed_cost
        0.3, # travel_time
        0.3, # tardiness
    ]
    @assert sum(λ) == 1.0

    return Dict{Symbol, Any}(
        :λ => λ,
        :velocity => 10.0, # km/h
        :cost_per_km => 1.0, # euros/km
        kwargs...,
    )
end

# Distances
function get_distance(source::Location, destination::Location)::Float64
    return euclidean(
        (source.latitude, source.longitude),
        (destination.latitude, destination.longitude),
    )
end

function build_matrices(jobs::Vector{Job}, sites::Vector{Site}, parameters::Dict{Symbol, Any})::Estimator
    velocity = parameters[:velocity]
    cost_per_km = parameters[:cost_per_km]
    time_matrix = zeros(Int64, length(jobs), length(sites))
    cost_matrix = zeros(Float64, length(jobs), length(sites))

    for i in 1:length(jobs)
        for j in 1:length(sites)
            distance = get_distance(jobs[i].location, sites[j].location)
            time_matrix[i, j] = ceil(Int64, distance / velocity)
            cost_matrix[i, j] = distance * cost_per_km
        end
    end

    return Estimator(time_matrix, cost_matrix)
end

# Helpers

function add_fake_job!(instance::PMSLPData)::Nothing
    push!(instance.jobs, Job("$(JOB_PREFIX)FAKE", 0, Location(0, 0), 0))

    return nothing
end

nb_sites(instance::PMSLPData)::Int64 = length(instance.sites)
nb_jobs(instance::PMSLPData)::Int64 = length(instance.jobs)
nb_machines(instance::PMSLPData)::Int64 = instance.nb_machines

"""
Assupmtion: all tasks are transported at the beginning of the horizon (transportation resources are not limited).
"""
function earliest_start(instance::PMSLPData, job::Union{Job, Int64}, site::Union{Site, Int64})::Int64
    return travel_time(instance, job, site) + 1
end

function earliest_start(instance::PMSLPData, job_id::String, site_id::String)::Int64
    return earliest_start(instance, get_idx(job_id), get_idx(site_id))
end

function earliest_finish(instance::PMSLPData, job::Union{Job, Int64}, site::Union{Site, Int64})::Int64
    return earliest_start(instance, job, site) + processing_time(instance, job) - 1
end

function latest_start(job::Job, period::Int64)::Int64
    return max(1, period - processing_time(job) + 1)
end

function latest_start(instance::PMSLPData, job_id::String, period::Int64)::Int64
    return latest_start(instance.jobs[get_idx(job_id)], period)
end

"""
Start windows for a given job assuming it finishes at period p.
    Latest start for a job j assuming it finished at period p.
    Assuption: job j finishes at period p, then it should have started at one point of: start at period max(1, period - processing_time(job) + 1 and current period
"""
function start_window(job::Job, period::Int64)::UnitRange{Int64}
    return latest_start(job, period):period
    # finish_processing = min(period + processing_time(job) - 1, last(instance.horizon))
    # return period:finish_processing
end

function start_window(instance::PMSLPData, job_idx::Int64, period::Int64)::UnitRange{Int64}
    return start_window(instance.jobs[job_idx], period)
end

"""
Total time for a job j in site s
"""
function service_time(instance::PMSLPData, job_idx::Int64, site_idx::Int64)::Int64
    return processing_time(instance, job_idx) + travel_time(instance, job_idx, site_idx)
end

function available_window(instance::PMSLPData, job::Job, site::Site)::UnitRange{Int64}
    return earliest_start(instance, job, site):latest_start(job, last(instance.horizon))
end

function available_window(instance::PMSLPData, job_idx::Int64, site_idx::Int64)::UnitRange{Int64}
    return available_window(instance, instance.jobs[job_idx], instance.sites[site_idx])
end

processing_time(job::Job)::Int64 = job.processing_time

processing_time(instance::PMSLPData, job_idx::Int64)::Int64 = processing_time(instance.jobs[job_idx])

processing_time(instance::PMSLPData, job_id::String)::Int64 = processing_time(instance, get_idx(job_id))

function travel_time(instance::PMSLPData, job_idx::Int64, site_idx::Int64)::Int64
    return instance.estimator.times[job_idx, site_idx]
end

function travel_time(instance::PMSLPData, job::Job, site::Site)::Int64
    return travel_time(instance, get_idx(job), get_idx(site))
end

function travel_time(instance::PMSLPData, job_id::String, site_id::String)::Int64
    return travel_time(instance, get_idx(job_id), get_idx(site_id))
end

function travel_cost(instance::PMSLPData, job_idx::Int64, site_idx::Int64)::Float64
    return instance.estimator.costs[job_idx, site_idx]
end

function travel_cost(instance::PMSLPData, job::Job, site::Site)::Float64
    return travel_cost(instance, get_idx(job), get_idx(site))
end

fixed_cost(site::Site)::Int64 = site.fixed_cost

fixed_cost(instance::PMSLPData, site_idx::Int64)::Int64 = fixed_cost(instance.sites[site_idx])

due_date(job::Job)::Int64 = job.due_date

due_date(instance::PMSLPData, job_idx::Int64)::Int64 = due_date(instance.jobs[job_idx])