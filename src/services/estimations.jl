module Estimations

using ..Models: Job, Location, Site
using Distances: euclidean

export build_matrices

function get_distance(source::Location, destination::Location)::Float64
    return euclidean(
        (source.latitude, source.longitude),
        (destination.latitude, destination.longitude),
    )
end

function build_matrices(jobs::Vector{Job}, sites::Vector{Site}, parameters::Dict{Symbol, Any})::Dict{Symbol, Matrix}
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

    return Dict{Symbol, Matrix}(
        :time => time_matrix,
        :cost => cost_matrix,
    )
end

end