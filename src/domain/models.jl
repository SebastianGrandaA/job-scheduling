module Models

export Location, Job, Site, PMSLPData, Assignment, PMSLPSolution

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

struct PMSLPData
    machines::Int64
    jobs::Vector{Job}
    sites::Vector{Site}
    parameters::Dict{Symbol, Any}
end

struct Assignment
    job_id::String
    site_id::String
    start_time::Int64
    end_time::Int64
    tardiness::Int64
end

struct PMSLPSolution
    value::Float64
    open_sites::Vector{Site}
    assignments::Vector{Assignment}
end

end