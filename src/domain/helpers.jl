module Helpers

using ..Models: PMSLPData

export build_horizon

"""
    build_horizon(instance::PMSLPData)

Builds the horizon of the problem assuming that all jobs are processed in sequence.
TODO revisar esto. Capaz de muy grande.
"""
function build_horizon(instance::PMSLPData, time_matrix::Matrix)::Int64
    total_processing_time = sum(job.processing_time for job in instance.jobs)
    # max_due_date = maximum(job.due_date for job in instance.jobs)
    max_travel_time = maximum(time_matrix)

    return ceil(Int64, total_processing_time + max_travel_time)
end

end