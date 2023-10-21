module Outputs

using ..Models: PMSLPData, PMSLPSolution
using DataFrames: DataFrame
using CSV: write
using Gadfly: PNG, Plot, Geom, Guide, Theme, plot, draw, cm
import Cairo

export export_solution!

function plot_gantt!(path::String, solution::DataFrame)::Nothing    
    p = plot(
        solution, x = :start, y = :site, xend = :finish, yend = :site, color = :job,
        Geom.segment,
        Guide.xlabel("Time"), Guide.ylabel("Site"), Guide.colorkey(title = "Job"),
        Theme(key_position = :top)
    )
    
    draw(PNG(path, 20cm, 10cm), p)
    
    return nothing
end

function export_solution!(path::String, instance::PMSLPData, solution::PMSLPSolution)::Nothing
    solution_data = DataFrame(
        job = String[],
        site = String[],
        start = Int[],
        finish = Int[],
        due_date = Int[]
    )
    due_dates = Dict{String, Int64}(
        job.id => job.due_date
        for job in instance.jobs
    )

    for assignment in solution.assignments
        row = (
            assignment.job_id,
            assignment.site_id,
            assignment.start_time,
            assignment.end_time,
            due_dates[assignment.job_id]
        )
        push!(solution_data, row)
    end

    plot_gantt!("$(path)_gantt.png", solution_data)
    write("$(path)_solution.csv", solution_data)

    return nothing
end

end