using DataFrames
using CSV: write
using Colors
using PlotlyJS

# Model helpers
function solve!(model::Model)::Nothing
    optimize!(model)
    validate(model)
end

function validate(model::Model)::Nothing
    is_optimal = termination_status(model) == MOI.OPTIMAL

    !(is_optimal) && @warn "Model is not optimal"

    return nothing
end

# Output

struct Metrics
    objective_value::Real
    execution_time::Float64
end

struct Window
    start::Int64
    finish::Int64
end

function Window(job::Job, start_at::Int64)::Window
    return Window(start_at, start_at + processing_time(job) - 1)
end

struct Assignment
    job_id::String
    site_id::String
    machine_usage::Window # with respect to the machine (without travel time)
    tardiness::Int64 # with respect to the job (including travel time)
end

machine_usage(assignment::Assignment)::Window = assignment.machine_usage
tardiness(assignment::Assignment)::Int64 = assignment.tardiness

function ID(assignment::Assignment)::String
    return "$(assignment.job_id)@$(assignment.site_id)"
end

struct PMSLPSolution
    method::Method
    open_sites::Vector{Site}
    assignments::Vector{Assignment}
    metrics::Metrics
end

method(solution::PMSLPSolution)::String = String(Symbol(solution.method))
nb_sites(solution::PMSLPSolution)::Int64 = length(solution.open_sites)
nb_assignments(solution::PMSLPSolution)::Int64 = length(solution.assignments)

JuMP.objective_value(solution::PMSLPSolution)::Real = round(solution.metrics.objective_value, digits=4)

execution_time(solution::PMSLPSolution)::Float64 = round(solution.metrics.execution_time, digits=4)

start_time(assignment::Assignment)::Int64 = assignment.machine_usage.start

finish_time(assignment::Assignment)::Int64 = assignment.machine_usage.finish

function str(instance::PMSLPData, assignment::Assignment)::String
    elements = [
        "Assignment : $(ID(assignment))",
        "Machine Usage : [$(machine_usage(assignment))]",
        "Duration : $(processing_time(instance, assignment.job_id))",
        "Travel time : $(travel_time(instance, assignment.job_id, assignment.site_id))",
        "Tardiness : $(tardiness(assignment))",
        "Earliest start (release time) : $(earliest_start(instance, assignment.job_id, assignment.site_id))",
        "Latest start time (based on due date) : $(latest_start(instance, assignment.job_id, last(instance.horizon)))",
        "\n",
    ]

    return join(elements, " | ")
end

function Base.show(instance::PMSLPData, solution::PMSLPSolution)
    # TODO show as dataframe
    @info "Opened sites: $(nb_sites(solution)) / $(nb_sites(instance))"
    @info "Assignments: $(nb_assignments(solution)) / $(nb_jobs(instance))"
    @info "Objective value: $(objective_value(solution)) | Execution time: $(execution_time(solution))"

    for site in solution.open_sites
        schedule = get_schedule(solution, site.id)
        assignments = [str(instance, a) for a in schedule]

        @info "Site $(site.id) \n$(join(assignments))"
    end
end

function validate(solution::PMSLPSolution)::Nothing
    has_overlapping(solution) && @error("Solution has overlapping assignments")
    # TODO: add validation that all jobs are assigned
    objective_value(solution) <= 0 && @error("Solution has negative objective value")

    @info "Solution is valid : no overlapping assignments and positive objective value"

    return nothing
end

function format!(solution::PMSLPSolution)::Nothing
    sort!(solution.open_sites, by=site -> site.id)
    sort!(solution.assignments, by=assignment -> start_time(assignment))

    return nothing
end

function get_schedule(solution::PMSLPSolution, site_id::String)::Vector{Assignment}
    return filter(assignment -> assignment.site_id == site_id, solution.assignments)
end

function get_schedule(solution::PMSLPSolution, site_nb::Int64)::Vector{Assignment}
    return get_schedule(solution, "$(SITE_PREFIX)$(site_nb)")
end

function find_job_assignment(solution::PMSLPSolution, job_id::String)::Assignment
    job = filter(assignment -> assignment.job_id == job_id, solution.assignments)
    @assert length(job) == 1

    return first(job)
end

function has_intersection(window_1::Window, window_2::Window)::Bool
    return (
        window_1.start <= window_2.finish &&
        window_2.start <= window_1.finish
    )
end

function has_intersection(first_assignment::Assignment, second_assignment::Assignment)::Bool
    return has_intersection(
        machine_usage(first_assignment),
        machine_usage(second_assignment)
    )
end

function has_overlapping(solution::PMSLPSolution)::Bool
    overlaps = []

    for site in solution.open_sites
        assignments = get_schedule(solution, site.id) # all jobs in a machine (site)
        N = length(assignments)

        for i in 1:N
            for j in i+1:N
                if has_intersection(assignments[i], assignments[j])
                    overlap = (ID(assignments[i]), ID(assignments[j]))
                    @warn "Overlapping assignments: $overlap"
                    push!(overlaps, overlap)
                end
            end
        end
    end

    return !isempty(overlaps)
end


# Plotting
function plot_gantt!(path::String, solution::DataFrame)::Nothing
    num_jobs = size(solution, 1)
    colors = distinguishable_colors(num_jobs, colorant"lightblue", lchoices=0.5:0.1:0.9)
    
    traces = map(eachrow(solution), 1:num_jobs) do row, idx
        start_time = row[:start]
        finish_time = row[:finish]
        duration = finish_time - start_time
        job_name = row[:job]
        site_name = string(row[:site])
    
        trace = bar(
            x=[duration],
            y=[site_name],
            base=[start_time],
            orientation="h",
            name=job_name,
            marker=attr(
                color=colors[idx],
                line=attr(
                    color="black",
                    width=0.5
                )
            ),
            hoverinfo="text",
            text=["Job: $job_name<br>Start: $start_time<br>Finish: $finish_time<br>Duration: $duration"]
        )
        return trace
    end
    
    layout = Layout(
        title="Gantt Chart",
        xaxis=attr(title="Time", zeroline=false),
        yaxis=attr(title="Site", automargin=true, zeroline=false),
        barmode="stack",
        margin=attr(l=100, r=100, t=100, b=100),
        paper_bgcolor="white",
        plot_bgcolor="white"
    )
    
    gantt_plot = plot(traces, layout)
    savefig(gantt_plot, path)

    return nothing
end

# Exporting
function export_solution(path::String, instance::PMSLPData, solution::PMSLPSolution)::Nothing
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
            start_time(assignment),
            finish_time(assignment),
            due_dates[assignment.job_id]
        )
        push!(solution_data, row)
    end

    plot_gantt!("$(path)_gantt.html", solution_data)
    write("$(path)_solution.csv", solution_data)

    return nothing
end

function add_model!(filename::String, instance::PMSLPData, solution::PMSLPSolution)::Nothing
    columns = ["instance_name", "model_name", "solution_value", "execution_time"]
    history = get_file(filename, columns)
    execution = [name(instance), method(solution), objective_value(solution), execution_time(solution)]
    push!(history, execution, promote=true)
    @info "Solution recorded in benchmark file"

    CSV.write(filename, history)

    return nothing
end
