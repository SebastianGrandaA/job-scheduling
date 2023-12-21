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
    to_optimality = termination_status(model) == MOI.OPTIMAL

    !(to_optimality) && @warn "Model is not optimal"

    return nothing
end

# Output
struct Cluster
    centroid::Location
    jobs::Vector{Job}
end

struct Window
    start::Int64
    finish::Int64
end

function Window(job::Job, start_at::Int64)
    return Window(start_at, start_at + processing_time(job))
end

struct Assignment
    job_id::String
    site_id::String
    machine_usage::Window # with respect to the machine (without travel time)
    tardiness::Int64 # with respect to the job (including travel time)
end

struct Pattern
    matches::Vector{Tuple{Cluster, Site, Real}} # one for each machine
end

function get_first(pattern::Pattern, nb_matches::Int64)::Vector{Tuple{Cluster, Site}}
    return pattern.matches[1:nb_matches]
end

"""
Assignment cost includes travel time to all the jobs to the site and fixed cost of the site
"""
function tardiness(instance::PMSLPData, site::Site, schedule::Vector{Assignment})::Real
    assignment = first(filter(assignment -> assignment.site_id == site.id, schedule))

    return tardiness(assignment) * tardiness_penalty(instance)
end

opened_sites(pattern::Pattern)::Vector{Site} = [site for (_, site, _) in pattern.matches]
closed_sites(instance::PMSLPData, pattern::Pattern)::Vector{Site} = filter(site -> !in(site, opened_sites(pattern)), instance.sites)
clusters(pattern::Pattern)::Vector{Cluster} = [cluster for (cluster, _, _) in pattern.matches]
nb_machines(pattern::Pattern)::Int64 = length(pattern.matches)

struct History
    step::String
    cost::Float64
    pattern::Pattern
end

struct Metrics
    objective_value::Real
    execution_time::Float64
    status::MOI.TerminationStatusCode
end

mutable struct SearchSolution
    cost::Float64
    pattern::Pattern
    assignments::Vector{Assignment}
end

function SearchSolution(instance::PMSLPData, pattern::Pattern, assignments::Vector{Assignment})
    λ = instance.parameters[:λ]
    β = tardiness_penalty(instance)
    sites = opened_sites(pattern)
    location_cost = sum(fixed_cost(site) for site in sites)
    traveling_cost = sum(travel_cost(instance, assignment.job_id, assignment.site_id) for assignment in assignments)
    tardiness_cost = sum(assignment.tardiness for assignment in assignments)
    total_cost = λ[1] * location_cost + λ[2] * 2 * traveling_cost + λ[3] * β * tardiness_cost

    return SearchSolution(total_cost, pattern, assignments)
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
    historical::Vector{History}
end

function job_ids(solution::PMSLPSolution)::Vector{String}
    return unique([assignment.job_id for assignment in solution.assignments])
end

method(solution::PMSLPSolution)::String = String(Symbol(solution.method))
status(solution::PMSLPSolution)::String = string(solution.metrics.status)
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

function all_jobs_assigned(instance::PMSLPData, solution::PMSLPSolution)::Bool
    instance_jobs = job_ids(instance)
    solution_jobs = job_ids(solution)

    return isempty(setdiff(instance_jobs, solution_jobs)) && isempty(setdiff(solution_jobs, instance_jobs))
end

function validate(instance::PMSLPData, solution::PMSLPSolution)::Nothing
    has_overlapping(solution) && error("Solution has overlapping assignments")
    !(all_jobs_assigned(instance, solution)) && error("Solution does not include all jobs")
    objective_value(solution) <= 0 && error("Solution has negative objective value")

    @info "Solution is valid : no overlapping assignments and positive objective value"

    return nothing
end

function format!(solution::PMSLPSolution)::Nothing
    sort!(solution.open_sites, by=site -> site.id)
    format!(solution.assignments)

    return nothing
end

function format!(assignments::Vector{Assignment})::Nothing
    sort!(assignments, by=assignment -> start_time(assignment))

    return nothing
end

function get_schedule(solution::PMSLPSolution, site_id::String)::Vector{Assignment}
    return filter(assignment -> assignment.site_id == site_id, solution.assignments)
end

function get_schedule(solution::PMSLPSolution, site_nb::Int64)::Vector{Assignment}
    return get_schedule(solution, "$(SITE_PREFIX)$(site_nb)")
end

function get_assignments(assignments::Vector{Assignment}, site_id::String)::Vector{Assignment}
    return filter(assignment -> assignment.site_id == site_id, assignments)
end

function has_intersection(window_1::Window, window_2::Window)::Bool
    return (
        window_1.start < window_2.finish &&
        window_2.start < window_1.finish
    )
end

function has_intersection(first_assignment::Assignment, second_assignment::Assignment)::Bool
    return has_intersection(
        machine_usage(first_assignment),
        machine_usage(second_assignment)
    )
end

function has_overlapping(assignments::Vector{Assignment})::Bool
    N = length(assignments)

    for i in 1:N
        for j in i+1:N
            if has_intersection(assignments[i], assignments[j])
                overlap = (ID(assignments[i]), ID(assignments[j]))
                @warn "Overlapping assignments: $overlap"

                return true
            end
        end
    end

    return false
end

function has_overlapping(solution::PMSLPSolution)::Bool
    for site in solution.open_sites
        assignments = get_schedule(solution, site.id) # all jobs in a machine (site)
        has_overlapping(assignments) && return true
    end

    return false
end


# Plotting
function plot_gantt!(model_name::String, path::String, solution::DataFrame)::Nothing
    num_jobs = size(solution, 1)
    colors = distinguishable_colors(num_jobs, colorant"lightblue", lchoices=0.5:0.1:0.9)
    
    traces = map(eachrow(solution), 1:num_jobs) do row, idx
        start_time = row[:start_at]
        finish_time = row[:finish_at]
        duration = finish_time - start_time
        due_date = row[:due_date]
        job_name = row[:job_idx]
        site_name = string(row[:site_idx])
        earliest_start = row[:earliest_start]
        latest_start = row[:latest_start]

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
            text=["Job: $job_name<br>Start: $start_time<br>Finish: $finish_time<br>Duration: $duration<br>Due date: $due_date<br>Earliest start: $earliest_start<br>Latest start: $latest_start"],
        )
        return trace
    end
    
    nb_jobs = size(solution, 1)
    nb_sites = length(unique(solution[!, :site_idx]))
    title = "$(model_name) | Jobs: $(nb_jobs) | Sites: $(nb_sites)"
    layout = Layout(
        title=title,
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

cost(history::History)::Float64 = history.cost
step(history::History)::String = history.step

function plot_history!(path::String, historical::Vector{History})::Nothing
    values = cost.(historical)
    steps = step.(historical)
    title = "Objective function value vs Iteration"
    layout = Layout(
        title=title,
        xaxis=attr(title="Iteration", zeroline=false),
        yaxis=attr(title="Objective function value", zeroline=false),
        margin=attr(l=100, r=100, t=100, b=100),
        paper_bgcolor="white",
        plot_bgcolor="white"
    )
    trace = scatter(
        x=1:length(values),
        y=values,
        mode="lines+markers",
        marker=attr(
            color="blue",
            size=10,
            line=attr(
                color="black",
                width=0.5
            )
        ),
        hoverinfo="text",
        text=["Operator: $(steps[i])<br>Cost: $(values[i])" for i in 1:length(values)]
    )
    plot = Plot(trace, layout)
    savefig(plot, "$(path)_history.html")

    return nothing
end

# Exporting
function export_solution(model_name::String, path::String, instance::PMSLPData, solution::PMSLPSolution)::Nothing
    !(isempty(solution.historical)) && plot_history!(path, solution.historical)

    solution_data = DataFrame(
        job_idx = Int[],
        site_idx = Int[],
        start_at = Int[],
        finish_at = Int[],
        due_date = Int[],
        earliest_start = Int[],
        latest_start = Int[],
    )

    for assignment in solution.assignments
        row = (
            get_idx(assignment.job_id),
            get_idx(assignment.site_id),
            start_time(assignment),
            finish_time(assignment),
            due_date(instance, get_idx(assignment.job_id)),
            earliest_start(instance, assignment.job_id, assignment.site_id),
            latest_start(instance, assignment.job_id, last(instance.horizon)),
        )
        
        push!(solution_data, row)
    end

    sort!(solution_data, :job_idx, rev=false)
    @info solution_data

    plot_gantt!(model_name, "$(path)_gantt.html", solution_data)

    summary = select(solution_data, [:site_idx, :start_at])
    push!(summary, ("", objective_value(solution)), promote=true)
    write("$(path)_solution.csv", summary)

    return nothing
end

function add_model!(filename::String, instance::PMSLPData, solution::PMSLPSolution)::Nothing
    columns = ["instance_name", "model_name", "solution_value", "execution_time", "status"]
    history = get_file(filename, columns)
    execution = [name(instance), method(solution), objective_value(solution), execution_time(solution), status(solution)]
    push!(history, execution, promote=true)
    @info "Solution recorded in benchmark file"

    CSV.write(filename, history)

    return nothing
end

function get_job_ids(pattern::Pattern)::Vector{Vector{String}}
    return unique(
        vcat(
            [job.id for job in cluster.jobs]...
        )
        for (cluster, _, _) in pattern.matches
    )
end

function are_similar(vector_1::Vector{String}, vector_2::Vector{String}, similarity_threshold::Float64)::Bool
    nb_common_jobs = length(intersect(vector_1, vector_2))

    return nb_common_jobs / length(vector_1) >= similarity_threshold
end

function are_similar(pattern_1::Pattern, pattern_2::Pattern, similarity_threshold::Float64 = 0.25)::Bool
    @assert length(pattern_1.matches) == length(pattern_2.matches)
    nb_common_clusters = 0

    for pattern_1_cluster in get_job_ids(pattern_1)
        for pattern_2_cluster in get_job_ids(pattern_2)
            if are_similar(pattern_1_cluster, pattern_2_cluster, similarity_threshold)
                nb_common_clusters += 1
                break
            end
        end
    end

    are_different = nb_common_clusters / length(pattern_1.matches) < similarity_threshold

    return !(are_different)
end

"""
We say that is optimal if there are at maximum 1 job with tardiness > 0
"""
function is_optimal(assignments::Vector{Assignment})::Bool
    nb_assignments_with_tardiness = count(assignment -> assignment.tardiness > 0, assignments)

    return nb_assignments_with_tardiness <= 1
end
