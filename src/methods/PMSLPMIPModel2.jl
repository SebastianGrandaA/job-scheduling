function solve(
    method::PMSLPMIPModel2,
    instance::PMSLPData,
    solver::SOLVER,
)::PMSLPSolution
    model = Model(solver)

    T = last(instance.horizon)
    sites = 1:nb_sites(instance)
    jobs = 1:nb_jobs(instance)
    dummy_node = nb_jobs(instance) + 1
    nodes = 1:dummy_node
    λ = instance.parameters[:λ]
    β = instance.parameters[:tardiness_penalty]
    Ω = [(i, j) for i in jobs, j in jobs if i != j]

    @variable(model, is_allocated[sites], Bin) # y_k : machine installation in each site k
    @variable(model, is_assigned[jobs, sites], Bin) # x_jk : job j is assigned to site k
    @variable(model, is_sequence[nodes, jobs, sites], Bin) # u_ijk : job i is sequenced (inmediately) before job j in site k
    @variable(model, starts_at[nodes] >= 0, Int) # S_j : job j start time
    @variable(model, finish_time[jobs] >= 0, Int) # C_j : job j finish time
    @variable(model, tardiness[jobs] >= 0, Int) # T_j : job j tardiness

    @objective(
        model,
        Min,
        λ[1] * sum(is_allocated[s] * fixed_cost(instance, s) for s in sites)
        + λ[2] * 2 * sum(
            is_assigned[j, s] * travel_cost(instance, j, s)
            for j in jobs, s in sites
        )
        + λ[3] * β * sum(tardiness[j] for j in jobs)
    )

    # Maximum machine installation constraint
    @constraint(
        model,
        maximum_installations,
        sum(is_allocated[s] for s in sites) <= nb_machines(instance)
    )

    # Machine activation only if installed constraint
    @constraint(
        model,
        machine_activation[j in jobs, s in sites],
        is_assigned[j, s] <= is_allocated[s]
    )

    @constraint(
        model,
        exclusive_assignment[j in jobs],
        sum(is_assigned[j, s] for s in sites) == 1
    )

    # Job sequence constraint
    @constraint(
        model,
        sequence[(i, j) in Ω, s in sites],
        is_assigned[i, s] >= is_sequence[i, j, s] + is_sequence[j, i, s]
    )

    @constraint(
        model,
        [(i, j) in Ω, s in sites],
        is_sequence[i, j, s] + is_sequence[j, i, s] >= is_assigned[i, s] + is_assigned[j, s] - 1
    )

    @constraint(
        model,
        [(i, j) in Ω, s in sites],
        starts_at[i] - starts_at[j] >= latest_start(instance, j, T) * (is_sequence[i, j, s] - 1) + processing_time(instance, j)
    )

    @constraint(
        model,
        [i in jobs, s in sites],
        starts_at[i] >= earliest_start(instance, i, s) + latest_start(instance, i) * (is_assigned[i, s] - 1)
    )

    # Consecutive sequence (no dead time) constraint
    @constraint(
        model,
        finish_time_definition[j in jobs],
        finish_time[j] == starts_at[j] + processing_time(instance, j) + sum(
            is_assigned[j, s] * earliest_start(instance, j, s)
            for s in sites
        )
    )

    @constraint(
        model,
        tardiness_definition[j in jobs],
        tardiness[j] >= finish_time[j] - due_date(instance, j)
    )

    elapsed_time = @elapsed solve!(model)

    return PMSLPSolution(method, instance, model, elapsed_time)
end

function PMSLPSolution(
    method::PMSLPMIPModel2,
    instance::PMSLPData,
    model::Model,
    execution_time::Float64,
)
    open_sites = Site[
        site
        for (site_idx, site) in enumerate(instance.sites)
        if value(model[:is_allocated][site_idx]) >= 0.5
    ]
    assignments = Assignment[]

    for (j, job) in enumerate(instance.jobs)
        delayed_time = value(model[:tardiness][j])

        for (s, site) in enumerate(instance.sites)
            for n in 1:nb_jobs(instance) + 1
                is_sequence = value(model[:is_sequence][n, j, s]) >= 0.5

                if is_sequence
                    raw_starts_at = value(model[:starts_at][n])
                    starts_at = round(Int64, raw_starts_at)
                    @assert starts_at == raw_starts_at

                    machine_usage = Window(job, starts_at)
                    assignment = Assignment(job.id, site.id, machine_usage, delayed_time)
                    push!(assignments, assignment)

                    break
                end
            end
        end
    end

    metrics = Metrics(objective_value(model), execution_time)
    solution = PMSLPSolution(method, open_sites, assignments, metrics)
    format!(solution)
    validate(solution)

    return solution
end
