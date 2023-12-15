function solve(
    method::PMSLPMIPModel1,
    instance::PMSLPData,
    solver::SOLVER,
)::PMSLPSolution
    model = Model(solver)

    sites = 1:nb_sites(instance)
    jobs = 1:nb_jobs(instance)
    periods = instance.horizon
    λ = instance.parameters[:λ]
    β = instance.parameters[:tardiness_penalty]
    Ω = [(i, j) for i in jobs, j in jobs if i != j]

    @variable(model, is_allocated[sites], Bin) # `y` : machine installation in each site k
    @variable(model, starts_at[periods, jobs, sites], Bin) # `x` : job j starts at time t in site k
    @variable(model, finish_time[jobs] >= 0, Int) # `C` : job j finish time
    @variable(model, tardiness[jobs] >= 0, Int) # `T`: job j tardiness

    @objective(
        model,
        Min,
        λ[1] * sum(is_allocated[s] * fixed_cost(instance, s) for s in sites)
        + λ[2] * 2 * sum(
            starts_at[p, j, s] * travel_cost(instance, j, s)
            for p in periods, j in jobs, s in sites
        )
        + λ[3] * β * sum(tardiness[j] for j in jobs)
    )

    # Maximum machine installation constraint
    @constraint(
        model,
        maximum_installations,
        sum(is_allocated[s] for s in sites) == nb_machines(instance)
    )

    # Job execution constraint
    @constraint(
        model,
        exclusive_assignment[j in jobs],
        sum(
            starts_at[p, j, s]
            for s in sites, p in periods # available_window(instance, j, s)
        ) == 1
    )

    # Machine activation only if installed constraint
    @constraint(
        model,
        machine_activation[p in periods, j in jobs, s in sites],
        starts_at[p, j, s] <= is_allocated[s]
    )

    # Start after arrival constraint (replace by job execution constraint)
    @constraint(
        model,
        start_after_arrival[j in jobs, s in sites],
        sum(
            starts_at[p, j, s]
            for p in 1:(travel_time(instance, j, s) - 1) # TODO Validar + 1 o -1 (como en doc) y refactor si esta bien
        ) == 0
    )

    # Sequence: non-overlapping jobs and no preemption constraint
    @constraint(
        model,
        sequence[(i, j) in Ω, s in sites, p in periods],
        starts_at[p, i, s] + sum(
            starts_at[t, j, s]
            for t in p:(min(p + processing_time(instance, i) - 1, last(instance.horizon)))
        ) <= is_allocated[s]
    )

    # Task completion constraint
    @constraint(
        model,
        task_completion[j in jobs],
        finish_time[j] >= sum(
            starts_at[p, j, s] * (p + processing_time(instance, j) + travel_time(instance, j, s))
            for p in periods, s in sites
        )
    )

    # Tardiness calculation constraint
    @constraint(
        model,
        tardiness_calculation[j in jobs],
        tardiness[j] >= finish_time[j] - due_date(instance, j)
    )

    execution_time = @elapsed solve!(model)

    return PMSLPSolution(method, instance, model, execution_time)

end

function PMSLPSolution(
    method::PMSLPMIPModel1,
    instance::PMSLPData,
    model::Model,
    execution_time::Float64,
)
    open_sites = Site[
        site
        for (s, site) in enumerate(instance.sites)
        if value(model[:is_allocated][s]) >= 0.5
    ]
    assignments = Assignment[]

    for (j, job) in enumerate(instance.jobs)
        delayed_time = value(model[:tardiness][j])

        for (s, site) in enumerate(instance.sites)
            for period in instance.horizon
                starts_at = value(model[:starts_at][period, j, s]) >= 0.5
                
                if starts_at
                    machine_usage = Window(job, period)
                    assignment = Assignment(job.id, site.id, machine_usage, delayed_time)
                    push!(assignments, assignment)

                    break
                end
            end
        end
    end

    metrics = Metrics(objective_value(model), execution_time, termination_status(model))
    solution = PMSLPSolution(method, open_sites, assignments, metrics)
    format!(solution)
    validate(solution)

    return solution
end

