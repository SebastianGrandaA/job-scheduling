function solve(
    method::BaseModel,
    instance::PMSLPData,
    solver::SOLVER,
)::PMSLPSolution
    model = Model(solver)

    sites = 1:nb_sites(instance)
    jobs = 1:nb_jobs(instance)
    periods = instance.horizon
    λ = instance.parameters[:λ]
    β = instance.parameters[:tardiness_penalty]

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
        sum(is_allocated[s] for s in sites) <= nb_machines(instance)
    )

    # Job execution constraint
    @constraint(
        model,
        exclusive_assignment[j in jobs],
        sum(
            starts_at[p, j, s]
            for s in sites, p in available_window(instance, j, s)
        ) == 1
    )

    # Machine activation only if installed constraint
    # Note: stronger formulation than (4) in the paper
    @constraint(
        model,
        machine_activation[p in periods, j in jobs, s in sites],
        starts_at[p, j, s] <= is_allocated[s]
    )

    # Sequence: non-overlapping jobs and no preemption constraint
    @constraint(
        model,
        sequence[s in sites, t in periods],
        sum(
            starts_at[p, j, s]
            for j in jobs, p in start_window(instance, j, t)
        ) <= is_allocated[s]
    )

    # Task completion constraint
    @constraint(
        model,
        task_completion[j in jobs],
        finish_time[j] >= sum(
            p * starts_at[p, j, s]
            for p in periods, s in sites
        ) + processing_time(instance, j) - 1
    )

    # Tardiness calculation constraint
    @constraint(
        model,
        tardiness_calculation[j in jobs],
        tardiness[j] >= finish_time[j] + sum(
            starts_at[p, j, s] * travel_time(instance, j, s)
            for p in periods, s in sites
        ) - due_date(instance, j)
    )

    execution_time = @elapsed solve!(model)

    return PMSLPSolution(method, instance, model, execution_time)
end

function PMSLPSolution(
    method::BaseModel,
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

    for (job_idx, job) in enumerate(instance.jobs)
        delayed_time = value(model[:tardiness][job_idx])

        for (site_idx, site) in enumerate(instance.sites)
            for period in instance.horizon
                starts_at = value(model[:starts_at][period, job_idx, site_idx]) >= 0.5
                
                if starts_at
                    machine_usage = Window(job, period)
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