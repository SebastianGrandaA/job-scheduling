function solve(
    method::PMSLPMIPModel2,
    instance::PMSLPData,
    solver::SOLVER,
)::PMSLPSolution
    model = Model(solver)

    sites = 1:nb_sites(instance)
    jobs = 1:nb_jobs(instance)
    dummy_node = nb_jobs(instance) + 1
    nodes = 1:dummy_node
    λ = instance.parameters[:λ]
    β = instance.parameters[:tardiness_penalty]

    @variable(model, is_in_sequence[nodes, jobs, sites], Bin) # job i is sequenced (inmediately) before job j in site k
    @variable(model, is_allocated[sites], Bin) # machine installation in each site k
    @variable(model, is_assigned[jobs, sites], Bin) # job j is assigned to site k
    
    @variable(model, starts_at[nodes] >= 0, Int) # job j start time
    @variable(model, finish_time[jobs] >= 0, Int) # job j finish time
    @variable(model, tardiness[jobs] >= 0, Int) # job j tardiness

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

    # Job sequence constraint
    # Continue from here...
    



    # Machine activation only if installed constraint
    @constraint(
        model,
        machine_activation[job in jobs, site in sites],
        is_assigned[job, site] <= is_allocated[site]
    )

    # Job sequence constraint
    @constraint(
        model,
        sequence[i in jobs, j in jobs, site in sites],
        is_sequence[i, j, site] <= 1/2 * (is_assigned[i, site] + is_assigned[j, site])
    )

    # Dummy job sequence constraint
    @constraint(
        model,
        dummy_job_sequence[job in jobs, site in sites],
        is_sequence[dummy_node, job, site] <= is_assigned[job, site]
    )

    # All nodes are sequenced constraint
    @constraint(
        model,
        sequence_nodes[job in jobs],
        sum(is_sequence[node, job, site] for node in nodes, site in sites) == 1
    )

    # Maximum sequence constraint
    @constraint(
        model,
        maximum_sequence[job in jobs],
        sum(is_sequence[job, i, site] for i in jobs, site in sites) <= 1
    )

    # Consecutive sequence (no dead time) constraint
    @constraint(
        model,
        consecutive_sequence[job in jobs],
        starts_at[job] == sum(
            is_sequence[i, job, site] * (starts_at[i] + processing_time(instance.jobs[i]))
            for i in jobs, site in sites
        )
    ) # Assumption : Start just after the previous job (no dead time) -- TODO CHECK !

    # Start time definition constraint
    @constraint(
        model,
        start_time_definition[job in jobs],
        starts_at[job] >= sum(
            is_assigned[job, site] * travel_time(instance, job, site)
            for site in sites
        )
    )

    # Dummy job start time constraint
    @constraint(
        model,
        dummy_start_time,
        starts_at[dummy_node] == 0
    )

    # Finish time definition constraint
    @constraint(
        model,
        finish_time_definition[job in jobs],
        finish_time[job] == starts_at[job] + processing_time(instance.jobs[job]) + sum(
            is_assigned[job, site] * travel_time(instance, job, site)
            for site in sites
        )
    )

    # Tardiness definition constraint
    @constraint(
        model,
        tardiness_definition[job in jobs],
        tardiness[job] >= finish_time[job] - due_date(instance.jobs[job])
    )

    solve!(model)


end