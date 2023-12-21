"""
    clusterize(jobs, nb_machines)

The first step in the construction heuristic is to clusterize the jobs into `nb_machines` partitions based on their geographical position.
This is achieved by using the k-means algorithm, because the number of clusters is known in advance.
... Add formulas ...
"""
function clusterize(jobs::Vector{Job}, nb_machines::Int64)::Vector{Cluster}
    points = reduce(hcat, [[latitude(job), longitude(job)] for job in jobs])
    clusters = kmeans(points, nb_machines)
    centroids = clusters.centers
    assignments = clusters.assignments

    return [
        Cluster(
            Location(centroids[1, i], centroids[2, i]),
            [jobs[j] for j in findall(x -> x == i, assignments)]
        )
        for i in 1:nb_machines
    ]
end

"""
    assign(instance, clusters, sites)

Given a set of job clusters, the second step of the construction heuristic aims to decide the machine locations (sites) to serve the jobs.
The machine location problem is modeled as a linear assignment problem (LAP) that minimizes the total cost of serving the jobs in the cluster from a given site.
This cost considers the installation cost of openning the site and the traveling cost from the site to the cluster centroid.
This problem is solved using the Hungarian algorithm for simplicity, but can be solved more efficiently using the Jonker-Volgenant algorithm.

In short, the machine location is simplified as a linear assignment problem between the clusters and the sites.
"""
function assign(instance::PMSLPData, clusters::Vector{Cluster}, sites::Vector{Site})::Pattern
    cost_matrix = zeros(length(clusters), length(sites))
    unit_cost = cost_per_km(instance)

    for (c, cluster) in enumerate(clusters)
        for (s, site) in enumerate(sites)
            traveling_cost = 2 * unit_cost * get_distance(cluster.centroid, site.location)
            cost_matrix[c, s] = fixed_cost(site) + traveling_cost
        end
    end

    assignments, _ = hungarian(cost_matrix)

    return Pattern([
        (cluster, sites[assignments[c]], cost_matrix[c, assignments[c]])
        for (c, cluster) in enumerate(clusters)
    ])
end

"""
    schedule(solver, instance, pattern)

Given that the location-allocation decisions have been made (i.e. job clusters and the machine locations are fixed), the third step of the construction heuristic aims to schedule the jobs in each cluster.
This is achieved by solving a 1|r_j|sumT_j scheduling problem for each cluster independently (in parallel).
To speed up the process in each cluster, we first schedule the jobs using the earliest due date heuristic (EDD) and evaluate optimality based on the quantity of jobs with tardiness greater than zero.
If there are more than one job with positive tardiness, then we solve the scheduling problem optimally using a mixed-integer programming (MIP) model.
"""
function schedule(solver::SOLVER, instance::PMSLPData, pattern::Pattern)::Union{Vector{Assignment}, Nothing}
    nworkers() < nb_machines(pattern) && addprocs(nb_machines(pattern) - nworkers())
    tasks = []
    schedules = Assignment[]

    for (cluster, site, _) in pattern.matches
        task = @async begin
            try
                scheduling = heuristic_schedule(instance, site, cluster)
                is_optimal(scheduling) && return scheduling

                return optimal_schedule(solver, instance, site, cluster)
            catch err
                error("Error while scheduling $(err)")
                return nothing
            end
        end

        push!(tasks, task)
    end

    for task in tasks
        schedule = fetch(task)
        isnothing(schedule) && return nothing

        append!(schedules, schedule)
    end

    return schedules
end

"""
    optimal_schedule(solver, instance, site, cluster)

Given a cluster of jobs and a site, solve the 1|r_j|sumT_j scheduling problem to optimality using a mixed-integer programming (MIP) model.
The solver by default is Gurobi.
"""
function optimal_schedule(solver::SOLVER, instance::PMSLPData, site::Site, cluster::Cluster)::Vector{Assignment}
    jobs = 1:length(cluster.jobs)
    Ω = [(i, j) for i in jobs, j in jobs if i != j]
    T = last(instance.horizon)

    model = Model(solver)

    @variable(model, is_sequence[jobs, jobs], Bin)
    @variable(model, starts_at[jobs] >= 0, Int)
    @variable(model, finish_time[jobs] >= 0, Int)
    @variable(model, tardiness[jobs] >= 0, Int)

    @objective(
        model,
        Min,
        sum(tardiness[j] for j in jobs)
    )

    # Job sequence constraint
    @constraint(
        model,
        [(i, j) in Ω],
        is_sequence[i, j] + is_sequence[j, i] <= 1
    )
    @constraint(
        model,
        [(i, j) in Ω],
        is_sequence[i, j] + is_sequence[j, i] >= 1
    )

    @constraint(
        model,
        [(i, j) in Ω],
        starts_at[i] - starts_at[j] >= T * (is_sequence[i, j] - 1) + processing_time(cluster.jobs[j])
    )

    @constraint(
        model,
        [i in jobs],
        starts_at[i] >= earliest_start(instance, cluster.jobs[i], site) - 2
    )

    @constraint(
        model,
        [j in jobs],
        finish_time[j] == starts_at[j] + processing_time(cluster.jobs[j]) + earliest_start(instance, cluster.jobs[j], site)
    )

    @constraint(
        model,
        [j in jobs],
        tardiness[j] >= finish_time[j] - due_date(cluster.jobs[j])
    )

    solve!(model)

    return [
        Assignment(
            job.id,
            site.id,
            Window(value(starts_at[j]), value(starts_at[j]) + processing_time(job)),
            # Window(job, value(starts_at[j])),
            value(tardiness[j]),
        )
        for (j, job) in enumerate(cluster.jobs)
    ]
end

"""
    heuristic_schedule(instance, site, cluster)

Given a cluster of jobs and a site, solve the 1|r_j|sumT_j scheduling problem using the earliest due date heuristic (EDD).
"""
function heuristic_schedule(instance::PMSLPData, site::Site, cluster::Cluster)::Vector{Assignment}
    sort!(cluster.jobs, by=job -> earliest_start(instance, job, site), rev=false)
    sort!(cluster.jobs, by=job -> due_date(job), rev=false)
    assignments = Assignment[]

    for (j, job) in enumerate(cluster.jobs)
        is_first = j == 1
        starts_at = is_first ? earliest_start(instance, job, site) : max(finish_time(assignments[j - 1]), earliest_start(instance, job, site))
        finish_at = starts_at + processing_time(job)
        tardiness = max(0, finish_at - due_date(job))
        assignment = Assignment(job.id, site.id, Window(starts_at, finish_at), tardiness)
        push!(assignments, assignment)
    end

    return assignments
end

"""
    construction(solver, instance)

The construction heuristic is composed of three steps:
1. Cluster jobs in |M| partitions based on their geographical position.
2. Assign clusters to sites based on the total cost of serving the jobs from the site (without considering the tardiness).
3. Schedule the jobs in each cluster independently.
"""
function construction(solver::SOLVER, instance::PMSLPData)::SearchSolution
    clusters = clusterize(instance.jobs, nb_machines(instance))
    pattern = assign(instance, clusters, instance.sites)
    assignments = schedule(solver, instance, pattern)

    return SearchSolution(instance, pattern, assignments)
end

"""
    exchange!(receptor, donor)

Given two clusters, exchange a random number of jobs from the donor to the receptor.
This quantity can be zero, in which case no jobs are exchanged.
"""
function exchange!(receptor::Cluster, donor::Cluster)::Int64
    nb_swaps = rand(0:length(donor.jobs))
    nb_swaps == 0 && return nb_swaps
    jobs_to_swap = sample(donor.jobs, nb_swaps, replace=false)

    for job in jobs_to_swap
        push!(receptor.jobs, job)
        filter!(j -> j != job, donor.jobs)
    end

    return nb_swaps
end

"""
    swap!(pattern)

This operator allows to exchange jobs between clusters.
Given two clusters A and B, A with the highest and B with the lowest total tardiness, exchange a random number of jobs Q between them: Q_1 from A to B and Q_2 from B to A.
Note that Q_1 or Q_2 can be zero but not both at the same time, meaning that there exist the possibility that only one cluster receives jobs (insertion) or only one cluster gives jobs (deletion).
This function assumes that the matches have been sorted by decreasing total tardiness.
"""
function swap!(::PMSLPData, pattern::Pattern)::String
    first_cluster, second_cluster = pattern.matches[1][1], pattern.matches[end][1]
    nb_swaps_1 = exchange!(first_cluster, second_cluster)
    nb_swaps_2 = exchange!(second_cluster, first_cluster)

    return "$(SWAP_PREFIX)$(nb_swaps_1)-$(nb_swaps_2)"
end

"""
    reassign!(instance, pattern)

This operator allows to reassign a cluster to a different site.
Given a set of opened sites A and a set of closed sites B in a solution, take the cluster in A with the highest total tardiness and reassign its jobs to the site in B with the lowest installation cost.
This function assumes that the matches have been sorted by decreasing total tardiness.
"""
function reassign!(instance::PMSLPData, pattern::Pattern)::String
    sites_opened = opened_sites(pattern)
    sites_closed = closed_sites(instance, pattern)
    sort!(sites_closed, by=site -> fixed_cost(site), rev=false)

    nb_reassignments = 1 # min(5, length(sites_closed), length(sites_opened))
    sites_to_open = sites_closed[1:nb_reassignments]
    sites_to_close = sites_opened[1:nb_reassignments]

    for i in 1:nb_reassignments
        site_to_open = sites_to_open[i]
        site_to_close = sites_to_close[i]
        cluster_to_open, _ = first(filter(match -> match[2] == site_to_close, pattern.matches))
        filter!(match -> match[2] != site_to_close, pattern.matches)

        assignment_cost = fixed_cost(site_to_open) + (
            get_distance(cluster_to_open.centroid, site_to_open.location) * cost_per_km(instance)
        )
        push!(pattern.matches, (cluster_to_open, site_to_open, assignment_cost))
    end

    return "$(REASSIGN_PREFIX)$(nb_reassignments)"
end

"""
    is_tabu(historical, new_pattern, to_intensify)

This strategy allows us to avoid evaluating similar patterns in the search space by keeping the history of the last N patterns evaluated.
The similarity between two patterns is measured using the quantity of clusters with similar jobs.
We evaluate the relative number of jobs that are common between each pair of clusters, and if this number is greater than a relative threshold, then we consider the patterns similar.

To diversify and intensify during the search, we adapt the similarity threshold and the tabu list length (adaptive tabu list)
When diversifying, we seek to avoid similar patterns, and therefore the similarity threshold is lower and the tabu list is larger.
In contrast, when intensifying, we seek evaluate similar patterns, thus the similarity threshold is higher and the tabu list is smaller.
"""
function is_tabu(historical::Vector{History}, new_pattern::Pattern, to_intensify::Bool)::Bool
    similarity_threshold = to_intensify ? 0.8 : 0.4
    tabu_lenght_porcentage = to_intensify ? 0.25 : 0.75
    N = length(historical)
    tabu_length = Int64(floor(N * tabu_lenght_porcentage))
    tabu_list = historical[N - tabu_length + 1:end]

    for history in tabu_list
        if are_similar(history.pattern, new_pattern, similarity_threshold)
            return true
        end
    end

    return false
end

"""
    take_step!(to_intensify, instance, operators, historical, solution)

This function allows us to take a step in the search space.
We implemented a set of operators to modify a pattern, such as reassigning a cluster to a different site or exchanging jobs between clusters.

A randomly-selected operator is applied as many times until we find a pattern that is not tabu.
"""
function take_step!(to_intensify::Bool, instance::PMSLPData, operators::Vector{Function}, historical::Vector{History}, solution::SearchSolution)::Tuple{Union{Pattern, Nothing}, String}
    new_pattern = deepcopy(solution.pattern)
    operator = first(sample(operators, 1))
    step = operator(instance, new_pattern)
    iterations, max_iterations = 0, 50
    is_valid = true

    while is_tabu(historical, new_pattern, to_intensify) && is_valid
        operator = first(sample(operators, 1))
        step = operator(instance, new_pattern)
        iterations += 1

        if iterations > max_iterations
            is_valid = false
            @warn "Reached Tabu max iterations"

            return nothing, step
        end
    end

    return new_pattern, step
end

"""
    should_continue(iteration, historical, max_iterations, max_iterations_no_improvement)

This function defines the stopping criteria of the search.
We stop the search when we reach the maximum number of iterations or when we do not improve the solution for a given number of iterations.
"""
function should_continue(iteration::Int64, historical::Vector{History}, max_iterations::Int64, max_iterations_no_improvement::Int64)::Bool
    iteration > max_iterations && return false

    if length(historical) > max_iterations_no_improvement
        improvements = diff([history.cost for history in historical[end - max_iterations_no_improvement:end]])
        did_not_improve = all(improvement -> improvement < 1, improvements)

        return did_not_improve
    end

    return true
end

"""
    acceptance_criteria(improvement, temperature, cooling_factor)

This function defines the acceptance criteria of the search.
We accept a new solution if it improves the current solution or if the acceptance probability is greater than a random number between 0 and 1 (uniform distribution).
The acceptance probability represents the willingness to accept a solution that does not improve the objective value but might lead to a better solution, and it is calculated as the exponential absolute improvement divided by the temperature.
The temperature is decreased by a cooling factor after each iteration that the acceptance probability is taken
"""
function acceptance_criteria(improvement::Float64, temperature::Float64, cooling_factor::Float64)::Bool
    improvement > 0 && return true

    probability = exp(improvement / temperature)
    temperature *= cooling_factor

    return rand() < probability
end

"""
    local_search(solver, instance, solution)

This function implements the local search algorithm based on strategies taken from well-known heuristics: simulated annealing, tabu search, and variable neighborhood search.
First, let a Cluster be a subset of jobs that will be planified together, and a Pattern be a set of Cluster-Site assignments. When the jobs in a cluster are scheduled in a sequence, we obtain a solution of our location-scheduling problem.

The local search algorithm iteratevely applies a set of operators that modifies an initial pattern until a stopping criteria is met.
These operators are selected based on the nature of the problem and the assumptions made. Since the quantity of machines is fixed, we can only modify the job-cluster and the cluster-site assignments, but not reduce or increase the number of clusters (i.e. split a cluster in two, merge two clusters, etc.).
Moreover, since we seek to solve the scheduling problem to optimality, we do not consider intra-cluster movements (i.e. shuffling jobs within a cluster). These operators, combined, allow us to obtain any solution in the search space from any other solution, therefore the operators are conex.
At each iteration, a randomly-selected operator is applied and the first-accepted resulting solution is considered the best solution (first-fit).
The acceptance criteria is based on the simulated annealing strategy, which allows us to accept solutions that do not improve the objective value but might lead to a better solution.
To prioritize the modifications of unattractive clusters in a pattern, we sort the Cluster-Site assignments by decreasing total tardiness and we select the first ones as the preferred for the operators (i.e. most inefficient clusters).
We decide to diversify or intensify the search based on the iteration number, diversifying at the beginning and intensifying at the end.
"""
function local_search(solver::SOLVER, instance::PMSLPData, solution::SearchSolution)::Tuple{SearchSolution, Vector{History}}
    operators = [reassign!, swap!]
    historical = [History("initial", solution.cost, solution.pattern)]
    best_solutions = SearchSolution[solution]
    iteration, max_iterations = 0, 1000
    max_iterations_no_improvement = ceil(Int64, max_iterations * 0.4)
    temperature, cooling_factor = 100.0, 0.99

    while should_continue(iteration, historical, max_iterations, max_iterations_no_improvement)
        iteration += 1
        to_intensify = iteration > 0.75 * max_iterations

        # sort!(solution.pattern.matches, by=match -> tardiness(instance, match[2], solution.assignments), rev=true)
        pattern, step = take_step!(to_intensify, instance, operators, historical, solution)
        isnothing(pattern) && continue
        assignments = schedule(solver, instance, pattern)
        next_solution = SearchSolution(instance, pattern, assignments)

        push!(historical, History(step, next_solution.cost, next_solution.pattern))
        improvement = solution.cost - next_solution.cost
        
        if acceptance_criteria(improvement, temperature, cooling_factor)
            push!(best_solutions, next_solution)
            solution = next_solution
        end

        @info "Iteration $(iteration) | Improvement $(improvement)" 
    end

    length(best_solutions) == 1 && @warn "Only one solution found"
    sort!(best_solutions, by=solution -> solution.cost, rev=false)

    return first(best_solutions), historical
end

"""
    metaheuristic(solver, instance)

This function implements the metaheuristic that combines the construction and local search heuristics.
The construction heuristic is applied first to obtain an initial solution, and then the local search heuristic is applied to improve the solution.
"""
function metaheuristic(solver::SOLVER, instance::PMSLPData)::Tuple{SearchSolution, Vector{History}}
    initial_solution = construction(solver, instance)
    improved_solution, history = local_search(solver, instance, initial_solution)
    @info "Local search | Improvement $(initial_solution.cost - improved_solution.cost)"

    return improved_solution, history
end

function solve(
    method::PMSLPHeuristic,
    instance::PMSLPData,
    solver::SOLVER,
)::PMSLPSolution
    time = @elapsed result, history = metaheuristic(solver, instance)
    solution = PMSLPSolution(
        method,
        opened_sites(result.pattern),
        result.assignments,
        Metrics(result.cost, time, MOI.LOCALLY_SOLVED),
        history,
    )
    format!(solution)
    validate(instance, solution)
    
    return solution
end

