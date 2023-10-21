module PMSLPMIPModel1

using ..Helpers: build_horizon
using ..Models: PMSLPData, PMSLPSolution, Assignment
using JuMP: Model, @variable, @objective, @constraint, optimize!, value,
    objective_value, set_optimizer_attribute, termination_status
using Gurobi: Optimizer

export MIP1, solve

struct MIP1 end

function solve(
    ::MIP1,
    instance::PMSLPData,
    matrices::Dict{Symbol, Matrix}
)::PMSLPSolution
    model = Model(Optimizer)
    set_optimizer_attribute(model, "TimeLimit", instance.parameters[:timeout])

    jobs = instance.jobs
    weights = instance.parameters[:weights]
    P = 1:build_horizon(instance, matrices[:time])
    K = 1:length(instance.sites)
    J = 1:length(jobs)
    M = instance.machines
    β = instance.parameters[:tardiness_penalty]
    λ1, λ2, λ3 = weights[:fixed_cost], weights[:travel_time], weights[:tardiness]
    Ω = [(i, j) for i=J, j=J if i != j]

    @variable(model, x[k=K], Bin) # open sites
    @variable(model, s[j=J, k=K, t=P], Bin) # start time
    @variable(model, T[j=J] >= 0, Int) # tardiness
    
    @objective(
        model,
        Min,
        λ1 * sum(site.fixed_cost * x[k] for (k, site) in enumerate(instance.sites))
        + λ2 * sum(2 * matrices[:cost][j, k] * s[j, k, t] for j=J, k=K, t=P)
        + λ3 * sum(T[j] * β for j=J)
    )
    
    @constraint(model, sum(x[k] for k=K) == M) # install all machines
    @constraint(model, [j=J], sum(s[j, k, t] for k=K, t=P) == 1) # jobs are executed once at one site
    @constraint(model, [j=J, k=K], sum(s[j, k, t] for t=1:matrices[:time][j, k]) == 0) # jobs are executed after they arrive
    @constraint(
        model,
        [k=K, t=P, (i, j)=Ω],
        s[i, k, t] + sum(s[j, k, τ] for τ=t:min(t+jobs[i].processing_time, maximum(P))) <= x[k]
    ) # non-overlapping jobs
    @constraint(
        model,
        [j=J],
        T[j] >= sum(s[j, k, t] * (t + jobs[j].processing_time + matrices[:time][j, k]) for k=K, t=P) - jobs[j].due_date
    ) # tardiness
    
    optimize!(model)

    @info "Model status: $(termination_status(model))"

    function build_output()
        open_sites = [site for (k, site) in enumerate(instance.sites) if value(x[k]) >= 0.5]
        assignments = []
    
        for (j, job) in enumerate(instance.jobs)
            tardiness = value(T[j])
    
            for (k, site) in enumerate(instance.sites)
                for t in P
                    if value(s[j, k, t]) >= 0.5
                        finish_at = t + job.processing_time + matrices[:time][j, k]
                        @assert tardiness == max(0, finish_at - job.due_date)
                        assignment = Assignment(job.id, site.id, t, finish_at, tardiness)
                        push!(assignments, assignment)
    
                        break
                    end
                end
            end
        end
    
        solution_value = objective_value(model)
    
        return PMSLPSolution(solution_value, open_sites, assignments)
    end

    return build_output()
end

end