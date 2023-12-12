using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using JuMP # : Model, @variable, @objective, @constraint, optimize!, value, objective_value, optimizer_with_attributes, termination_status
using Gurobi #: Optimizer
using ArgParse #: ArgParseSettings, @add_arg_table!, parse_args

abstract type Method end
struct PMSLPMIPModel1 <: Method end
struct PMSLPMIPModel2 <: Method end
struct PMSLPHeuristic <: Method end
struct PMSLPBenders <: Method end

const SOLVER = MOI.OptimizerWithAttributes
const JOB_PREFIX = "JOB-"
const SITE_PREFIX = "SITE-"

include("src/services/instances.jl")
include("src/services/outputs.jl")

include("src/methods/PMSLPMIPModel1.jl")
include("src/methods/PMSLPMIPModel2.jl")
include("src/methods/PMSLPHeuristic.jl")

"""
Dispatches to the correct method to solve the problem and returns the solution.
"""
function optimize(model_name::String, instance::PMSLPData, solver::SOLVER)::PMSLPSolution
    model_number = match(r"\d+", model_name)
    method = nothing

    if !isnothing(model_number)
        model = Symbol("PMSLPMIPModel$(model_number.match)")
        method = eval(model)()
    elseif occursin(uppercase(model_name), uppercase("PMSLPHeuristic"))
        method = eval(Symbol("Heuristic"))()
    else
        @error "Invalid model name: $model_name"
    end

    @info "Solving $(method) | Jobs: $(nb_jobs(instance)) | Sites: $(nb_jobs(instance))"

    return solve(method, instance, solver)
end

function main()
    parser = ArgParseSettings()
    @add_arg_table! parser begin
        "--instance"
            help="Instance file name"
            arg_type = String
            required=true
        "--model"
            help="Model name"
            arg_type = String
            required=true
            # TODO restrict choices: If not here, after reading the args (validate with possible values)
            # choices=[
            #     "MP1", "PMSLPMIPModel1",
            #     "MP2", "PMSLPMIPModel2",
            #     "H", "Heuristic",
            #     "Benders",
            # ]
        "--limit"
            help="Time limit"
            arg_type = Int
            default=3600
    end
    args = parse_args(parser)

    # Set solver
    solver = optimizer_with_attributes(
        Gurobi.Optimizer,
        "OutputFlag" => 1,
        "TimeLimit" => args["limit"],
    )

    # Load instance
    filename = args["instance"]
    instance = PMSLPData(filename)

    # Optimize
    model_name = args["model"]
    solution = optimize(model_name, instance, solver)
    show(instance, solution)

    # Export solution
    output_path = joinpath(dirname(@__FILE__), "outputs", "$(filename)_$(model_name)")
    export_solution(output_path, instance, solution)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end