using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using JuMP
using Gurobi
using ArgParse
using Clustering
using Hungarian
using Distributed
using Random
using Dates

abstract type Method end
struct BaseModel <: Method end
struct PMSLPMIPModel1 <: Method end
struct PMSLPMIPModel2 <: Method end
struct PMSLPHeuristic <: Method end
struct PMSLPBenders <: Method end

const SOLVER = MOI.OptimizerWithAttributes
const JOB_PREFIX = "JOB-"
const SITE_PREFIX = "SITE-"
const SWAP_PREFIX = "swap_"
const REASSIGN_PREFIX = "reassign_"

include("src/services/instances.jl")
include("src/services/outputs.jl")
include("src/services/benchmark.jl")

include("src/methods/BaseModel.jl")
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
        method = eval(Symbol("PMSLPHeuristic"))()
    else
        try
            method = eval(Symbol(model_name))()
        catch e
            error("Invalid model name: $model_name. Not registered. Error: $e")
        end
    end

    @info "Solving $(instance.name) with $(method) | Jobs: $(nb_jobs(instance)) | Sites: $(nb_jobs(instance)) | Machines: $(nb_machines(instance))"

    return solve(method, instance, solver)
end

function execute(args::Dict)::PMSLPSolution
    # Set solver
    verbose = get(args, "verbose", false) == true ? 1 : 0
    solver = optimizer_with_attributes(
        Gurobi.Optimizer,
        "OutputFlag" => verbose,
        "TimeLimit" => args["limit"],
    )

    # Load instance
    filename = get(args, "filename", nothing)
    instance = isnothing(filename) ? args["instance"] : PMSLPData(filename)

    # Optimize
    model_name = args["model"]
    solution = optimize(model_name, instance, solver)
    show(instance, solution)

    # Export solution
    output_path = joinpath(dirname(@__FILE__), "outputs")
    export_solution(model_name, joinpath(output_path, "$(filename)_$(model_name)"), instance, solution)

    # If args has the key : benchmark and its value is true, then record the model metrics in a csv file
    run_benchmark = get(args, "benchmark", true)
    run_benchmark && add_model!(joinpath(output_path, "benchmark.csv"), instance, solution)

    return solution
end

function main()::Nothing
    parser = ArgParseSettings()
    @add_arg_table! parser begin
        "--filename"
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
            # ]
        "--limit"
            help="Time limit"
            arg_type = Int
            default=3600
        # verbose = default en true?
    end

    _ = execute(parse_args(parser))

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end