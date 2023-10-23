using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

include("src/domain/models.jl")
include("src/domain/helpers.jl")
include("src/services/instances.jl")
include("src/services/estimations.jl")
include("src/domain/PMSLPMIPModel1.jl")
include("src/services/outputs.jl")

using .Models: PMSLPSolution
using .Instances: load
using .Estimations: build_matrices
using .PMSLPMIPModel1: MIP1, solve
using .Outputs: export_solution!

using ArgParse: ArgParseSettings, @add_arg_table!, parse_args

function optimize(model::String; kwargs...)::PMSLPSolution
    model_number = match(r"\d+", model)

    if !isnothing(model_number)
        model_struct = eval(Symbol("MIP$(model_number.match)"))()
    elseif occursin(model, "PMSLPHeuristic")
        model_struct = eval(Symbol("Heuristic"))()
    else
        @error "Invalid model name: $model"
    end

    return solve(model_struct, kwargs[:instance], kwargs[:matrices])
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
        "--limit"
            help="Time limit"
            arg_type = Int
            default=3600
    end

    args = parse_args(parser)

    # Load instance
    filename = args["instance"]
    instance = load(filename, timeout=args["limit"])

    # Build time matrix
    matrices = build_matrices(instance.jobs, instance.sites, instance.parameters)

    # Optimize
    model = args["model"]
    solution = optimize(model, instance=instance, matrices=matrices)

    # Export solution
    output_path = joinpath(dirname(@__FILE__), "outputs", "$(filename)_$(model)")
    export_solution!(output_path, instance, solution)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end