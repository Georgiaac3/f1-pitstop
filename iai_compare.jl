# External check of our exact policy-tree solvers with Interpretable AI (P2).
#
# For each problem exported by run_p2.py (<dir>/<name>_*.csv) this fits
#   - OptimalTreePolicyMaximizer with cp = 0 at each depth 1..max_depth, the same
#     minbucket as our solver: same objective, so training values are comparable;
#   - a GridSearch over max_depth with cp tuned on a validation set of whole races.
# and writes the predicted actions on train and test, the fit times and the trees.
#
# Run with the IAI system image (IAI is preloaded, no `using IAI`):
#   julia +1.11.6/IAI/3.2.2 iai_compare.jl results_p2/iai stay_pit,compound

using CSV, DataFrames

dir = ARGS[1]
for name in split(ARGS[2], ",")
    rd(f) = CSV.read(joinpath(dir, "$(name)_$f.csv"), DataFrame)
    Xtr, G, Xte, valid, meta = rd("Xtrain"), rd("Gtrain"), rd("Xtest"), rd("valid"), rd("meta")
    mb, maxd = meta.minbucket[1], meta.max_depth[1]
    v = Bool.(valid.valid)
    pred_tr, pred_te = DataFrame(), DataFrame()
    info = DataFrame(model = String[], seconds = Float64[], params = String[])
    trees = IOBuffer()

    function record(label, lnr, secs, params)
        pred_tr[!, label] = string.(IAI.predict(lnr, Xtr))
        pred_te[!, label] = string.(IAI.predict(lnr, Xte))
        push!(info, (label, secs, params))
        println(trees, "==== ", label, " (", params, ")")
        show(trees, MIME("text/plain"), lnr)
        println(trees, "\n")
    end

    for d in 1:maxd
        lnr = IAI.OptimalTreePolicyMaximizer(max_depth = d, cp = 0.0, minbucket = mb,
                                             random_seed = 1)
        secs = @elapsed IAI.fit!(lnr, Xtr, G)
        record("iai_d$d", lnr, secs, "max_depth=$d, cp=0")
        println("$name depth $d: $(round(secs, digits = 1)) s")
    end

    grid = IAI.GridSearch(IAI.OptimalTreePolicyMaximizer(minbucket = mb, random_seed = 1),
                          max_depth = 1:maxd)
    secs = @elapsed IAI.fit!(grid, Xtr[.!v, :], G[.!v, :], Xtr[v, :], G[v, :])
    best = IAI.get_best_params(grid)
    record("iai_grid", IAI.get_learner(grid), secs,
           join(["$k=$(best[k])" for k in keys(best)], ", "))
    println("$name grid: $best")

    CSV.write(joinpath(dir, "$(name)_iai_pred_train.csv"), pred_tr)
    CSV.write(joinpath(dir, "$(name)_iai_pred_test.csv"), pred_te)
    CSV.write(joinpath(dir, "$(name)_iai_info.csv"), info)
    write(joinpath(dir, "$(name)_iai_trees.txt"), String(take!(trees)))
end
