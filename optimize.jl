#!/usr/bin/env julia
# Design joint T₂-prep/FatSat RF with native KomaMRI.
module BSSFPPrep

using TOML, LinearAlgebra, Statistics

const ROOT = @__DIR__

# Study inputs
conventional_path = "$ROOT/data/conventional.seq"
initial_path = "$ROOT/data/initial_40ms.seq"
phantom_path = "$ROOT/data/cardiac.phantom"
output_dir = "$ROOT/results/joint_p8"
initial_controls_path = "$ROOT/data/initial_controls.jld2"
fat_targets_path = "$ROOT/data/fat_targets.toml"
b0_map = nothing
b1_gains = [.85, .925, 1., 1.075, 1.15]

# Optimization
voxels = 256
rf_controls = 256
iterations = 500
seconds = 2400.
energy_fraction = 1.
replay_rf_step = 20e-6

module NativePrep
include("src/native_prep.jl")
end
const N = NativePrep

module Reconstruction
using Reexport
@reexport using KomaMRICore, KomaMRIFiles
using FFTW: fftshift, ifftshift, fft, ifft
using MRIReco
include("deps/KomaMRI/src/reconstruction/Recon.jl")
end

export prepare!, fit!, replay!, report!, run!
result(name) = joinpath(output_dir,name)
const OPERATORS = Ref{Any}(nothing)

function select_gpu()
    N.CUDA.functional() || error("A CUDA GPU is required")
    println((stage="GPU",device=N.CUDA.name(N.CUDA.device())))
end

function input_stamp()
    files=filter(!isnothing,(conventional_path,initial_path,phantom_path,b0_map))
    (;files=[(abspath(p),mtime(p),filesize(p)) for p in files],gains=copy(b1_gains),voxels,
      koma=string(Base.pkgversion(N.C)),rf_step=N.native_params()["Δt_rf"],model="joint_p8_heatmap")
end

function read_b0(p)
    field=isnothing(b0_map) ? zeros(length(p)) : vec(N.JLD2.load(b0_map,"b0_hz"))
    length(field)==length(p) && all(isfinite,field) || error("B0 map must match phantom spins")
    field
end

function protocol()
    conventional = N.read_cached_sequence(conventional_path)
    initial = N.read_cached_sequence(initial_path)
    for key in ("Nx", "Ny", "Nz", "FOV", "ViewsPerSegment", "TotalHeartbeats", "DummyHeartbeats")
        conventional.DEF[key] == initial.DEF[key] || error("Sequences differ in $key")
    end
    initial.DEF["Nz"] == 96 || error("The existing coherent training model requires 96 partitions")
    all(isapprox.(Float64.(initial.DEF["FOV"]) ./ [initial.DEF[k] for k in ("Nx", "Ny", "Nz")], .0015; atol=1e-9)) ||
        error("The existing training model requires 1.5 mm voxels")
    isapprox(initial.DEF["HeartbeatInterval"], 1.; atol=1e-10) || error("The existing training model uses 60 bpm")
    Int(initial.DEF["DummyHeartbeats"]) == 5 || error("The existing model requires five dummy heartbeats")
    nbeats = Int(initial.DEF["TotalHeartbeats"])
    nb = length(initial.DUR) ÷ nbeats
    prep = findall(j -> maximum(abs, N.ampls(initial.RF[1,j]); init=0.) > 0 &&
                       !(initial.RF[1,j].use isa N.Excitation), 1:nb)
    prep == [3] || error("Initial sequence must have one joint preparation RF in block 3 of each heartbeat")
    conventional, initial, nb
end

# Fixed acquisition
function prepare!(;reuse=false)
    mkpath(output_dir)
    if reuse
        OPERATORS[]=N.JLD2.load(result("operators.jld2"))
        return operators()
    end
    select_gpu()
    conventional,initial,_=protocol()
    p=N.read_phantom(phantom_path);b0=read_b0(p)
    println((stage="FIELD",mean_hz=mean(b0),std_hz=std(b0;corrected=false),range_hz=extrema(b0)))
    training=N.build_native_training(p,b0,conventional,initial;nvoxels=voxels,gains=b1_gains)
    excluded=Set(zip(training["voxel_tissues"],training["voxel_keys"]))
    heldout=N.build_native_training(p,b0,conventional,initial;nvoxels=max(1,voxels÷4),
        gains=[.875,1.,1.125],ensemble_seed=2032026,exclude_voxels=excluded)
    heatmap=N.build_heatmap_training(training,conventional,p)
    inputs=input_stamp()
    N.save_checkpoint(result("operators.jld2");training,heldout,heatmap,inputs)
    OPERATORS[]=Dict("training"=>training,"heldout"=>heldout,"heatmap"=>heatmap,"inputs"=>inputs)
    settings=Dict(string(k)=>getfield(@__MODULE__,k) for k in
        (:conventional_path,:initial_path,:phantom_path,:output_dir,:initial_controls_path,:fat_targets_path,:b1_gains,:voxels,:rf_controls,:iterations,:seconds,:energy_fraction,:replay_rf_step))
    isnothing(b0_map) || (settings["b0_map"]=b0_map)
    open(result("parameters.toml"),"w") do io;TOML.print(io,settings);end
    OPERATORS[]
end

function operators()
    isnothing(OPERATORS[]) && (OPERATORS[]=N.JLD2.load(result("operators.jld2")))
    OPERATORS[]["inputs"]==input_stamp() || error("Inputs changed; run prepare!()")
    OPERATORS[]
end

# RF fitting
function fit!(;resume=initial_controls_path)
    select_gpu()
    0<energy_fraction<=1 || error("Energy fraction must be in (0,1]")
    data=operators()
    Core.eval(N,:(using Optim))
    Base.invokelatest(fit_impl!,data;resume)
end

function fit_impl!(data;resume)
    conventional=N.read_cached_sequence(conventional_path)
    energy=prep_energy(conventional)
    targets=TOML.parsefile(fat_targets_path)
    targets["b1_gains"]==b1_gains || error("Fat targets use different B1 gains")
    targets["fields"]==data["training"]["fields"] || error("Fat targets use different field conditions")
    receipt=N.fit_native_prep(initial_path,data["training"],energy;output=output_dir,
        rf_controls,energy_fraction,seconds,iterations,heatmap=data["heatmap"],fat_targets=targets["fat_targets"],resume)
    cp(conventional_path,result("conventional_t2.seq");force=true)
    for name in ("training","heldout");write_case_table(data[name],name);end
    println((stage="FIT_COMPLETE",receipt));flush(stdout)
    receipt
end

function prep_energy(seq)
    first_excitation = findfirst(j -> seq.RF[1,j].use isa N.Excitation &&
        maximum(abs, N.ampls(seq.RF[1,j]); init=0.) > 0, eachindex(seq.DUR))
    isnothing(first_excitation) && error("Sequence contains no imaging RF")
    energy = 0.
    for j in 1:first_excitation-1
        rf = seq.RF[1,j]
        time = N.times(rf)
        power = abs2.(N.ampls(rf))
        energy += sum(diff(time) .* (power[1:end-1] .+ power[2:end]) ./ 2)
    end
    energy * 1e15                         # T² s -> µT² ms
end

function prep_maps(path,data)
    seq = N.read_cached_sequence(path)
    # Evaluate the exported waveform.
    prob = N.rf_problem(seq,data;rf_controls,energy_cap=Inf)
    # Zero controls reproduce the seed RF.
    N.native_prep_maps!(prob, zeros(2prob.k))
end

function write_case_table(data,name)
    n = length(data["prep_t1"])
    identity = zeros(3,4,n)
    for j in 1:3; identity[j,j,:] .= 1.; end
    reference = N.acquisition_steady_response(identity, data["convcycle"], data["convadc"], data).stats
    maps = prep_maps(result("optimized_t2.seq"),data)
    stats = N.acquisition_steady_response(maps, data["post"], data["adc"], data).stats
    gains = data["gains"]
    nominal = argmin(abs.(gains .- 1.))
    contrast(a, k) = (a[1,k] - a[2,k]) / (a[1,k] + a[2,k])
    open(joinpath(output_dir, name * "_metrics.tsv"), "w") do io
        println(io, "field\tb1\tcontrast_ratio\tabsolute_contrast_ratio\tfat_rms_ratio\tblood_ratio\tcontrast_drop\tconventional_drop")
        for k in axes(stats,2)
            field = data["fields"][(k-1) ÷ length(gains) + 1]
            gain = gains[mod1(k, length(gains))]
            nom = (k-1) ÷ length(gains) * length(gains) + nominal
            values = (field, gain, contrast(stats,k)/contrast(reference,k),
                (stats[1,k]-stats[2,k])/(reference[1,k]-reference[2,k]),
                stats[3,k]/reference[3,k], stats[1,k]/reference[1,k],
                1-contrast(stats,k)/contrast(stats,nom), 1-contrast(reference,k)/contrast(reference,nom))
            println(io, join(values, '\t'))
        end
    end
end

# Full phantom
function verify_export()
    seq = N.read_cached_sequence(result("optimized_t2.seq"))
    sys = N.native_system()
    N.check_timing(seq, sys)
    N.check_hw_limits(seq, sys)
    seq, sys
end

function simulate_image(phantom, seq, sys)
    dims = Tuple(Int(seq.DEF[k]) for k in ("Nx", "Ny", "Nz"))
    raw = Reconstruction.simulate(phantom, seq, sys;
        sim_params=Dict{String,Any}("gpu"=>true, "precision"=>"f64", "Δt"=>1e-3,
                                   "Δt_rf"=>replay_rf_step, "sim_method"=>N.C.Bloch()),
        physio=Reconstruction.CardiacSignal(;heart_rate=1.), verbose=false)
    raw.params["trajectory"] = "cartesian"
    for profile in raw.profiles
        profile.head.center_sample = UInt16(dims[1] ÷ 2)
    end
    image = Reconstruction.reconstruct_with_labels(raw; rec_params=Dict{Symbol,Any}(:reco=>"direct"))
    reshape(ComplexF32.(Array(only(image.images).image.data)), dims)
end

function replay!()
    select_gpu()
    optimized, sys = verify_export()
    conventional = N.read_cached_sequence(conventional_path)
    N.JLD2.jldsave(result("optimized_prep.jld2"); rf=optimized.RF[1,3],
        energy_uT2_ms=prep_energy(optimized), conventional_energy_uT2_ms=prep_energy(conventional))
    phantom = Reconstruction.read_phantom(phantom_path)
    b0 = read_b0(phantom)
    fields = isnothing(b0_map) ? ("native",) : ("native", "map")
    mkpath(result("replay"))
    for field in fields, (name,seq) in (("conventional",conventional),("optimized",optimized)), tissue in ("full","fat")
        p = deepcopy(phantom)
        field == "map" && (p.Δw .+= Float32.(2π .* b0))
        tissue == "fat" && (p = p[findall(<(-200*2π), phantom.Δw)])
        out = result("replay/$(name)_$(field)_$(tissue).jld2")
        println((stage="REPLAY_START", name, field, tissue, spins=length(p)))
        flush(stdout)
        seconds = @elapsed volume = simulate_image(p, seq, sys)
        N.JLD2.jldsave(out; volume, seconds, spins=length(p), b1_scale=1., rf_step_seconds=replay_rf_step)
        println((stage="REPLAY_DONE", out, seconds))
        flush(stdout)
        N.CUDA.reclaim()
        GC.gc()
    end
    output_dir
end

# Comparison
function report!()
    Core.eval(N,:(using CairoMakie))
    Base.invokelatest(make_report!)
end

function preparation_trace(seq)
    first_excitation=findfirst(j->seq.RF[1,j].use isa N.Excitation &&
        maximum(abs,N.ampls(seq.RF[1,j]);init=0.)>0,eachindex(seq.DUR))
    waves=Tuple{Vector{Float64},Vector{Float64}}[];offset=0.
    for j in 2:first_excitation-1
        rf=seq.RF[1,j];amplitude=abs.(N.ampls(rf))
        maximum(amplitude;init=0.)>0 && push!(waves,((offset .+N.times(rf)).*1e3,amplitude.*1e6))
        offset+=seq.DUR[j]
    end
    (;waves,window_ms=offset*1e3)
end

include("src/report.jl")

function make_report!()
    masks, dims = water_masks()
    fields = isnothing(b0_map) ? ("native",) : ("native", "map")
    names = ("conventional", "optimized")
    crop = [max(1,n÷2-51):min(n,n÷2+52) for n in dims[1:2]]
    results, panels = Dict{String,Any}(), Dict{String,Any}()
    for field in fields
        panels[field] = (Any[], Any[])
        for name in names
            base = result("replay/$(name)_$(field)")
            full, fat = [abs.(N.JLD2.load(base*"_$(tissue).jld2", "volume")) for tissue in ("full", "fat")]
            results["$(field)_$(name)"] = image_metrics(full, fat, masks)
            for (kind,volume) in enumerate((full, fat))
                push!(panels[field][kind], volume[crop...,dims[3]÷2+1])
            end
        end
        ref, opt = [results["$(field)_$(name)"] for name in names]
        results[field*"_ratios"] = Dict(k => opt[k]/ref[k] for k in keys(ref))
    end
    sequences = N.read_cached_sequence.((conventional_path, result("optimized_t2.seq")))
    traces = preparation_trace.(sequences)
    results["rf"] = Dict(name*"_"*metric => value for (name,seq,trace) in zip(names,sequences,traces)
        for (metric,value) in (("energy_uT2_ms",prep_energy(seq)), ("window_ms",trace.window_ms)))
    open(io -> TOML.print(io, results), result("image_metrics.toml"), "w")
    plot_comparison(traces, panels, fields)
    println((stage="IMAGE_METRICS", results))
    results
end
function run!()
    prepare!()
    fit!()
    replay!()
    report!()
end

end

if abspath(PROGRAM_FILE)==@__FILE__
    isempty(ARGS) || error("Set study variables at the top of the script")
    BSSFPPrep.run!()
end
