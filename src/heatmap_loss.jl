# One p8 aggregation per tissue. No mean-squared term.
function heatmap_loss(z,labels,goals;scale=.02)
    value=0.;dz=zeros(length(z))
    for tissue in 1:4
        ids=findall(==(tissue),labels)
        residual=(z[ids].-goals[tissue])./scale
        u=mean(residual.^8)
        value+=u^.25
        dz[ids] .= (2/length(ids))*max(u,1e-30)^(-.75) .* residual.^7 ./ scale
    end
    value,dz
end

function model_tissues(train)
    rows=unique(collect(zip(train["prep_t1"],train["prep_t2"],train["prep_frequency_hz"])))
    fat=rows[argmin(last.(rows))]
    water=filter(r->r[3]>-200,rows)
    blood=water[argmax(getindex.(water,2))];myo=water[argmin(getindex.(water,2))]
    [("water",1.,.1,0.),("blood",blood...), ("myocardium",myo...), ("fat",fat...)]
end
function heatmap_grid(tissues,offsets,gains;absolute=false)
    a=Float64[];b=Float64[];w=Float64[];g=Float64[];labels=Int[];off=Float64[]
    for (t,(_,t1,t2,shift)) in enumerate(tissues),gain in gains,f in offsets
        push!(a,t1);push!(b,t2);push!(w,absolute ? f : f+shift)
        push!(g,gain);push!(labels,t);push!(off,f)
    end
    Dict("prep_t1"=>a,"prep_t2"=>b,"prep_frequency_hz"=>w,"prep_b1"=>g,
        "labels"=>labels,"offsets"=>collect(offsets),"gains"=>collect(gains),"case_offset"=>off)
end
function prep_end(seq)
    findfirst(j->seq.RF[1,j].use isa Excitation && maximum(abs,ampls(seq.RF[1,j]);init=0.)>0,eachindex(seq.DUR))-1
end
function conventional_heatmap(seq,data,zs)
    n=length(data["prep_t1"]);nz=length(zs);nspin=n*nz
    e=(;x=zeros(nspin),y=zeros(nspin),z=repeat(zs,n),t1=repeat(data["prep_t1"];inner=nz),
        t2=repeat(data["prep_t2"];inner=nz),w=repeat(data["prep_frequency_hz"];inner=nz),gain=repeat(data["prep_b1"];inner=nz))
    maps,_=native_operators(seq[2:prep_end(seq)],e;readout=false)
    result=vec(mean(reshape(maps[3,3,:].+maps[3,4,:],nz,n);dims=1))
    CUDA.reclaim();GC.gc()
    result
end
function grid_response(prob,x,data)
    maps=native_prep_maps!(prob,x)
    post=sum(prob.seq.DUR[4:prep_end(prob.seq)])
    E=exp.(-post./data["prep_t1"])
    z=E.*(maps[3,3,:].+maps[3,4,:]).+1 .-E
    (;z,maps,E)
end

function build_heatmap_training(train, conventional, phantom)
    tissues = model_tissues(train)
    data = heatmap_grid(tissues, collect(-100.:10.:100.), collect(.85:.05:1.15))
    zs = sort(unique(Float64.(phantom.z[abs.(phantom.z) .< .00075])))
    target = conventional_heatmap(conventional, data, zs)
    nominal = (abs.(data["case_offset"]) .< 1e-8) .& (abs.(data["prep_b1"] .- 1.) .< 1e-8)
    goals = [target[only(findall((data["labels"] .== tissue) .& nominal))] for tissue in 1:3]
    push!(goals, 0.)
    Dict("data"=>data, "goals"=>goals, "conventional_Mz"=>target, "tissues"=>tissues, "z_positions"=>zs)
end
