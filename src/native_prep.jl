# Native RF propagation and manual adjoint.
using KomaMRIBase, KomaMRICore, KomaMRIFiles, CUDA, JLD2, Random, LinearAlgebra, Statistics, SparseArrays
const C = KomaMRICore
const KA = C.KA
include("prep_objective.jl")
include("heatmap_loss.jl")

const SEQUENCE_CACHE = Dict{String,Any}()
function read_cached_sequence(path)
    path=abspath(path);stamp=(mtime(path),filesize(path))
    cached=get(SEQUENCE_CACHE,path,nothing)
    !isnothing(cached) && cached.stamp==stamp && return cached.sequence
    sequence=read_seq(path;verbose=false)
    SEQUENCE_CACHE[path]=(;stamp,sequence)
    sequence
end

native_params() = C.default_sim_params(Dict{String,Any}("gpu"=>true,"precision"=>"f64",
    "Δt"=>1e-3,"Δt_rf"=>1e-6,"max_block_length"=>64,"max_rf_block_length"=>512))
function native_discretize(seq)
    params = native_params()
    d=discretize(seq; sampling_rule=C.simulation_sampling_rule(params["sim_method"],params), freq_in_phase=params["freq_in_phase"])
    gap=sum(seq.DUR)-sum(d.Δt)
    if gap>1e-10
        # Preserve recovery after a trailing trigger.
        d=DiscreteSequence(vcat(d.Gx,0.),vcat(d.Gy,0.),vcat(d.Gz,0.),vcat(d.B1,0im),
            vcat(d.Δf,0.),vcat(d.ψ,0.),vcat(d.ADC,false),vcat(d.excitation_bool,false),
            vcat(d.t,last(d.t)+gap),vcat(d.Δt,gap))
    end
    d
end
native_system() = Scanner(limits=HardwareLimits(B0=3.,B1=20e-6,Gmax=.08,Smax=180.,
    RF_Δt=1e-6,GR_Δt=10e-6,DUR_Δt=10e-6,ADC_Δt=1e-7,
    RF_dead_time=100e-6,RF_ring_down_time=20e-6,ADC_dead_time=10e-6))

function basis_state(n, column)
    # Columns: x, y, z, recovery.
    xy = column==1 ? ones(ComplexF64,n) : column==2 ? fill(1.0im,n) : zeros(ComplexF64,n)
    z = column==3 ? ones(n) : zeros(n)
    C.Mag(xy,z)
end
function native_phantom(x,y,z,t1,t2,w; rho=ones(length(x)))
    Phantom(; x=Float64.(x),y=Float64.(y),z=Float64.(z),T1=Float64.(t1),T2=Float64.(t2),
        T2s=Float64.(t2),Δw=2π.*Float64.(w),ρ=Float64.(rho))
end

# Native RF and per-spin ADC operators.
function native_operators(seq, e; readout=true)
    n = length(e.x); ns = count(a->a.N>0,seq.ADC)
    maps=zeros(3,4,n); adcmap=zeros(2,4,ns,n)
    sys=native_system(); backend=C.get_backend(true;verbose=false)
    for gain in unique(e.gain)
        ids=findall(==(gain),e.gain)
        s=deepcopy(seq)
        for j in eachindex(s.RF)
            rf=deepcopy(s.RF[j]);rf.A=rf.A.*gain;s.RF[j]=rf
        end
        d=native_discretize(s)
        # Sample each ADC center.
        centers=Int[];cursor=0
        for a in s.ADC
            a.N>0 || continue
            push!(centers,cursor+a.N÷2+1);cursor+=a.N
        end
        ai=findall(d.ADC); selected=ai[centers]; d.ADC .= false; d.ADC[selected] .= true
        parts, active=C.get_sim_ranges(d;max_block_length=64,max_rf_block_length=512)
        dd=C.gpu(d)
        k=last(get_kspace(s))[centers,:]; rx=get_adc_phase_compensation(s)[centers]
        correction=[rx[j]*cis(2π*sum(k[j,a]*round(evalue/.0015)*.0015 for (a,evalue) in enumerate((e.x[i],e.y[i],e.z[i])))) for j in 1:ns,i in ids]
        for col in 1:4
            p=native_phantom(e.x[ids],e.y[ids],e.z[ids],e.t1[ids],e.t2[ids],e.w[ids];rho=fill(col==4 ? 1. : 0.,length(ids)))
            p=C.gpu(p); state=C.gpu(basis_state(length(ids),col))
            pre=C.prealloc(C.Bloch(),backend,p,state,513,0,128,sys)
            sample=1
            signal=CUDA.zeros(ComplexF64,ns,length(ids),1)
            for (r,on) in zip(parts,active)
                q=@view dd[r]; na=count(@view d.ADC[first(r)+1:last(r)])
                if on
                    na==0 || error("Training ADC during RF is unsupported")
                    C.run_spin_excitation!(p,q,CUDA.zeros(ComplexF64,0,1),state,sys,C.Bloch(),128,backend,pre)
                else
                    C.run_spin_precession!(p,q,@view(signal[sample:sample+na-1,:,:]),state,sys,C.BlochDict(),128,backend,C.DefaultPrealloc{Float64}())
                end
                KA.synchronize(backend); sample+=na
            end
            maps[1,col,ids].=real.(Array(state.xy)); maps[2,col,ids].=imag.(Array(state.xy)); maps[3,col,ids].=Array(state.z)
            if ns>0
                sig=dropdims(Array(signal);dims=3).*correction
                adcmap[1,col,:,ids].=real.(sig);adcmap[2,col,:,ids].=imag.(sig)
            end
            CUDA.reclaim()
        end
    end
    maps,adcmap
end

function build_native_training(p,b0,conv,opt; nvoxels=256,gains=[.85,.925,1.,1.075,1.15],
        ensemble_seed=3032026,acquisition_arms=8,exclude_voxels=Set{Tuple{Int,NTuple{3,Int}}}())
    p.motion isa NoMotion || error("This training model requires a static phantom")
    e=ensemble(p,b0;nvoxels,gains,seed=ensemble_seed,exclude_voxels)
    minbeats=min(Int(conv.DEF["TotalHeartbeats"]),Int(opt.DEF["TotalHeartbeats"]))
    beats=unique(round.(Int,range(6,minbeats-1;length=acquisition_arms)))
    F=Dict{String,Any}();A=Dict{String,Any}()
    for (name,s,lo,hi) in (("conventional",conv,2,1),("optimized",opt,4,2))
        nb=length(s.DUR)÷Int(s.DEF["TotalHeartbeats"]);ads=Array{Float64,4}[]
        for beat in beats
            pair=resolve_triggers(s[(beat-1)*nb+1:(beat+1)*nb],CardiacSignal(;heart_rate=1.))
            part=pair[lo:nb+hi]
            started=time();f,a=native_operators(part,e)
            if isempty(ads);F[name]=f
            else;norm(f-F[name])/norm(f)<1e-8 || error("Heartbeat state map varies across arms")
            end
            push!(ads,a);println((stage="NATIVE_OPERATORS",name,beat,seconds=time()-started));flush(stdout)
        end
        A[name]=cat(ads...;dims=3)
    end
    Dict("post"=>F["optimized"],"adc"=>A["optimized"],"convcycle"=>F["conventional"],"convadc"=>A["conventional"],
        "classes"=>e.cls,"spin_weights"=>e.rho,"voxel_ids"=>e.vid,"voxel_tissues"=>e.vt,
        "voxel_cases"=>e.vc,"voxel_weights"=>e.vw,"voxel_keys"=>e.vk,
        "prep_t1"=>[k[1] for k in e.keys],"prep_t2"=>[k[2] for k in e.keys],
        "prep_frequency_hz"=>[k[3] for k in e.keys],"prep_b1"=>[k[4] for k in e.keys],
        "gains"=>e.gains,"fields"=>collect(e.fieldnames),"positions"=>hcat(e.x,e.y,e.z),
        "acquisition_beats"=>beats,"solver"=>"KomaMRICore")
end

function rf_interpolation(rf, t; sample_times=Float64.(times(rf)))
    # Preserve both sides of RF discontinuities.
    n=length(rf.A)
    length(sample_times)==n+2 || error("Expected sampled RF with two zero endpoints")
    rows=Int[];cols=Int[];vals=Float64[]
    function add(i,k,w)
        2<=k<=n+1 || return
        push!(rows,i);push!(cols,k-1);push!(vals,w)
    end
    sample=1;i=1
    while i<=length(t)
        at=t[i];j=i
        while j<length(t) && t[j+1]==at;j+=1;end
        while sample<=length(sample_times) && sample_times[sample]<at;sample+=1;end
        if sample<=length(sample_times) && sample_times[sample]==at
            lastsample=sample
            while lastsample<length(sample_times) && sample_times[lastsample+1]==at;lastsample+=1;end
            for k in i:j;add(k,min(sample+k-i,lastsample),1.);end
            sample=lastsample+1
        elseif first(sample_times)<=at && sample<=length(sample_times)
            lo=sample-1;w=(at-sample_times[lo])/(sample_times[sample]-sample_times[lo])
            for k in i:j;add(k,lo,1-w);add(k,sample,w);end
        end
        i=j+1
    end
    sparse(rows,cols,vals,length(t),n)
end

function rf_problem(seq,data;rf_controls=256,energy_cap,chunk=512)
    prep=seq[3];rf=prep.RF[1,1];wave=ComplexF64.(rf.A).*cis(rf.ϕ)
    n=length(wave);k=min(rf_controls,n);u=collect(range(0.,1.;length=n));v=collect(range(0.,1.;length=k))
    B=[exp(-.5*((a-b)/max(1.5/k,1/n))^2)*sinpi(a)^2 for a in u,b in v]
    Q=Matrix(qr(B).Q*Matrix{Float64}(I,n,k))
    cap=20e-6;raw=cap.*atanh.(min.(abs.(wave)./cap,.999999)).*wave./max.(abs.(wave),1e-30)
    q=native_discretize(prep)
    samples=KomaMRIBase.block_event_waveforms(prep,1;freq_in_phase=false).rf
    L=rf_interpolation(rf,q.t;sample_times=samples.t)
    all(g -> maximum(abs,g;init=0.)<1e-12, (q.Gx,q.Gy,q.Gz)) ||
        error("The RF-only class model requires gradient-free preparation RF")
    maximum(abs.(L*wave.-q.B1))<1e-10 || error("RF interpolation differs from KomaMRI")
    rt=Float64.(times(rf));weights=zeros(n);dt=diff(rt)
    for i in 1:length(dt)
        2<=i<=n+1 && (weights[i-1]+=dt[i]/2)
        2<=i+1<=n+1 && (weights[i]+=dt[i]/2)
    end
    groups=Any[];nc=length(data["prep_t1"]);sys=native_system();be=C.get_backend(true;verbose=false)
    for gain in unique(data["prep_b1"])
        ids=findall(==(gain),data["prep_b1"]);m=length(ids)
        p=native_phantom(zeros(4m),zeros(4m),zeros(4m),repeat(data["prep_t1"][ids],4),repeat(data["prep_t2"][ids],4),repeat(data["prep_frequency_hz"][ids],4);rho=vcat(zeros(3m),ones(m)))
        initial=C.Mag(vcat(ones(ComplexF64,m),fill(1.0im,m),zeros(ComplexF64,2m)),vcat(zeros(2m),ones(m),zeros(m)))
        pp=C.gpu(p);state=C.gpu(deepcopy(initial));dd=C.gpu(q)
        pre=C.prealloc(C.Bloch(),be,pp,state,chunk+1,0,128,sys)
        push!(groups,(;ids,gain,p=pp,initial=C.gpu(initial),state,q=dd,pre,
            checkpoints=Any[],ranges=[i:min(i+chunk,length(q.t)) for i in 1:chunk:length(q.Δt)]))
    end
    (;seq,rf,wave,raw,Q,L,weights,cap,energy_cap,k,nc,q,groups,sys,be)
end

function rf_wave(prob,x)
    raw=prob.raw+1e-6.*prob.Q*complex.(x[1:prob.k],x[prob.k+1:end])
    r=abs.(raw);unit=raw./max.(r,1e-30)
    w=prob.cap.*tanh.(r./prob.cap).*unit
    e=sum(prob.weights.*abs2.(w));scale=min(1.,sqrt(prob.energy_cap/max(e,1e-30)))
    (;wave=scale.*w,raw,r,unit,unscaled=w,scale,energy=e)
end
function wave_pullback(prob,x,g)
    v=rf_wave(prob,x)
    if v.scale<1
        g=v.scale.*(g.-prob.weights.*v.unscaled.*(real(dot(v.unscaled,g))/v.energy))
    end
    scale=ifelse.(v.r.>1e-30,prob.cap.*tanh.(v.r./prob.cap)./max.(v.r,1e-30),1.)
    radial=1 .-tanh.(v.r./prob.cap).^2
    gg=scale.*g.+(radial.-scale).*real.(conj.(v.unit).*g).*v.unit
    vcat(real.(1e-6.*prob.Q'*gg),imag.(1e-6.*prob.Q'*gg))
end

function native_prep_maps!(prob,x)
    wave=rf_wave(prob,x).wave;b=prob.L*wave;P=zeros(3,4,prob.nc)
    for g in prob.groups
        copyto!(g.q.B1,g.gain.*b);copyto!(g.state.xy,g.initial.xy);copyto!(g.state.z,g.initial.z)
        for pair in g.checkpoints;foreach(CUDA.unsafe_free!,pair);end
        empty!(g.checkpoints)
        for r in g.ranges
            q=@view g.q[r]
            C.run_spin_excitation!(g.p,q,CUDA.zeros(ComplexF64,0,1),g.state,prob.sys,C.Bloch(),128,prob.be,g.pre)
            KA.synchronize(prob.be)
            push!(g.checkpoints,(copy(g.state.xy),copy(g.state.z)))
        end
        m=length(g.ids);xy=reshape(Array(g.state.xy),m,4);z=reshape(Array(g.state.z),m,4)
        for c in 1:4;P[1,c,g.ids].=real.(xy[:,c]);P[2,c,g.ids].=imag.(xy[:,c]);P[3,c,g.ids].=z[:,c];end
    end
    P
end

@inline function adjoint_coeff(u,v,w,h)
    r=sqrt(u*u+v*v+w*w);k=-π*γ
    s,a=sincos(k*r*h);q=iszero(r) ? k*h : s/r
    a,-w*q,v*q,-u*q,r,s
end
@inline function undo_rotation(x,y,z,a,c,b,d)
    # Transpose rotation.
    ((a*a-c*c-b*b+d*d)*x-2*(a*c+b*d)*y-2*(a*b-c*d)*z,
     2*(a*c-b*d)*x+(a*a-c*c+b*b-d*d)*y-2*(a*d+c*b)*z,
     2*(c*d+a*b)*x+2*(a*d-c*b)*y+(a*a+c*c-b*b-d*d)*z)
end
@inline function rf_step_pullback(x,y,z,u,v,w,h,t1,t2,sx,sy,sz)
    a,c,b,d,r,s=adjoint_coeff(u,v,w,h);k=-π*γ
    sx*=exp(-h/t2);sy*=exp(-h/t2);sz*=exp(-h/t1)
    dx,dy,dz=undo_rotation(sx,sy,sz,a,c,b,d)
    ga=sx*(2a*x+2c*y+2b*z)+sy*(2a*y-2c*x+2d*z)+sz*(2a*z-2b*x-2d*y)
    gc=sx*(-2c*x+2a*y+2d*z)+sy*(-2c*y-2a*x-2b*z)+sz*(2c*z+2d*x-2b*y)
    gb=sx*(-2b*x-2d*y+2a*z)+sy*(2b*y-2d*x-2c*z)+sz*(-2b*z-2a*x-2c*y)
    gd=sx*(2d*x-2b*y+2c*z)+sy*(-2d*y-2b*x+2a*z)+sz*(-2d*z+2c*x-2a*y)
    if r>0
        radial=(-ga*s*k*h+(-w*gc+v*gb-u*gd)*(a*k*h/r-s/(r*r)))/r
        du=-gd*s/r+radial*u;dv=gb*s/r+radial*v
    else
        du=-gd*k*h;dv=gb*k*h
    end
    dx,dy,dz,du,dv
end
@inline function warp_sum(x)
    for o in (16,8,4,2,1);x+=CUDA.shfl_down_sync(UInt32(0xffffffff),x,o);end
    x
end
function native_reverse_kernel!(gr,gi,xy,z,sxy,sz,q,t1,t2,rho,pw,lo,hi,n)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    x=0.;y=0.;zz=0.;sx=0.;sy=0.;ssz=0.
    if i<=n
        x=real(xy[i]);y=imag(xy[i]);zz=z[i];sx=real(sxy[i]);sy=imag(sxy[i]);ssz=sz[i]
        sn,cs=sincos(-q.psi[hi]);x,y=cs*x-sn*y,sn*x+cs*y;sx,sy=cs*sx-sn*sy,sn*sx+cs*sy
    end
    for j in hi-1:-1:lo
        dr=0.;di=0.
        if i<=n
            h=q.dt[j];u=real(q.b1[j]);v=imag(q.b1[j]);w=pw[i]/(2π*γ)-q.df[j]/γ
            a,c,b,d,_,_=adjoint_coeff(u,v,w,h)
            # Reconstruct the prior state.
            ux=x*exp(h/t2[i]);uy=y*exp(h/t2[i]);uz=(zz-rho[i]*(1-exp(-h/t1[i])))*exp(h/t1[i])
            px,py,pz=undo_rotation(ux,uy,uz,a,c,b,d)
            sx,sy,ssz,dr,di=rf_step_pullback(px,py,pz,u,v,w,h,t1[i],t2[i],sx,sy,ssz)
            x,y,zz=px,py,pz
        end
        dr=warp_sum(dr);di=warp_sum(di)
        if (threadIdx().x-1)%32==0
            CUDA.@atomic gr[j]+=dr
            CUDA.@atomic gi[j]+=di
        end
    end
    if i<=n
        sn,cs=sincos(q.psi[lo]);sx,sy=cs*sx-sn*sy,sn*sx+cs*sy
        sxy[i]=complex(sx,sy);sz[i]=ssz
    end
    return
end

function native_prep_pullback(prob,x,seed)
    gradient=zeros(ComplexF64,length(prob.q.t))
    for g in prob.groups
        m=length(g.ids)
        sx=CuArray(vec(permutedims(complex.(seed[1,:,g.ids],seed[2,:,g.ids]),(2,1))))
        sz=CuArray(vec(permutedims(seed[3,:,g.ids],(2,1))))
        gr=CUDA.zeros(Float64,length(prob.q.t));gi=similar(gr);fill!(gi,0.)
        q=(b1=g.q.B1,dt=g.q.Δt,df=g.q.Δf,psi=g.q.ψ)
        for k in reverse(eachindex(g.ranges))
            r=g.ranges[k];xy,z=g.checkpoints[k]
            CUDA.@sync CUDA.@cuda threads=128 blocks=cld(4m,128) native_reverse_kernel!(
                gr,gi,xy,z,sx,sz,q,g.p.T1,g.p.T2,g.p.ρ,g.p.Δw,first(r),last(r),4m)
        end
        gradient .+= g.gain.*complex.(Array(gr),Array(gi))
        foreach(CUDA.unsafe_free!,(sx,sz,gr,gi))
    end
    wave_pullback(prob,x,prob.L'*gradient)
end

function normalize_rf_raster(seq, scanner)
    out = deepcopy(seq)
    normalized_rf = Matrix{RF}(out.RF)
    raster = hasproperty(scanner, :limits) ? scanner.limits.RF_Δt : scanner.RF_Δt
    for i in eachindex(out.RF)
        rf = out.RF[i]
        maximum(abs, rf.A isa AbstractVector ? rf.A : [rf.A]) == 0 && continue
        if rf.T isa AbstractVector
            cumulative = cumsum(vcat(0.0, rf.T))
            aligned = round.(cumulative ./ raster) .* raster
            maximum(abs.(aligned .- cumulative)) < 1e-11 ||
                error("RF is materially off the requested raster")
            intervals = diff(aligned)
            if !isempty(intervals) && maximum(abs.(intervals .- first(intervals))) < 1e-11
                replacement = RF(copy(rf.A), last(aligned), rf.Δf, rf.delay;
                                 center=rf.center, ϕ=rf.ϕ, use=rf.use)
                maximum(abs.(times(replacement) .- times(rf))) < 1e-11 ||
                    error("Uniform RF representation changes sample times")
                normalized_rf[i] = replacement
            else
                rf.T .= intervals
            end
        end
        # Preserve half-raster RF delays.
    end
    return Sequence(out.GR, normalized_rf, out.ADC, out.DUR, out.EXT, out.DEF)
end


function export_native_prep(prob,x,path)
    out=deepcopy(prob.seq);wave=rf_wave(prob,x).wave
    nb=length(out.DUR)÷Int(out.DEF["TotalHeartbeats"])
    for j in 3:nb:length(out.DUR)
        rf=deepcopy(out.RF[1,j]);rf.A=wave.*cis(-rf.ϕ);out.RF[1,j]=rf
    end
    out.DEF["OptimizationForwardSolver"]="KomaMRICore.Bloch"
    out.DEF["OptimizationGradient"]="manual_adjoint_native_checkpoints"
    sys=native_system()
    write_seq(normalize_rf_raster(out,sys),path;sys,verbose=false,check_hw_limits=true)
    path
end

function save_checkpoint(path;values...)
    jldsave(path*".pending";values...)
    mv(path*".pending",path;force=true)
end

function fit_native_prep(initial_path,data,energy;output,rf_controls,energy_fraction,seconds,iterations,heatmap,fat_targets,resume=nothing)
    seq=read_cached_sequence(initial_path)
    prob=rf_problem(seq,data;rf_controls,energy_cap=energy*energy_fraction*1e-15)
    gridprob=rf_problem(seq,heatmap["data"];rf_controls,energy_cap=energy*energy_fraction*1e-15)
    identity=zeros(3,4,prob.nc);for j in 1:3;identity[j,j,:].=1.;end
    ref=acquisition_steady_response(identity,data["convcycle"],data["convadc"],data).stats
    ng=length(data["gains"]);nom=argmin(abs.(data["gains"].-1.));nominal=[(k-1)÷ng*ng+nom for k in axes(ref,2)]
    cached=fill(NaN,2prob.k);value=Ref(Inf);gradient=zeros(2prob.k);latest=Ref{Any}(nothing);calls=Ref(0)
    function update!(x)
        x==cached && return
        P=native_prep_maps!(prob,x)
        response=acquisition_steady_response(P,data["post"],data["adc"],data)
        loss,ds,metrics=acquisition_deficits(response.stats,ref;contrast_margin=.005,
            fat_margin=.02,robustness_margin=.005,contrast_scale=.01,
            fat_scale=.05,signal_scale=.02,smooth_worst=true,mean_weight=0.,nominal_indices=nominal,fat_targets)
        ps=acquisition_steady_pullback(P,data["post"],data["adc"],data,response,10 .*ds)
        grad=native_prep_pullback(prob,x,ps)
        field_response=grid_response(gridprob,x,heatmap["data"])
        field_loss,dz=heatmap_loss(field_response.z,heatmap["data"]["labels"],heatmap["goals"])
        seed=zeros(size(field_response.maps))
        seed[3,3,:].=field_response.E.*dz
        seed[3,4,:].=field_response.E.*dz
        grad.+=native_prep_pullback(gridprob,x,seed)
        wave=rf_wave(prob,x).wave;soft=17e-6
        excess=max.(abs.(wave)./soft.-1,0.)
        penalty=.05*mean(abs2,excess)
        pg=0.1/length(wave).*excess./soft.*wave./max.(abs.(wave),1e-30)
        grad.+=wave_pullback(prob,x,pg)
        cached.=x;gradient.=grad;value[]=10loss+field_loss+penalty;calls[]+=1
        latest[]=(;P,stats=response.stats,metrics,acquired_loss=loss,field_loss,peak_loss=penalty,
            heatmap_Mz=copy(field_response.z),energy_uT2_ms=1e15sum(prob.weights.*abs2.(wave)))
    end
    objective(x)=(update!(x);value[])
    gradient!(g,x)=(update!(x);copyto!(g,gradient))
    x=zeros(2prob.k)
    if !isnothing(resume)
        old=JLD2.load(resume)
        isapprox(old["seed_rf"],prob.rf;rtol=1e-9,atol=1e-12) || error("Resume pulse differs from the initial RF")
        length(old["parameters"])==length(x) || error("Resume control count differs")
        x.=old["parameters"]
    end
    cold=@elapsed initial=objective(x)
    direction=randn(MersenneTwister(1032026),length(x));direction./=norm(direction);h=1e-3
    analytic=dot(gradient,direction);numeric=(objective(x+h*direction)-objective(x-h*direction))/(2h)
    derivative_error=abs(analytic-numeric)/max(abs(analytic),abs(numeric),1e-12)
    derivative_error<1e-4 || error("Manual adjoint differs from KomaMRI finite difference: $derivative_error")
    warm=@elapsed objective(x)
    println((stage="NATIVE_FIT_READY",cold,warm,initial,derivative_error));flush(stdout)
    history=NamedTuple[];started=time()
    callback=s->begin
        update!(s.x)
        push!(history,(iteration=length(history),loss=value[],acquired=latest[].acquired_loss,
            field=latest[].field_loss,peak=latest[].peak_loss,seconds=time()-started))
        save_checkpoint(joinpath(output,"checkpoint.jld2");parameters=copy(s.x),history,seed_rf=prob.rf)
        length(history)%20==1 && (println((stage="NATIVE_FIT_ITER",history[end]...));flush(stdout))
        false
    end
    fit=Optim.optimize(objective,gradient!,x,Optim.LBFGS(m=20,linesearch=Optim.LineSearches.BackTracking(iterations=20,order=2,maxstep=1.)),
        Optim.Options(iterations=iterations,time_limit=seconds,g_abstol=1e-7,f_reltol=1e-10,callback=callback))
    x=Optim.minimizer(fit);final=objective(x);fit_seconds=time()-started
    mkpath(output);path=joinpath(output,"optimized_t2.seq")
    export_started=time();export_native_prep(prob,x,path)
    receipt=(;path,initial_loss=initial,final_loss=final,optimization_seconds=fit_seconds,
        export_seconds=time()-export_started,
        iterations=Optim.iterations(fit),cold,warm,derivative_error,forward_calls=calls[],
        acquired_loss=latest[].acquired_loss,field_loss=latest[].field_loss,peak_loss=latest[].peak_loss,
        objective="10*acquired_p8 + heatmap_p8 + RF_peak",
        rf_energy_uT2_ms=latest[].energy_uT2_ms,rf_energy_ratio=latest[].energy_uT2_ms/energy,
        forward_solver="KomaMRICore.Bloch",gradient="manual_native_checkpoint_adjoint")
    save_checkpoint(joinpath(output,"fit.jld2");receipt,parameters=x,history,seed_rf=prob.rf,
        final_stats=latest[].stats,final_prep_maps=latest[].P,heatmap,final_heatmap_Mz=latest[].heatmap_Mz,fat_targets)
    receipt
end
