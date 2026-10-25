# Voxel sampling and acquired-signal loss.
function ensemble(p,b0;nvoxels=256,gains=[.85,.925,1.,1.075,1.15],seed=3032026,exclude_voxels=Set{Tuple{Int,NTuple{3,Int}}}())
 tissue=[w<-200*2π ? 3 : t>.1 ? 1 : 2 for (w,t) in zip(p.Δw,p.T2)]
 voxels=Dict{NTuple{3,Int},Vector{Int}}()
 for i in eachindex(p.z)
  key=(round(Int,p.x[i]/.0015),round(Int,p.y[i]/.0015),round(Int,p.z[i]/.0015))
  push!(get!(Vector{Int},voxels,key),i)
 end
 # Fold aliased fat along z.
 fatvoxels=Dict{NTuple{3,Int},Vector{Int}}()
 for i in eachindex(p.z)
  tissue[i]==3 || continue
  key=(round(Int,p.x[i]/.0015),round(Int,p.y[i]/.0015),mod(round(Int,p.z[i]/.0015)+48,96)-48)
  push!(get!(Vector{Int},fatvoxels,key),i)
 end
 selected=NamedTuple[];rng=MersenneTwister(seed)
 for t in 1:3
  cells=Dict{Tuple{Int,Int},Vector{NamedTuple}}()
  candidates = t==3 ? fatvoxels : voxels
  for key in sort!(collect(keys(candidates)))
   (t,key) in exclude_voxels && continue
   ids=candidates[key]
   keep=filter(i->tissue[i]==t,ids);isempty(keep) && continue
   t<3 && (!( -36 <= key[3] <= 35) || length(keep)/length(ids)<.9) && continue
   cell=(clamp(floor(Int,(key[3]*.0015+.063)/.021),0,7),floor(Int,mean(b0[keep])/20))
   push!(get!(Vector{NamedTuple},cells,cell),(ids=keep,key=key))
  end
  isempty(cells) && error("No eligible held-out voxels remain for tissue $t")
  population=sum(length,v for v in values(cells));chosen=NamedTuple[]
  budget=nvoxels isa Number ? nvoxels : nvoxels[t]
  for cell in sort(collect(keys(cells)))
   v=cells[cell];m=budget>=population ? length(v) : max(1,round(Int,budget*length(v)/population));m=min(m,length(v))
   for q in randperm(rng,length(v))[1:m]
    push!(chosen,(tissue=t,ids=v[q].ids,weight=length(v)/(m*population),key=v[q].key))
   end
  end
  append!(selected,chosen)
 end
 x=Float64[];y=Float64[];z=Float64[];t1=Float64[];t2=Float64[];w=Float64[];chemical=Float64[];gain=Float64[];rho=Float64[];vid=Int[]
 vt=Int[];vc=Int[];vw=Float64[];vk=NTuple{3,Int}[];fieldnames=("native","map")
 for (field,f) in enumerate(fieldnames),(gi,g) in enumerate(gains),v in selected
  push!(vt,v.tissue);push!(vc,(field-1)*length(gains)+gi);push!(vw,v.weight);push!(vk,v.key);j=length(vt)
  for i in v.ids
   push!(x,p.x[i]);push!(y,p.y[i]);push!(z,p.z[i]);push!(t1,p.T1[i]);push!(t2,p.T2[i]);push!(gain,g)
   # Preserve B0 phase during recovery.
   push!(w,Float64(p.Δw[i])/(2π)+(f=="native" ? 0. : b0[i]))
   push!(chemical,Float64(p.Δw[i])/(2π));push!(rho,p.ρ[i]);push!(vid,j)
  end
 end
 prep_w=chemical.+2 .*round.((w.-chemical)./2)
 class_keys=unique(collect(zip(t1,t2,prep_w,gain)));lookup=Dict(k=>i for (i,k) in enumerate(class_keys))
 cls=[lookup[k] for k in zip(t1,t2,prep_w,gain)]
 println((stage="ACQUISITION_ENSEMBLE",spins=length(x),voxels=length(vt),prep_classes=length(class_keys),tissue_voxels=[count(v.tissue==t for v in selected) for t in 1:3]));flush(stdout)
 class_shift=zeros(length(class_keys));for i in eachindex(cls);class_shift[cls[i]]=chemical[i];end
 (;x,y,z,t1,t2,w,gain,rho,vid,vt,vc,vw,vk,cls,keys=class_keys,class_shift,gains,fieldnames)
end


function acquisition_steady_response(P,post,adc,d)
    classes=d["classes"]::Vector{Int};vid=d["voxel_ids"]::Vector{Int};rho=d["spin_weights"]::Vector{Float64}
    vt=d["voxel_tissues"]::Vector{Int};vc=d["voxel_cases"]::Vector{Int};vw=d["voxel_weights"]::Vector{Float64}
    ns=length(classes);nv=length(vt);na=size(adc,3)
    X=zeros(3,ns);U=similar(X);inverse=zeros(3,3,ns);S=zeros(ComplexF64,na,nv)
    @inbounds for spin in 1:ns
        cls=classes[spin]
        aa,bb,cc,dd,ee,ff,gg,hh,ii=ntuple(9) do k
            row=(k-1)÷3+1;col=(k-1)%3+1
            (row==col ? 1. : 0.)-sum(post[row,j,spin]*P[j,col,cls] for j in 1:3)
        end
        det=aa*(ee*ii-ff*hh)+bb*(ff*gg-dd*ii)+cc*(dd*hh-ee*gg)
        abs(det)>1e-12 || error("Singular heartbeat steady state")
        vals=(ee*ii-ff*hh,cc*hh-bb*ii,bb*ff-cc*ee,ff*gg-dd*ii,aa*ii-cc*gg,cc*dd-aa*ff,dd*hh-ee*gg,bb*gg-aa*hh,aa*ee-bb*dd)
        for k in 1:9;inverse[(k-1)÷3+1,(k-1)%3+1,spin]=vals[k]/det;end
        b=ntuple(a->sum(post[a,j,spin]*P[j,4,cls] for j in 1:3)+post[a,4,spin],3)
        for a in 1:3;X[a,spin]=sum(inverse[a,j,spin]*b[j] for j in 1:3);end
        for a in 1:3;U[a,spin]=sum(P[a,j,cls]*X[j,spin] for j in 1:3)+P[a,4,cls];end
        for sample in 1:na
            re=sum(adc[1,j,sample,spin]*U[j,spin] for j in 1:3)+adc[1,4,sample,spin]
            im=sum(adc[2,j,sample,spin]*U[j,spin] for j in 1:3)+adc[2,4,sample,spin]
            S[sample,vid[spin]]+=rho[spin]*complex(re,im)
        end
    end
    stats=zeros(3,maximum(vc))
    @inbounds for v in 1:nv,sample in 1:na
        t=vt[v];stats[t,vc[v]]+=vw[v]/na*(t==3 ? abs2(S[sample,v]) : abs(S[sample,v]))
    end
    stats[3,:].=sqrt.(max.(stats[3,:],1e-30))
    (;stats,S,X,U,inverse)
end

function acquisition_steady_pullback(P,post,adc,d,r,ds)
    classes=d["classes"]::Vector{Int};vid=d["voxel_ids"]::Vector{Int};rho=d["spin_weights"]::Vector{Float64}
    vt=d["voxel_tissues"]::Vector{Int};vc=d["voxel_cases"]::Vector{Int};vw=d["voxel_weights"]::Vector{Float64}
    ns=length(classes);nv=length(vt);na=size(adc,3);seedS=zeros(ComplexF64,size(r.S))
    @inbounds for v in 1:nv,sample in 1:na
        t=vt[v];c=vc[v];q=r.S[sample,v]
        seedS[sample,v]=ds[t,c]*vw[v]/na*q/(t==3 ? max(r.stats[3,c],1e-15) : max(abs(q),1e-15))
    end
    seed=zeros(size(P));gu=zeros(3);gx=zeros(3);lambda=zeros(3);total=zeros(3)
    @inbounds for spin in 1:ns
        cls=classes[spin];fill!(gu,0.)
        for sample in 1:na
            q=rho[spin]*seedS[sample,vid[spin]]
            for a in 1:3;gu[a]+=adc[1,a,sample,spin]*real(q)+adc[2,a,sample,spin]*imag(q);end
        end
        for a in 1:3;gx[a]=sum(P[j,a,cls]*gu[j] for j in 1:3);end
        for a in 1:3;lambda[a]=sum(r.inverse[j,a,spin]*gx[j] for j in 1:3);end
        for a in 1:3;total[a]=gu[a]+sum(post[j,a,spin]*lambda[j] for j in 1:3);end
        for a in 1:3
            for j in 1:3;seed[a,j,cls]+=total[a]*r.X[j,spin];end
            seed[a,4,cls]+=total[a]
        end
    end
    seed
end


function acquisition_deficits(stats,reference;contrast_margin=.05,fat_margin=.20,robustness_margin=.01,contrast_scale=1.,fat_scale=1.,signal_scale=1.,smooth_worst=false,nominal_indices=nothing,mean_weight=1.,fat_targets=nothing)
    mean_weight in (0.,1.) || error("mean_weight must be 0 or 1 for this ablation")
    smooth_worst || mean_weight==1. || error("p8-only requires smooth_worst=true")
    b=stats[1,:];m=stats[2,:];f=stats[3,:];br=reference[1,:];mr=reference[2,:];fr=reference[3,:]
    c=(b.-m)./(b.+m);cr=(br.-mr)./(br.+mr);nc=length(c)
    dc=zeros(nc);db=zeros(nc);dm=zeros(nc);df=zeros(nc);loss=0.
    fat_goal=isnothing(fat_targets) ? (1 .-fat_margin).*fr : fat_targets
    length(fat_goal)==nc && all(isfinite,fat_goal) && all(>=(0.),fat_goal) || error("Invalid fixed fat target")
    terms=((max.((1 .+contrast_margin).*cr.-c,0.)./(max.(cr,.03).*contrast_scale),:contrast),
        (max.(f.-fat_goal,0.)./(max.(fr,1e-4).*fat_scale),:fat),
        (max.(0.98 .*br.-b,0.)./(max.(br,1e-4).*signal_scale),:blood),
        (max.(0.85 .*mr.-m,0.)./(max.(mr,1e-4).*signal_scale),:myocardium),
        (max.((1 .+contrast_margin).*(br.-mr).-(b.-m),0.)./(max.(br.-mr,.01).*contrast_scale),:absolute_contrast))
    for (res,kind) in terms
        if smooth_worst
            u=mean(res.^8);loss+=mean_weight*mean(abs2,res)+u^.25
            dr=mean_weight.*2 .*res./nc.+(2/nc)*max(u,1e-30)^(-.75).*res.^7
        else
            loss+=sum(abs2,res)/nc+maximum(abs2,res)
            dr=2 .*res./nc;j=argmax(res);dr[j]+=2res[j]
        end
        kind==:contrast && (dc.-=dr./(max.(cr,.03).*contrast_scale))
        kind==:fat && (df.+=dr./(max.(fr,1e-4).*fat_scale))
        if kind==:absolute_contrast
            da=dr./(max.(br.-mr,.01).*contrast_scale)
            db.-=da;dm.+=da
        end
        kind==:blood && (db.-=dr./(max.(br,1e-4).*signal_scale))
        kind==:myocardium && (dm.-=dr./(max.(mr,1e-4).*signal_scale))
    end
    # Compare B1 variation within each field.
    nominal=isnothing(nominal_indices) ? fill(3,nc) : nominal_indices
    drop=(c[nominal].-c)./c[nominal];dropr=(cr[nominal].-cr)./cr[nominal]
    margin=fill(robustness_margin,nc);margin[unique(nominal)].=0.
    res=max.(drop.-dropr.+margin,0.)./.05
    if smooth_worst
        u=mean(res.^8);loss+=mean_weight*mean(abs2,res)+u^.25
        dr=mean_weight.*2 .*res./nc.+(2/nc)*max(u,1e-30)^(-.75).*res.^7
    else
        loss+=sum(abs2,res)/nc+maximum(abs2,res)
        dr=2 .*res./nc;j=argmax(res);dr[j]+=2res[j]
    end
    dr./=.05
    dc.-=dr./c[nominal]
    for k in 1:nc;dc[nominal[k]]+=dr[k]*c[k]/c[nominal[k]]^2;end
    db.+=dc.*(2 .*m)./(b.+m).^2;dm.-=dc.*(2 .*b)./(b.+m).^2
    loss,vcat(reshape(db,1,:),reshape(dm,1,:),reshape(df,1,:)),(;contrast=c,reference_contrast=cr,fat_ratio=f./fr,blood_ratio=b./br,relative_contrast_drop=drop,reference_drop=dropr)
end
