function water_masks()
    seq = N.read_cached_sequence(initial_path)
    dims = Tuple(Int(seq.DEF[k]) for k in ("Nx", "Ny", "Nz"))
    spacing = Float64.(seq.DEF["FOV"]) ./ collect(dims)
    p = N.KomaMRIFiles.read_phantom(phantom_path)
    density = zeros(Float32, dims..., 3)
    for i in eachindex(p.x)
        ix = (round(Int,p.x[i]/spacing[1])+dims[1]÷2+1,
              round(Int,p.y[i]/spacing[2])+dims[2]÷2+1,
              round(Int,p.z[i]/spacing[3])+dims[3]÷2+1)
        all(j -> 1 <= ix[j] <= dims[j], 1:3) || continue
        tissue = p.Δw[i] < -200*2π ? 3 : p.T2[i] > .1 ? 1 : 2
        density[ix...,tissue] += p.ρ[i]
    end
    total = dropdims(sum(density; dims=4); dims=4)
    slab = get(seq.DEF, "SlabThickness", dims[3]*spacing[3])
    # Water ROIs: central 90% of slab, >90% tissue purity.
    z = ((1:dims[3]) .- (dims[3]÷2+1)) .* spacing[3]
    # Use a half-open slab interval.
    slabmask = reshape((z .>= -0.45slab - 1e-10) .& (z .< 0.45slab - 1e-10), 1,1,:)
    masks = [(density[:,:,:,k] ./ max.(total,1e-20) .> .9) .& (total .> 0) .& slabmask for k in 1:2]
    masks, dims
end

function image_metrics(full, fat, masks)
    blood, myocardium = [mean(full[mask]) for mask in masks]
    Dict("blood_mean" => blood, "myocardium_mean" => myocardium,
         "normalized_contrast" => (blood-myocardium)/(blood+myocardium),
         "absolute_contrast" => blood-myocardium, "fat_rms" => sqrt(mean(abs2, fat)))
end

function plot_comparison(traces, panels, fields)
    figure = N.Figure(size=(1100, 300+560length(fields)))
    titles = ("Conventional T₂ prep + FatSat", "Optimized joint preparation")
    for col in 1:2
        axis = N.Axis(figure[1,col]; title=titles[col], xlabel="Preparation time (ms)", ylabel="|B₁| (µT)")
        for (t,b) in traces[col].waves
            N.lines!(axis, t, b; color=col==1 ? "#526879" : "#087e8b", linewidth=1.5)
        end
        N.xlims!(axis, 0., maximum(t.window_ms for t in traces))
        N.ylims!(axis, 0., 20.)
    end
    for (fi,field) in enumerate(fields), kind in 1:2
        images = panels[field][kind]
        vmax = maximum(maximum, images)
        title = kind==1 ? "Magnitude" : "Fat-only signal"
        length(fields)>1 && (title *= " · "*field)
        for col in 1:2
            axis = N.Axis(figure[1+2(fi-1)+kind,col]; title, aspect=N.DataAspect())
            plot = N.heatmap!(axis, images[col]; colormap=kind==1 ? :grays : :magma, colorrange=(0.,vmax))
            N.hidedecorations!(axis)
            N.hidespines!(axis)
            col==2 && N.Colorbar(figure[1+2(fi-1)+kind,3], plot; label="a.u.", width=15)
        end
    end
    for extension in ("png", "svg")
        N.save(result("comparison."*extension), figure)
    end
end
