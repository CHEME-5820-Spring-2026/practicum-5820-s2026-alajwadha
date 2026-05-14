include("Include.jl"); # load packages, src/ files, set random seed

# â”€â”€ Setup: hyperparameters for the whole practicum â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STUDENT TASK: Uncomment the block below, then tweak the TODO knobs.
# We unpack everything in a `let` block so each binding is created exactly once
# and returned together as a tuple. Touch a knob, re-run this cell, and every
# downstream cell that references these names picks up the new value.

n_subjects, images_per_subject, Î²_gen, Î²_cond, Î²_soft = let

    # initialize -
    n_subjects = 10;          # TODO: how many of the 40 Olivetti subjects to memorize (Int 1..40)
    images_per_subject = 10;  # TODO: how many portraits per subject (Int 1..10)

    # Î² controls the sharpness of the softmax inside Stochastic Attention.
    # Large Î² â†’ near-one-hot weights (snap onto a single stored portrait).
    # Small Î² â†’ uniform weights (blended/mean state).
    Î²_gen  = 0.02; # TODO: SA temperature for Task 1 (unconditional generation). Try 0.01, 0.02, 0.05.
    Î²_cond = 8.0; # TODO: SA temperature for the hard-mask sampler (Task 2a) â€” large so the chain locks onto in-class memories.
    Î²_soft = 0.02; # TODO: SA temperature for the log-multiplicity bias sampler (Task 2b). Picked so Î²Â·X'Â·s logit range Î” is comparable to log(Ï_max). Try 0.01, 0.02, 0.05.

    # return all five as a tuple.
    n_subjects, images_per_subject, Î²_gen, Î²_cond, Î²_soft;
end;

# STUDENT TASK: Uncomment the block below to load the Olivetti faces.
# Loads `n_subjects` subjects Ã— `images_per_subject` portraits from the Olivetti
# face dataset. `X` is a (4096, N) matrix â€” one column per portrait, each column
# is a flattened 64Ã—64 grayscale image with pixel values in [0, 1]. `y` is a
# length-N vector of subject ids in 0..n_subjects-1.

X, y = load_olivetti_subset(images_per_subject; n_subjects = n_subjects);
println("loaded $(size(X, 2)) faces of dimension $(size(X, 1)) covering $(length(unique(y))) subjects")

y

# Shape sanity checks. If any of these fail, every downstream cell will be
# wrong, so we want a loud failure here rather than a silent surprise later.
let
    @assert size(X, 1) == 4096                                # 64 Ã— 64 pixels per face, flattened column-major
    @assert size(X, 2) == n_subjects * images_per_subject     # one column per portrait, total = n_subjects Ã— images_per_subject
    @assert length(y) == size(X, 2)                           # one subject id per column
    @assert sort(unique(y)) == collect(0:(n_subjects - 1))    # the first n_subjects subjects, 0..n_subjects-1
end

# Preview: one portrait per subject, laid out in a balanced grid with per-tile subject ids.
# This is sanity-checking the data, not part of the model â€” if the faces look like
# faces and the subject ids are correctly aligned with the images, we're good.
let
    ncols = 5
    nrows = cld(n_subjects, ncols) # ceil-division so every subject gets a tile
    plots = [];

    # One tile per subject: pick the FIRST stored portrait of that subject.
    for s in 0:(n_subjects - 1)
        idx = findfirst(==(s), y);                                    # column index of the first portrait of subject s
        img  = Gray.(reshape(X[:, idx], 64, 64));                     # un-flatten into a 64Ã—64 grayscale image
        push!(plots, heatmap(img, color = :grays, axis = false, ticks = false,
            title = "subjid: $s", titlefontsize = 8, aspect_ratio = :equal));
    end

    # Pad the grid with empty plots so the layout is rectangular even when
    # n_subjects is not a multiple of ncols (e.g. 10 subjects in a 3Ã—5 grid).
    for _ in (n_subjects + 1):(nrows * ncols)
        push!(plots, plot(framestyle = :none));
    end

    # Render the grid.
    plot(plots...; layout = (nrows, ncols),
        size = (150 * ncols, 160 * nrows),
        margin = 0Plots.PlotMeasures.mm)
end

# â”€â”€ Build the three Stochastic Attention samplers and the per-class means â”€â”€
# STUDENT TASK: Uncomment the block below. Tweak step_size / noise_scale TODOs
# if you want to explore â€” defaults are reasonable.
# We need three different SA models because each task uses a different Î².
# Building them once here means later cells can just call `stochastic_attention_sample`
# without re-constructing the model on every call.

sa_gen, sa_cond, sa_soft, class_means = let

    # initialize - we set these to `nothing` so the build calls below fill them in.
    sa_gen      = nothing; # Task 1 â€” unconditional generation (low Î², soft blend)
    sa_cond     = nothing; # Task 2a â€” hard-mask conditional (high Î², in-class lock-on)
    sa_soft     = nothing; # Task 2b â€” log-multiplicity bias (low Î² so log(Ï) can compete)
    class_means = nothing; # one mean portrait per subject

    # Task 1 sampler â€” low Î² so the softmax is soft and the chain blends memories.
    sa_gen = build(MyStochasticAttentionModel, (
        memories     = X,
        labels       = y,
        Î²            = Î²_gen,
        step_size    = 1.0,  # TODO: try 0.25, 0.5, 1.0
        noise_scale  = 0.10, # TODO: try 0.02 (regurgitation), 0.10, 0.30
    ));

    # Task 2a sampler â€” high Î² so attention is near one-hot. Combined with the
    # hard mask the chain locks onto in-class memories.
    sa_cond = build(MyStochasticAttentionModel, (
        memories     = X,
        labels       = y,
        Î²            = Î²_cond,
        step_size    = 1.0,
        noise_scale  = 0.10,
    ));

    # Task 2b sampler â€” low Î² so log(Ï) (the soft-bias logit shift) can compete
    # with the data term Î²Â·X'Â·s. If Î² is too large the soft bias gets steam-rolled.
    sa_soft = build(MyStochasticAttentionModel, (
        memories     = X,
        labels       = y,
        Î²            = Î²_soft,
        step_size    = 1.0,
        noise_scale  = 0.10,
    ));

    # Per-subject mean portraits, used by the nearest-centroid classifier.
    class_means = build_class_means(X, y);

    # return -
    sa_gen, sa_cond, sa_soft, class_means;
end;

# â”€â”€ Task 1: unconditional Stochastic Attention generation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Run `n_samples_unmasked` Langevin chains starting from random stored portraits
# and let each chain run for n_steps. At low Î² the chain mixes between memories
# rather than collapsing onto one of them.
n_samples_unmasked, unmasked_initial, unmasked_samples = let

    # how many parallel chains to run. Each becomes one column of the output matrix.
    n_samples_unmasked = 10;

    # Build the warm-start matrix EXPLICITLY so we can show "start vs final" later.
    # `sa_initial_states` returns a (d, n) matrix where each column is a random
    # stored portrait plus a small Gaussian kick of size sa_gen.noise_scale.
    unmasked_initial = sa_initial_states(sa_gen, n_samples_unmasked);

    # STUDENT TASK: call `stochastic_attention_sample` on `sa_gen`, draw
    # `n_samples_unmasked` chains, run for n_steps = 3000, starting from
    # `unmasked_initial`. Assign the result to `unmasked_samples`.
    unmasked_samples = stochastic_attention_sample(sa_gen, n_samples_unmasked; n_steps = 3000, sâ‚’ = unmasked_initial);

    # return -
    n_samples_unmasked, unmasked_initial, unmasked_samples;
end;

# â”€â”€ Visualization: time evolution of a single SA chain â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# We pull one chain (chain_idx = 1) out of the warm-start matrix and run it
# forward in three stages, capturing intermediate states. This shows the
# trajectory: noisy start â†’ burn-in â†’ more steps â†’ final denoised read-out.
let

    # initialize -
    chain_idx = 1                                    # which chain to follow (1..n_samples_unmasked)
    sâ‚’      = unmasked_initial[:, chain_idx:chain_idx]  # keep as (d, 1) column matrix, not a vector

    # Run the chain in stages so we can snapshot intermediate states.
    # `denoise = false` means we keep the noisy Langevin state; `true` means
    # we strip the final noise term and return the underlying signal.
    s_burn  = stochastic_attention_sample(sa_gen, 1; n_steps = 2000, sâ‚’ = sâ‚’,     denoise = false)  # after burn-in
    s_more  = stochastic_attention_sample(sa_gen, 1; n_steps =  500, sâ‚’ = s_burn, denoise = false)  # +500 more steps
    s_final = stochastic_attention_sample(sa_gen, 1; n_steps =  500, sâ‚’ = s_more, denoise = true)   # +500 more, denoised read-out

    # Lay out the snapshots side-by-side as a 1Ã—4 grid: start, burn-in, more, final.
    cols   = [sâ‚’, s_burn, s_more, s_final]
    titles = ["start", "+2000", "+2500", "+3000 (final)"]
    plots  = []
    for (col, t) in zip(cols, titles)
        # un-flatten 4096-vector â†’ 64Ã—64 image, clamp to [0, 1] for display
        img = Gray.(clamp.(reshape(col[:, 1], 64, 64), 0.0, 1.0))
        push!(plots, heatmap(img; color = :grays, axis = false, ticks = false, colorbar = false,
            title = t, titlefontsize = 8, aspect_ratio = :equal))
    end
    plot(plots...; layout = (1, 4), size = (640, 180), margin = 0Plots.PlotMeasures.mm)
end

# Pick one of the 10 unconditional chains to inspect more closely.
chain_index = let
    chain_index = 1; # TODO: pick a chain index in 1:n_samples_unmasked
    @assert 1 <= chain_index <= n_samples_unmasked  # guard against typos
    chain_index;
end
println("unconditional Â· chain $chain_index")

# Show start vs final for the picked chain. Top row of the row-pair model:
# left = warm-start input, right = SA sample after 3000 steps.
let
    cols   = [unmasked_initial[:, chain_index], unmasked_samples[:, chain_index]]
    titles = ["start", "final"]
    plots  = []
    for (col, t) in zip(cols, titles)
        img = Gray.(clamp.(reshape(col, 64, 64), 0.0, 1.0))   # 4096-vec â†’ 64Ã—64 image
        push!(plots, heatmap(img; color = :grays, axis = false, ticks = false, colorbar = false,
            title = t, titlefontsize = 8, aspect_ratio = :equal))
    end
    plot(plots...; layout = (1, 2), size = (320, 180), margin = 0Plots.PlotMeasures.mm)
end

# â”€â”€ Novelty diagnostic: how close are SA samples to the nearest stored portrait? â”€â”€
# We compare two distributions:
#   â€¢ median nearest-stored distance  â€” for each SA sample, distance to the closest stored portrait
#   â€¢ median pairwise distance        â€” typical distance between two distinct stored portraits
# If the first is much smaller than the second, the chain is regurgitating memories.
# If the two are comparable, the chain is generating novel blends.
let
    nearest_dist = Float64[]
    for k in 1:size(unmasked_samples, 2)
        s = unmasked_samples[:, k]
        # min over all stored portraits â€” distance from sample k to its nearest memory.
        push!(nearest_dist, minimum(norm(s .- X[:, j]) for j in 1:size(X, 2)))
    end

    # Reference scale: typical pairwise distance among stored portraits.
    # Cap the number of pairs at 50Ã—50 to keep this fast for any n_subjects.
    Nstored = size(X, 2)
    pairs_n = min(50, Nstored)
    pairwise_sample = [norm(X[:, i] .- X[:, j]) for i in 1:pairs_n for j in (i + 1):pairs_n]

    println("median nearest-stored distance for SA samples: $(round(median(nearest_dist), digits=3))")
    println("median pairwise distance among stored portraits: $(round(median(pairwise_sample), digits=3))")
end

# â”€â”€ Task 2a: hard-mask subject-conditional sampling â€” pick a target subject â”€â”€
# A "hard mask" is a boolean vector telling SA which memories are visible at
# sampling time. By masking out everything except one subject's portraits, the
# chain can only attend to in-class memories â€” so at high Î² it locks onto that
# subject's identity.
target_class, n_samples_class, class_mask = let

    # initialize -
    target_class = 2;          # TODO: which subject id to generate (0..n_subjects-1)
    n_samples_class = 10;      # TODO: how many samples to draw for this subject

    # STUDENT TASK: build a Bool vector `class_mask` of length size(X, 2) where
    # entry j is true iff y[j] == target_class. A comprehension is the cleanest
    # one-liner.
    class_mask = [yâ±¼ == target_class for yâ±¼ âˆˆ y];

    println("class_mask leaves ", count(class_mask), " of ", length(class_mask), " stored portraits visible")

    # return -
    target_class, n_samples_class, class_mask;
end;

# â”€â”€ Task 2a: run the hard-masked chains â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STUDENT TASK: Uncomment the block below to run the hard-masked SA chains.
# Builds the warm-start matrix EXPLICITLY (drawn only from in-subject memories
# because we pass `hard_mask = class_mask`) so we can show start vs final per chain.

class_initial, class_samples = let

    # `sa_initial_states` with a hard_mask draws warm-starts ONLY from in-class
    # memories. So column k of class_initial is "in-class memory + small noise".
    class_initial = sa_initial_states(sa_cond, n_samples_class; hard_mask = class_mask);

    # Run the conditional chains. `hard_mask` is forwarded to the dynamics, so
    # at every step attention is restricted to in-class memories. At Î²_cond high
    # the softmax is sharp enough that the chain locks onto a single stored
    # portrait of `target_class`.
    class_samples = stochastic_attention_sample(sa_cond, n_samples_class;
        n_steps = 3000, hard_mask = class_mask, sâ‚’ = class_initial);

    # return -
    class_initial, class_samples;
end;

# â”€â”€ Visualize Task 2a: stored portraits vs hard-mask SA samples â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Top row: actual stored portraits OF the target subject.
# Bottom row: SA samples drawn under the hard mask.
# At high Î² the bottom row should look like specific stored portraits (Hebbian
# recall) â€” i.e. each SA sample matches some top-row image closely.
let

    # initialize -
    n_show = 5  # how many tiles per row (1..n_samples_class)

    # First n_show stored portraits OF the target subject (top row).
    target_idx = findall(==(target_class), y)
    stored_target = X[:, target_idx[1:min(n_show, length(target_idx))]]

    # First n_show SA samples (bottom row).
    sa_target = class_samples[:, 1:n_show]

    # Build the plot list row-by-row: top row first, then bottom row.
    plots = []
    for i âˆˆ 1:n_show
        img = Gray.(clamp.(reshape(stored_target[:, i], 64, 64), 0.0, 1.0))
        push!(plots, heatmap(img; color = :grays, axis = false, ticks = false, colorbar = false,
            title = "S$(target_class) stored $i", titlefontsize = 7, aspect_ratio = :equal))
    end

    for i âˆˆ 1:n_show
        img = Gray.(clamp.(reshape(sa_target[:, i], 64, 64), 0.0, 1.0))
        push!(plots, heatmap(img; color = :grays, axis = false, ticks = false, colorbar = false,
            title = "S$(target_class) SA $i", titlefontsize = 7, aspect_ratio = :equal))
    end

    # Render: 2 rows Ã— n_show columns. Plots fill row-major in the order pushed.
    plot(plots...; layout = (2, n_show), size = (140 * n_show, 160 * 2),
        margin = 0Plots.PlotMeasures.mm)
end

# Pick one of the n_samples_class hard-mask chains to inspect more closely.
class_chain_index = let

    # initialize -
    class_chain_index = 9; # TODO: pick a chain index in 1:n_samples_class
    @assert 1 <= class_chain_index <= n_samples_class # guard against typos
    println("subject $target_class Â· chain $class_chain_index")
    class_chain_index;
end;

# Start vs final for the picked class-conditional chain.
# Left = in-class warm-start memory + noise; right = SA sample after 3000 steps
# under the hard mask. At Î²_cond = 8 the right tile should snap to a stored
# portrait of `target_class`.
let

    # initialize -
    cols   = [class_initial[:, class_chain_index], class_samples[:, class_chain_index]]
    titles = ["start", "final"]
    plots  = [] # storage to hold the plot objects

    # build one tile per (column, title) pair
    for (col, t) âˆˆ zip(cols, titles)
        img = Gray.(clamp.(reshape(col, 64, 64), 0.0, 1.0))
        push!(plots, heatmap(img; color = :grays, axis = false, ticks = false, colorbar = false,
            title = t, titlefontsize = 8, aspect_ratio = :equal))
    end
    plot(plots...; layout = (1, 2), size = (320, 180), margin = 0Plots.PlotMeasures.mm)
end

# â”€â”€ Task 2a evaluation: hard-mask hit-rate sweep over all subjects â”€â”€â”€â”€â”€â”€â”€â”€
# STUDENT TASK: Uncomment the block below to run the hit-rate sweep.
# For each subject c, run the hard-masked sampler with mask = (y .== c), then
# classify each sample by the nearest-class-mean rule. A "hit" is when the
# classifier label matches the target subject c. We expect hit_rate â‰ˆ 1.0 for
# all subjects when Î²_cond is high.

hardmask_table = let

    # initialize -
    df = DataFrame(target = Int[], hits = Int[], total = Int[], hit_rate = Float64[]);

    # Loop over all subjects, draw n_samples_class samples per subject under the
    # subject-specific hard mask, and evaluate with the nearest-centroid classifier.
    for c âˆˆ 0:(n_subjects - 1)
        m = [yâ±¼ == c for yâ±¼ âˆˆ y]                                                  # mask for subject c
        samples = stochastic_attention_sample(sa_cond, n_samples_class;
            n_steps = 3000, hard_mask = m)
        labels = [classify_by_nearest_mean(samples[:, k], class_means) for k âˆˆ 1:n_samples_class]
        h = count(==(c), labels)                                                  # hits = correctly classified samples
        push!(df, (target = c, hits = h, total = n_samples_class, hit_rate = h / n_samples_class))
    end

    # pretty-print the table
    pretty_table(df;
        backend = :text,
        table_format = TextTableFormat(borders = text_table_borders__compact)
    );

    df; # return the DataFrame so it can be used downstream
end;

# â”€â”€ Task 2b: log-multiplicity (soft-bias) sweep â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STUDENT TASK: Uncomment the block below to run the Ï sweep.
# Instead of hard-masking out non-target memories, we ADD log(Ï) to the logits
# of in-class memories. This is a soft preference, not a hard restriction. As
# Ï â†’ âˆž the soft bias dominates and the sampler approaches the hard-mask limit;
# at Ï = 1 the soft bias vanishes and we recover the unconditional sampler.

softbias_table = let

    # Ï: multiplicity ratio. Ï = 1 is the unconditional sampler;
    # Ï â†’ âˆž approaches the hard mask. The in-class logit shift is log(Ï).
    # We use sa_soft (Î²_soft â‰ª Î²_cond) so log(Ï) is competitive with Î²Â·X'Â·s.
    df = DataFrame(Ï = Float64[], log_Ï = Float64[], target = Int[], hits = Int[],
                   total = Int[], hit_rate = Float64[])

    in_class = [yj == target_class for yj in y]

    # Sweep Ï from 1 (no bias) up to 10000 (very strong bias, near hard-mask limit).
    for Ï in [1.0, 10.0, 100.0, 1000.0, 10000.0]
        # soft-bias vector: log(Ï) on in-class memories, 0 on out-of-class.
        sb_vec = Float64.([m ? log(Ï) : 0.0 for m in in_class])

        # Run n_samples_class chains under this soft bias and classify them.
        samples = stochastic_attention_sample(sa_soft, n_samples_class;
            n_steps = 3000, soft_bias = sb_vec)
        labels = [classify_by_nearest_mean(samples[:, k], class_means) for k in 1:n_samples_class]
        h = count(==(target_class), labels)

        push!(df, (Ï = Ï, log_Ï = log(Ï), target = target_class,
                   hits = h, total = n_samples_class, hit_rate = h / n_samples_class))
    end

    # pretty-print the table
    pretty_table(df;
        backend = :text,
        table_format = TextTableFormat(borders = text_table_borders__compact)
    );

    df; # return -
end;

# â”€â”€ Visualize Task 2b: Ï sweep on a fixed warm-start â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Layout: one row per Ï value, columns = | start | chain 1 | chain 2 | chain 3 |.
# All chains in all rows share a SINGLE fixed warm-start (the first stored portrait
# of target_class) so the row-to-row comparison isolates the effect of Ï on the
# trajectory â€” same starting point, only the soft bias changes.
# Uses sa_soft (Î² = Î²_soft) so log(Ï) is in the regime where it can steer the softmax.
let
    Ï_values = [1.0, 10.0, 100.0, 1000.0, 10000.0]
    n_per    = 3                                            # chains per row
    in_class = [yj == target_class for yj in y]

    # Fixed warm-start: first in-class stored portrait, broadcast to all n_per chains.
    j_seed     = findfirst(in_class)                        # column index in X of the first in-class portrait
    seed_state = X[:, j_seed]                               # the actual portrait (a 4096-vector)
    sâ‚’_mat     = repeat(seed_state, 1, n_per)               # (d, n_per) matrix â€” same column n_per times
    start_img  = Gray.(clamp.(reshape(seed_state, 64, 64), 0.0, 1.0))   # pre-rendered start image, reused on every row

    plots = []
    for (i, Ï) in enumerate(Ï_values)
        # Build the soft-bias logit shift: log(Ï) on in-class memories, 0 elsewhere.
        sb_vec  = Float64.([m ? log(Ï) : 0.0 for m in in_class])
        # Run n_per chains from the same warm-start under this soft bias.
        samples = stochastic_attention_sample(sa_soft, n_per;
            n_steps = 3000, soft_bias = sb_vec, sâ‚’ = sâ‚’_mat)

        # column 0: warm-start image. Same picture in every row, but we re-emit it
        # so the plot grid has a consistent left "start" column. Title only on row 1
        # to avoid title clutter; ylabel labels the row with its Ï value.
        push!(plots, heatmap(start_img;
            color = :grays, axis = false, ticks = false, colorbar = false,
            title = i == 1 ? "start" : "", titlefontsize = 9,
            ylabel = "Ï = $(Ï)", ylabelfontsize = 9,
            aspect_ratio = :equal))

        # columns 1..n_per: chain end states for this Ï.
        for k in 1:n_per
            img = Gray.(clamp.(reshape(samples[:, k], 64, 64), 0.0, 1.0))
            push!(plots, heatmap(img;
                color = :grays, axis = false, ticks = false, colorbar = false,
                title = i == 1 ? "chain $k" : "", titlefontsize = 9,
                aspect_ratio = :equal))
        end
    end

    # Render: length(Ï_values) rows Ã— (n_per + 1) columns. Order pushed = row-major.
    plot(plots...; layout = (length(Ï_values), n_per + 1),
        plot_title = "Log-multiplicity sweep at Î²_soft = $(sa_soft.Î²), target_class = $(target_class) (start = stored portrait $(j_seed))",
        plot_titlefontsize = 10,
        size = (170 * (n_per + 1), 160 * length(Ï_values)),
        left_margin = 6Plots.PlotMeasures.mm,
        margin = 1Plots.PlotMeasures.mm)
end

# â”€â”€ DQ3: Î² sweep â€” bridge novelty (low Î²) to recall (high Î²) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STUDENT TASK: Uncomment the block below to run the Î²-sweep.
# We hold the hard mask fixed and sweep only Î². For each Î²:
#   1. build a fresh SA model with this Î²
#   2. draw n_samples_class hard-masked samples (same mask = class_mask)
#   3. compute hit_rate (correctly-classified rate via nearest-class-mean)
#   4. compute median_nearest_stored â€” the regime witness for novelty
# We also stash the lowest-Î² and highest-Î² sample matrices so DQ3 figures
# can show the two ends of the bridge.

Î²_grid, bridge_table, bridge_samples_low, bridge_samples_high = let

    # initialize -
    Î²_grid = [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0, 32.0, 128.0]; # TODO: try a denser or wider grid
    df = DataFrame(Î² = Float64[], hit_rate = Float64[], median_nearest_stored = Float64[])
    bridge_samples_low  = nothing  # filled in on the first iteration
    bridge_samples_high = nothing  # filled in on the last  iteration

    # Sweep Î² with everything else fixed (mask, step_size, noise_scale, n_steps).
    for (i, Î²) âˆˆ enumerate(Î²_grid)
        # Build a per-Î² model. Cheap because building just stores hyperparameters.
        sa_Î² = build(MyStochasticAttentionModel, (
            memories     = X,
            labels       = y,
            Î²            = Î²,
            step_size    = sa_cond.step_size,   # reuse the Task 2a step/noise so Î² is the only knob moving
            noise_scale  = sa_cond.noise_scale,
        ))

        # Hard-masked chains for this Î².
        samples = stochastic_attention_sample(sa_Î², n_samples_class;
            n_steps = 3000, hard_mask = class_mask)

        # hit_rate: nearest-class-mean classifier hits on target_class.
        labels = [classify_by_nearest_mean(samples[:, k], class_means) for k in 1:n_samples_class]
        hit_rate = count(==(target_class), labels) / n_samples_class

        # novelty witness: median over chains of "distance to closest stored portrait".
        nearest = [minimum(norm(samples[:, k] .- X[:, j]) for j in 1:size(X, 2))
                   for k in 1:n_samples_class]

        push!(df, (Î² = Î², hit_rate = hit_rate, median_nearest_stored = median(nearest)))

        # Stash sample matrices at the two ends of the sweep for the DQ3 figures.
        i == 1                && (bridge_samples_low  = samples)
        i == length(Î²_grid)   && (bridge_samples_high = samples)
    end

    # pretty-print the table
    pretty_table(df;
        backend = :text,
        table_format = TextTableFormat(borders = text_table_borders__compact)
    );

    # return -
    Î²_grid, df, bridge_samples_low, bridge_samples_high;
end;

# â”€â”€ DQ3 plot: novelty (median nearest-stored distance) vs Î² â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# As Î² increases, attention sharpens and the chain locks onto stored memories,
# so median_nearest_stored â†’ 0 (perfect recall). As Î² decreases, the chain
# blends memories, so median_nearest_stored grows (novel outputs).
# Hit rate is flat at 1.0 across the sweep (the mask, not Î², enforces conditioning),
# so we only plot the novelty axis.
let
    plot(bridge_table.Î², bridge_table.median_nearest_stored;
        xscale = :log10, marker = :circle, lw = 2,                       # log scale on Î² so the wide range fits
        xlabel = "Î² (log scale)", ylabel = "median nearest-stored",
        label = "subject $(target_class)",
        title = "Novelty vs Î²",
        titlefontsize = 10, labelfontsize = 9, tickfontsize = 8, legendfontsize = 8,
        size = (600, 380),
        left_margin = 6Plots.PlotMeasures.mm,
        bottom_margin = 6Plots.PlotMeasures.mm,
        top_margin = 3Plots.PlotMeasures.mm,
        right_margin = 3Plots.PlotMeasures.mm)
end

# â”€â”€ DQ3 figure (low Î² end): warm-start vs SA sample â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Top row: actual warm-start inputs (in-class memory + small noise).
# Bottom row: the SA samples those starts produced after 3000 steps at Î² = Î²_grid[1].
# At very low Î² attention is nearly uniform â€” the chain blends memories, so the
# bottom row looks like a soft averaged "ghost" of the in-class portraits.
let
    n_show = 5
    Î²_low  = Î²_grid[1]                                                      # smallest Î² in the sweep

    # Build a fresh model at Î²_low using the same step/noise as Task 2a, so
    # Î² is the only thing differing from the high-Î² case.
    sa_Î²   = build(MyStochasticAttentionModel, (
        memories    = X,
        labels      = y,
        Î²           = Î²_low,
        step_size   = sa_cond.step_size,
        noise_scale = sa_cond.noise_scale,
    ))

    # Warm-starts drawn ONLY from in-class memories (because of class_mask).
    s0_warm = sa_initial_states(sa_Î², n_show; hard_mask = class_mask)

    # Run the chains from those warm-starts under the same hard mask.
    sa_low  = stochastic_attention_sample(sa_Î², n_show;
        n_steps = 3000, hard_mask = class_mask, sâ‚’ = s0_warm)

    # Build a 2Ã—n_show grid: row 1 = warm-starts, row 2 = SA samples.
    plots = []
    for i in 1:n_show
        img = Gray.(clamp.(reshape(s0_warm[:, i], 64, 64), 0.0, 1.0))
        push!(plots, heatmap(img; color = :grays, axis = false, ticks = false, colorbar = false,
            title = "sâ‚€ warm #$i", titlefontsize = 7, aspect_ratio = :equal))
    end
    for i in 1:n_show
        img = Gray.(clamp.(reshape(sa_low[:, i], 64, 64), 0.0, 1.0))
        push!(plots, heatmap(img; color = :grays, axis = false, ticks = false, colorbar = false,
            title = "SA Î²=$(Î²_low) #$i", titlefontsize = 7, aspect_ratio = :equal))
    end
    plot(plots...; layout = (2, n_show), size = (140 * n_show, 160 * 2),
        margin = 0Plots.PlotMeasures.mm)
end

# â”€â”€ DQ3 figure (high Î² end): cold-start, 3-row story â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Cold-start = start the chain from pure Gaussian noise (no information about X).
# Restructured as a 3-row story:
#   row 1 â€” the random Gaussian initial state sâ‚€ (looks like static)
#   row 2 â€” the SA sample after 3000 Langevin steps at Î² = Î²_grid[end]
#   row 3 â€” the in-subject stored memory each sample is closest to (the attractor it landed on)
# Each column is one chain: noise â†’ final sample â†’ matched memory. Because Î²
# is large and the mask restricts attention to in-class memories, the chain
# lands on one of the stored target-class portraits â€” Hebbian recall from noise.
let
    Random.seed!(42) # reset the RNG so the figure is reproducible, even after running the previous cells
    n_show = 5
    Î²_high = Î²_grid[end]                                                    # largest Î² in the sweep

    # Fresh model at Î²_high (everything else matches Task 2a).
    sa_Î²   = build(MyStochasticAttentionModel, (
        memories    = X,
        labels      = y,
        Î²           = Î²_high,
        step_size   = sa_cond.step_size,
        noise_scale = sa_cond.noise_scale,
    ))

    # Cold-start: pure Gaussian noise, no peek at X. d = pixel dimension (4096).
    d = size(X, 1)
    s0_cold = randn(d, n_show)

    # Run the chains under the hard mask. Even from noise, sharp attention +
    # mask should pull the chain to an in-class stored memory.
    sa_cold = stochastic_attention_sample(sa_Î², n_show;
        n_steps = 3000, hard_mask = class_mask, sâ‚’ = s0_cold)

    # For each final sample, find the closest stored memory across ALL of X
    # (not just in-class) â€” this is which attractor the chain landed on.
    nearest_idx = [argmin([norm(sa_cold[:, k] .- X[:, j]) for j in 1:size(X, 2)])
                   for k in 1:n_show]

    # Build a 3Ã—n_show grid: row 1 = sâ‚€ noise, row 2 = SA sample, row 3 = nearest stored.
    plots = []
    for i in 1:n_show
        img = Gray.(clamp.(reshape(s0_cold[:, i], 64, 64), 0.0, 1.0))
        push!(plots, heatmap(img; color = :grays, axis = false, ticks = false, colorbar = false,
            title = "sâ‚€ cold #$i", titlefontsize = 7, aspect_ratio = :equal))
    end
    for i in 1:n_show
        img = Gray.(clamp.(reshape(sa_cold[:, i], 64, 64), 0.0, 1.0))
        push!(plots, heatmap(img; color = :grays, axis = false, ticks = false, colorbar = false,
            title = "SA Î²=$(Î²_high) #$i", titlefontsize = 7, aspect_ratio = :equal))
    end
    for i in 1:n_show
        j = nearest_idx[i]
        img = Gray.(clamp.(reshape(X[:, j], 64, 64), 0.0, 1.0))
        push!(plots, heatmap(img; color = :grays, axis = false, ticks = false, colorbar = false,
            title = "nearest X[:, $j]", titlefontsize = 7, aspect_ratio = :equal))
    end
    plot(plots...; layout = (3, n_show), size = (140 * n_show, 160 * 3),
        margin = 0Plots.PlotMeasures.mm)
end

# DQ1 scaffold: defined here in the notebook (not in src/). Once you run this cell,
# `try_sa` is available in the notebook's session scope and can be called from any
# subsequent cell. It is meant for quick parameter exploration â€” for production code
# you would move a function like this into src/Compute.jl and export it.

"""
    try_sa(Î², Î·, Ïƒ; n_show = 8, n_steps = 3000)

Build a fresh unconditional `MyStochasticAttentionModel` over the **centered**
memory bank `X .- mean(X; dims=2)` with the given inverse temperature `Î²`, step
size `Î·`, and noise scale `Ïƒ`, run `n_show` independent Langevin chains for
`n_steps`, and display a 2-row grid:

- **Row 1 (starts):** each chain's warm-start state â€” a randomly chosen stored
  portrait + a small Gaussian kick.
- **Row 2 (finals):** the same chain's state after `n_steps` Langevin updates.

Comparing the two rows column-by-column shows what each chain did with its
starting prompt: lock onto a stored portrait (Regime A), drift to a sharp blend
(Regime B), or collapse to the global mean (Regime C).

### Why centering?
On raw Olivetti pixels, every face has a strong DC component (mean pixel â‰ˆ 0.56)
and pairwise inner products âŸ¨X[:,i], X[:,j]âŸ© are nearly as large as the diagonals
â€–X[:,j]â€–Â². At sharp Î², that pushes the softmax onto a single "winner" column
for *any* state â€” every chain collapses to the same memory regardless of where
it started (attention collapse). Subtracting the per-pixel mean kills the DC
term and lets each chain lock onto its own starting basin at high Î². The mean
is added back at display time so tiles stay on the [0,1] grayscale.

### Arguments
- `Î²::Real`: inverse temperature for the softmax.
- `Î·::Real`: Langevin step size in (0, 1].
- `Ïƒ::Real`: Langevin noise scale (â‰¥ 0).

### Keyword arguments
- `n_show::Int = 8`: number of independent chains.
- `n_steps::Int = 3000`: number of Langevin iterations per chain.

### Returns
A `Plots.Plot` object: 2 Ã— `n_show` grid of 64Ã—64 grayscale heatmaps. Title
shows (Î², Î·, Ïƒ) and `med_near = median_k min_j â€–sâ‚– âˆ’ Xc[:,j]â€–`.
"""
function try_sa(Î²::Real, Î·::Real, Ïƒ::Real; n_show::Int = 8, n_steps::Int = 3000)

    # â”€â”€ Step 1: center the memory bank â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # We compute the per-pixel mean across all 100 stored faces. Subtracting it
    # gives Xc, a memory bank where every column has mean 0. This kills the DC
    # component that otherwise dominates inner products on Olivetti and causes
    # "attention collapse" at sharp Î². We keep Î¼ around to add back later when
    # we display tiles, so the rendered images still live on the [0,1] grayscale.
    Î¼  = mean(X; dims = 2)              # shape (d, 1) â€” per-pixel mean over the 100 columns of X
    Xc = X .- Î¼                         # shape (d, M) â€” centered memory bank used by the dynamics
    d, M = size(Xc)                     # d = 4096 (= 64Ã—64 pixels), M = 100 (10 subjects Ã— 10 photos)

    # â”€â”€ Step 2: build a fresh SA model on the centered bank â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # `build` is the project's factory for MyStochasticAttentionModel. We pass in
    # the centered memories Xc (NOT raw X), the labels y, and the three knobs:
    #   Î² = inverse temperature (sharpness of the softmax)
    #   Î· = step size (how much of the modern-Hopfield update to apply per step)
    #   Ïƒ = noise scale (Gaussian noise added to each Langevin step)
    sa = build(MyStochasticAttentionModel, (
        memories    = Xc,
        labels      = y,
        Î²           = Float64(Î²),
        step_size   = Float64(Î·),
        noise_scale = Float64(Ïƒ),
    ))

    # â”€â”€ Step 3: build the warm-start matrix S0 EXPLICITLY â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Normally `stochastic_attention_sample` would internally call
    # `sa_initial_states` to pick warm-start states. We replicate that recipe
    # here (pick a random stored column per chain, add a 5%-amplitude Gaussian
    # kick) so we can *display* the starts in row 1 of the figure. Without this,
    # the chain inputs would be invisible to students reading the plot.
    rng = Random.GLOBAL_RNG
    S0  = zeros(Float64, d, n_show)     # one column per chain â€” d-dimensional state
    for k in 1:n_show
        j = rand(rng, 1:M)              # pick a random stored portrait index for chain k
        S0[:, k] = Xc[:, j] .+ 0.05 .* randn(rng, d)  # centered portrait + small noise
    end

    # â”€â”€ Step 4: run the Langevin chains â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # We pass S0 explicitly via `sâ‚’ = S0` so the displayed starts match what the
    # chains actually see. Each chain runs for `n_steps` Langevin updates of the
    # form  s_{t+1} = (1âˆ’Î·)Â·s_t + Î·Â·XÂ·softmax(Î²Â·X^TÂ·s_t) + ÏƒÂ·Î¾_t.
    samples = stochastic_attention_sample(sa, n_show; n_steps = n_steps, sâ‚’ = S0)

    # â”€â”€ Step 5: compute the regime witness `med_near` â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # For each chain, find the nearest stored memory in centered space and
    # measure that distance. The median across chains is our scalar regime
    # diagnostic â€” it goes 0 (chains landed ON stored portraits, Regime A)
    # â†’ small (chains landed near a blend of a few memories, Regime B)
    # â†’ â‰ˆâ€–Xc[:,j]â€– (chains landed at the centered mean = origin, Regime C).
    nearest = [minimum(norm(samples[:, k] .- Xc[:, j]) for j in 1:M)
               for k in 1:n_show]
    med_near = median(nearest)

    # â”€â”€ Step 6: build the 2-row display â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Each chain becomes a column. Top row = where the chain started, bottom row
    # = where it ended. Reading topâ†’bottom in any column tells you what the
    # chain DID with its starting prompt at this Î².
    plots = []

    # Row 1: chain starts. We add Î¼ back so the centered residual reads as a face.
    # `clamp.(..., 0, 1)` prevents the small noise kick from pushing pixels off
    # the grayscale range â€” without it the renderer would saturate and look ugly.
    for i in 1:n_show
        img = Gray.(clamp.(reshape(S0[:, i] .+ vec(Î¼), 64, 64), 0.0, 1.0))
        push!(plots, heatmap(img; color = :grays, axis = false, ticks = false,
            colorbar = false, title = "start #$i", titlefontsize = 7,
            aspect_ratio = :equal))
    end

    # Row 2: chain finals â€” same column k as row 1, so column k tells one chain's story.
    for i in 1:n_show
        img = Gray.(clamp.(reshape(samples[:, i] .+ vec(Î¼), 64, 64), 0.0, 1.0))
        push!(plots, heatmap(img; color = :grays, axis = false, ticks = false,
            colorbar = false, title = "final #$i", titlefontsize = 7,
            aspect_ratio = :equal))
    end

    # 2 rows Ã— n_show cols. Title prints the knobs plus med_near so the regime is
    # legible from the figure title alone, even before the eye reads the tiles.
    plot(plots...; layout = (2, n_show), size = (140 * n_show, 320),
        plot_title = "Î²=$(Î²), Î·=$(Î·), Ïƒ=$(Ïƒ)   med_near=$(round(med_near; digits=2))",
        plot_titlefontsize = 9,
        margin = 0Plots.PlotMeasures.mm)
end;

# â”€â”€ DQ1 â€” Regime A (replication) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# At Î² = 64 the softmax Î²Â·Xc^TÂ·s is so sharp that it concentrates almost all
# its mass on a single column of Xc. The update s â† XcÂ·softmax(...) therefore
# returns essentially that one stored memory (in centered coordinates). The
# Ïƒ = 0.10 noise is small relative to inter-memory distances in centered space,
# so chains can't escape the basin they land in.
#
# What you should see:
#   â€¢ Row 1 (starts):  8 different noisy stored portraits.
#   â€¢ Row 2 (finals):  8 sharp, contrasty stored portraits.
#   â€¢ Most columns have top â‰ˆ bottom (chain locked onto its starting basin),
#     though 1â€“2 chains may drift to a different in-class memory because
#     centered face-face cross-correlations aren't perfectly zero.
#   â€¢ Title shows  med_near â‰ˆ 0.0  â€” every chain ended ON a stored memory.
try_sa(64.0, 1.0, 0.10) # TODO: uncomment to run

# â”€â”€ DQ1 â€” Regime B (transition / sharp blend) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# At Î² = 0.1 the softmax is partially soft: it concentrates on a few centered
# columns rather than one, so XcÂ·softmax(...) is a *sharp blend* of those
# memories â€” face-shaped but no longer a verbatim stored portrait. Note: the
# transition band shifts with centering (centered â€–Xc[:,j]â€–Â² is ~20Ã— smaller
# than the raw value), so Î² = 0.1 is the right "mid" knob here even though it
# would land in the ghost-mean regime on the raw memory bank.
#
# What you should see:
#   â€¢ Row 1 (starts):  8 different noisy stored portraits.
#   â€¢ Row 2 (finals):  blurry, face-shaped tiles that don't match their starts.
#   â€¢ Tiles in row 2 look like soft mixtures of a couple of in-class faces,
#     not a single sharp portrait and not the ghost average.
#   â€¢ Title shows  med_near â‰ˆ 1â€“2  â€” chains landed near, but not on, stored memories.
try_sa(0.1, 1.0, 0.10) # TODO: uncomment to run

# â”€â”€ DQ1 â€” Regime C (full blend / ghost mean) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# At Î² = 0.005 the softmax weights are nearly uniform over all 100 columns of
# Xc. The update XcÂ·softmax(...) â‰ˆ mean of Xc columns â‰ˆ 0 (because Xc is
# centered). After we add Î¼ back at display time, the rendered tile is the
# global-mean face â€” the average of all 100 stored portraits.
#
# What you should see:
#   â€¢ Row 1 (starts):  8 different noisy stored portraits.
#   â€¢ Row 2 (finals):  8 nearly-identical, smooth, low-contrast "ghost average" faces.
#   â€¢ The chain has forgotten its starting basin entirely; identity is washed out.
#   â€¢ Title shows  med_near â‰ˆ 4.5  â€” chains landed far from any stored memory,
#     near the centered origin.
try_sa(0.005, 1.0, 0.10) # TODO: uncomment to run

#= Put your answer to DQ1 here. =#

did_I_answer_DQ1 = true; # set to `true` after answering DQ1

# DQ2 scaffold: print the hard-mask hit rate for the target subject alongside the
# log-multiplicity sweep, so the two regimes sit next to each other.
let
    hard_target = filter(:target => ==(target_class), hardmask_table)
    println("Hard mask, subject $(target_class):")
    pretty_table(hard_target;
        backend = :text,
        table_format = TextTableFormat(borders = text_table_borders__compact));
end

let
    println("Log-multiplicity bias, subject $(target_class), Ï sweep:")
    pretty_table(softbias_table;
        backend = :text,
        table_format = TextTableFormat(borders = text_table_borders__compact))
end

#= Put your answer to DQ2 here. =#

did_I_answer_DQ2 = true; # set to `true` after answering DQ2

#= Put your answer to DQ3 here. =#

did_I_answer_DQ3 = true; # set to `true` after answering DQ3

let
    @testset verbose = true "CHEME 5820 Practicum S2026" begin

        @testset "Setup, Data, and Prerequisites" begin
            @test _DID_INCLUDE_FILE_GET_CALLED == true
            @test isnothing(n_subjects) == false
            @test isnothing(images_per_subject) == false
            @test isnothing(Î²_gen) == false
            @test isnothing(Î²_cond) == false
            @test isnothing(Î²_soft) == false
            @test Î²_gen > 0
            @test Î²_cond > 0
            @test Î²_soft > 0
            @test size(X, 1) == 4096
            @test size(X, 2) == n_subjects * images_per_subject
            @test length(y) == size(X, 2)
            @test sort(unique(y)) == collect(0:(n_subjects - 1))
            @test isnothing(sa_gen) == false
            @test sa_gen.Î² == Î²_gen
            @test isnothing(sa_cond) == false
            @test sa_cond.Î² == Î²_cond
            @test isnothing(sa_soft) == false
            @test sa_soft.Î² == Î²_soft
            @test sa_gen.step_size > 0
            @test sa_gen.noise_scale >= 0
            @test isnothing(class_means) == false
            @test length(class_means) == n_subjects
        end

        @testset "Task 1: Stochastic Attention sampling on faces" begin
            @test size(unmasked_initial) == (4096, n_samples_unmasked)
            @test size(unmasked_samples) == (4096, n_samples_unmasked)
            @test all(isfinite, unmasked_initial)
            @test all(isfinite, unmasked_samples)
            @test isnothing(chain_index) == false
            @test 1 <= chain_index <= n_samples_unmasked
            @test did_I_answer_DQ1 == true
        end

        @testset "Task 2: Masked SA = subject-conditional generation" begin
            @test 0 <= target_class <= n_subjects - 1
            @test length(class_mask) == size(X, 2)
            @test count(class_mask) == count(==(target_class), y)
            @test size(class_initial) == (4096, n_samples_class)
            @test all(isfinite, class_initial)
            @test size(class_samples) == (4096, n_samples_class)
            @test isnothing(class_chain_index) == false
            @test 1 <= class_chain_index <= n_samples_class
            @test mean(hardmask_table.hit_rate) > 0.7
            # Calibration gap: at moderate Ï (log_Ï â‰¤ 5, i.e. Ï â‰² 150) the soft-bias
            # under-performs the hard mask. At very large Ï the gap closes, so we only
            # test the regime where the gap is visible.
            @test mean(hardmask_table.hit_rate) > maximum(filter(:log_Ï => <=(5.0), softbias_table).hit_rate)
            @test did_I_answer_DQ2 == true
        end

        @testset "Task 3: Î²-sweep recovers the Hebbian/Hopfield bridge" begin
            @test length(Î²_grid) >= 4
            @test issorted(Î²_grid)
            @test all(>(0), Î²_grid)
            @test nrow(bridge_table) == length(Î²_grid)
            @test all(isfinite, bridge_table.hit_rate)
            @test all(isfinite, bridge_table.median_nearest_stored)
            @test mean(bridge_table.hit_rate) > 0.7
            # bridge claim: novelty (distance to nearest stored) at the
            # smallest Î² should exceed novelty at the largest Î²: the
            # Hebbian-limit collapse onto stored portraits.
            @test bridge_table.median_nearest_stored[1] > bridge_table.median_nearest_stored[end]
            @test size(bridge_samples_low)  == (4096, n_samples_class)
            @test size(bridge_samples_high) == (4096, n_samples_class)
            @test did_I_answer_DQ3 == true
        end
    end
end;


