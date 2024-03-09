using GLMakie, LinearAlgebra, ProgressMeter, Distributions

# Here, we reproduce the example model of [Crooks' 1999 paper, "Entropy production fluctuation theorem and the nonequilibrium work relation for free energy differences"](https://arxiv.org/pdf/cond-mat/9901352.pdf).
# The model, quoting directly from Crooks, is:
#
# > A single particle occupies [one of 32 possible] positions in a one-dimensional
# > box with periodic boundaries, and is coupled to a heat bath of
# > temperature $T = 5$. The energy surface, $E(x)$, is [given by $\operatorname{abs}(x - (x_0 + 1/2))$, with $x_0$ some particular site].
# > At each discrete time step the particle attempts to move
# > left, right, or stay put with equal probability. The move is accepted
# > with the standard Metropolis acceptance probability [$= \exp(-\Delta E/T)$]. Every
# > eight time steps, the energy surface moves right one position. Thus
# > the system is driven away from equilibrium, and eventually settles
# > into a time symmetric nonequilibrium steady state. 
#
# First, we implement the periodic motion of the energy surface by
# defining the center position $x_0$ as a function of time:

box_size = 32
time_between_shifts = 8

## +0.5 because x0 is inbetween sites
x0(t) = mod1(div(t,time_between_shifts),box_size)
nothing#hide

# Next, we define the energy surface at $t=0$:

initial_energy_surface = [0:15 ; 15:-1:0]
nothing#hide

# Now, the energy function $E(x,t)$ simply indexes the $t=0$ at the appropriate offset:

energy_function(x,t) = initial_energy_surface[mod1(x - x0(t),box_size)]
nothing#hide

# With the energy surface established, we can write the standard Metropolis algorithm for
# taking one step:

metropolis(ΔE,kT) = exp(-ΔE/kT)

function make_step(x0,t;kT = 5)
  ## Move left, right, or stay (equal probability)
  step = rand([-1,0,1])
  if step == 0
    return x0, 0. # If staying, no need to check anything
  end

  ## Periodic boundaries
  new_x = mod1(x0+step,box_size)

  ## Move with probability exp(-ΔE/kT), i.e. exponentially
  ## small in the amount of heat imported from the bath
  e_stay, e_move = energy_function.([x0,new_x],t)
  heat_imported_from_bath_by_step = e_move - e_stay
  if rand() < metropolis(heat_imported_from_bath_by_step,kT)
    new_x, heat_imported_from_bath_by_step
  else
    x0, 0.
  end
end
nothing#hide

# We additionally return the amount of heat the bath gives to the system during the step.
#
# Before doing any sampling, we need to "burn in" to avoid biasing
# the distribution with our choice of initial state.
# After the burn in period, we begin accumulating statistics.

function take_samples(f;n_burn = 10_000,n_sample = 1_000_000)
  ## Burn in
  for j = 1:n_burn
    f(false) ## Flag that this is the burn in period; don't take stats
  end

  ## Sample
  for j = 1:n_sample
    f(true) ## Flag that statistics should be taken
  end
end
nothing#hide

# First, we calculate the nonequilibrium steady state distribution
# by letting the energy surface move in time and recording a histogram
# of the position relative to $x_0$:

hist_ss = zeros(Int64,32)
x_cur = 0
t_cur = 0

take_samples() do take_stats
  global x_cur, t_cur
  ## Make a step, and increment the time
  x_cur, _ = make_step(x_cur,t_cur)
  t_cur = t_cur + 1
  if take_stats
    ## Record the position in the frame of x0
    hist_ss[mod1(x_cur - x0(t_cur),box_size)] += 1
  end
end

# Next, we calculate the equilibrium distribution when the
# energy surface is frozen in time by fixing $t=0$:

hist_eq = zeros(Int64,32)
x_cur = 0

take_samples() do take_stats
  global x_cur
  ## Frozen at t = 0
  x_cur, _ = make_step(x_cur,0)

  if take_stats
    hist_eq[mod1(x_cur - x0(0),box_size)] += 1
  end
end

# Finally, plot the results:

## Rotate the energy minimum to the center
Ix = sortperm(mod1.(-15:16,box_size))
in_frame(x) = x[Ix]

## Compute probability from histogram counts
p_ss = hist_ss ./ sum(hist_ss)
p_eq = hist_eq ./ sum(hist_eq)

f = Figure(); ax = Axis(f[1,1])
plot!(ax,in_frame(p_ss),color = :black)
lines!(ax,in_frame(p_ss),color = :black)

lines!(ax,in_frame(p_eq),color = :black)
scatter!(ax,in_frame(p_eq),color = :white, strokewidth = 3,strokecolor = :black)

## Rescale energy to fit in plot
plot!(ax,0.1/15 * in_frame(initial_energy_surface),marker = '-',color = :black,markersize = 25)

f

# The non-equilibrium steady state (black dots) resembles a fluid sloshing to the side
# as its container (dashes) is moved at a fixed speed to the right.
# If the container doesn't move, the equilibrium distribution (white dots) results.
#
# Imagine that the system starts from equilibrium (non-moving), and that the motion is
# suddenly turned on.
# We can measure the time course of the evolution from equilibrium to the transient
# to the final non-equilibrium steady state by burning in to the equilibrium state,
# and only then turning on the movement of the energy surface.

n_sample = 2048
hist_resolved = zeros(Int64,32,n_sample)

n_trial = 10_000

## Additional statitics for later
t_save = [128,256,512,1024,2048]
works_experienced = zeros(Int64,n_trial,length(t_save))
heats_experienced = zeros(Int64,n_trial,length(t_save))

prog = Progress(n_trial)#hide
for trial = 1:n_trial
  ## Make burn in easier by sampling from known equilibrium
  x = rand(Categorical(p_eq))

  t = 0
  total_W = 0
  total_Q = 0

  take_samples(;n_burn = 10,n_sample) do take_stats
    ## Account for work of energy surface at current state
    ## *before* applying heat (see Crooks 1998 eqns 5,6,7)
    W = energy_function(x,t) - energy_function(x,t-1)

    ## Burn in to equilibrium at first
    x, Q = make_step(x,t)

    if take_stats
      hist_resolved[mod1(x - x0(t),box_size),1 + t] += 1

      ## Only enable time evolution after burn in
      t = t + 1
      total_W += W
      total_Q += Q

      ix = findfirst(x -> x == t,t_save)
      if !isnothing(ix)
        works_experienced[trial,ix] = total_W
        heats_experienced[trial,ix] = total_Q
      end
    end
  end
  next!(prog)#hide
end
finish!(prog)#hide

heatmap(1:32,0:200,hist_resolved[Ix,1:201], axis = (;title = "Evolution to steady state", ylabel = "Time", xlabel = "x", yticks = 0:8:200,xticks = 4:4:32),interpolate=true)

# The horizontal banding is due to the discrete movement of the energy surface every eight
# timesteps.
# We can see the same effect in the movie:

p_of_t = Observable(Float64.(hist_resolved[:,1]))

f_anim = Figure()
ax = Axis(f_anim[1,1])

## Reference steady state
plot!(ax,in_frame(p_ss),color = :black)
lines!(ax,in_frame(p_ss),color = :black)

## Reference equilibrium
lines!(ax,in_frame(p_eq),color = :black)
scatter!(ax,in_frame(p_eq),color = :white, strokewidth = 3,strokecolor = :black)

## Time-evolving distribution
lines!(ax, p_of_t, color = :blue, linewidth = 3)

ylims!(ax,0,0.1)

record(f_anim, "anim.mp4", [hist_resolved[:,i] for i = 1:n_sample]; framerate = 30) do h
  p_of_t[] .= in_frame(h ./ n_trial)
  notify(p_of_t)
end
nothing#hide

#md # ![](anim.mp4)

# Using the time-resolved data, we can decompose the energy changes into work (the 
# change in energy due to moving the energy surface) and heat (all other changes of energy
# during the periods of partial equilibriation while the energy surface is stationary).
#
# The typical (over many realizations) work done on the system by the movement of
# the energy surface, during one of the shifts at time $t = 8n$, is just the expected
# value of the change in height of the surface over the distribution of states at time $t$:

timestamps = 0:1:(n_sample - 1)
energy_surface = [energy_function(x,t) for x = 1:32,t = timestamps]

## Only nonzero at t = 8n
energy_shifts = energy_surface[:,2:end] .- energy_surface[:,1:end-1]

## ∑ᵢpᵢΔEᵢ, with pᵢ taken immediately before the energy change
typical_work_done_on_system = sum((hist_resolved ./ n_trial)[:,1:end-1] .* energy_shifts,dims = 1)[:]

f = Figure()
cycle_length = box_size * time_between_shifts
ax = Axis(f[1,1]; xlabel = "Time", ylabel = "Work", xticks = 0:cycle_length:n_sample)

## Typical work at all time steps (7/8 are zero)
lines!(ax,typical_work_done_on_system,color = :blue,linewidth = 0.2)

## Nonzero points only
ix_nonzero_work = time_between_shifts:time_between_shifts:(n_sample-1)
scatter!(ax,ix_nonzero_work,typical_work_done_on_system[ix_nonzero_work], color = :blue)
xlims!(ax,0,512)

f#hide

# Similarly, the heat accounts for all other changes in energy (even during the timesteps
# when the energy surface moves), so we can find it as $Q = \Delta E - W$.

typical_system_energy = sum((hist_resolved ./ n_trial) .* energy_surface,dims = 1)[:]
change_in_typical_system_energy = typical_system_energy[2:end] .- typical_system_energy[1:end-1]

## First law of thermodynamics
typical_heat_imported_from_bath = change_in_typical_system_energy .- typical_work_done_on_system
nothing#hide

# With both quantities in hand, we can now plot the full work-heat resolved
# evolution of the energy.
# Here, the red x's represent additional "virtual" timesteps inbetween the timesteps
# used to simulate the heat bath.
# They occur whenever the energy surface moves, and represent the work-only part of their
# respective time step.
# All blue lines represent *only* the exchange of heat with the bath, and no external work.

work_xs = zeros(Float64,3,length(ix_nonzero_work))
work_ys = zeros(Float64,3,length(ix_nonzero_work))
work_xs[3,:] .= NaN
work_ys[3,:] .= NaN

work_xs[1,:] .= timestamps[ix_nonzero_work]
work_xs[2,:] .= timestamps[ix_nonzero_work]

work_ys[1,:] .= typical_system_energy[ix_nonzero_work]
work_ys[2,:] .= typical_system_energy[ix_nonzero_work] .+ typical_work_done_on_system[ix_nonzero_work]

heat_xs = zeros(Float64,3,length(typical_heat_imported_from_bath))
heat_ys = zeros(Float64,3,length(typical_heat_imported_from_bath))
heat_ys[3,:] .= NaN

heat_xs[1,:] .= timestamps[1:end-1]
heat_xs[2,:] .= timestamps[2:end]

heat_ys[1,:] .= typical_system_energy[2:end] .- typical_heat_imported_from_bath
heat_ys[2,:] .= typical_system_energy[2:end] 

f = Figure()
ax = Axis(f[1,1]; xlabel = "Time", ylabel = "Energy", xticks = 0:cycle_length:n_sample)
lines!(ax,work_xs[:],work_ys[:],color = :red)
scatter!(ax,work_xs[2,:],work_ys[2,:],color = :red,marker = 'x')
lines!(ax,heat_xs[:],heat_ys[:],color = :blue)
xlims!(ax,0,512)
f#hide
