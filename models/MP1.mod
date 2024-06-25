# states
set S;

# action set for players
set A1{S};
set A2{S};


# transition probabilities
param P{st in S, s in S, a2 in A2[s]};

# gamma
param gamma{S};

# alphas
param F_inv_alph1;
param F_inv_alph2;

# mus
param mu1{s in S, a1 in A1[s], a2 in A2[s]};
param mu2{s in S, a1 in A1[s], a2 in A2[s]};

# sigmas
param sigma1{s in S, a1 in A1[s], a2 in A2[s], s_ in S, a1_ in A1[s_], a2_ in A2[s_]};
param sigma2{s in S, a1 in A1[s], a2 in A2[s], s_ in S, a1_ in A1[s_], a2_ in A2[s_]};

# discounting factor
param beta = 0.75;

# strategy for players
var f{s in S, a1 in A1[s]} >= 0, <= 1;
var x{s in S, a2 in A2[s]} >= 0, <= 1;

# vs
var v1{s in S, a1 in A1[s], a2 in A2[s]};
var v2{s in S, a1 in A1[s], a2 in A2[s]};

var lambda1{s in S};
var lambda2{s in S};

# Constraints
subject to D1_constraint{s in S, a1 in A1[s]}:
    lambda1[s] <= sum{a2 in A2[s]} (
        x[s, a2] * (
            sum{s_ in S, a1_ in A1[s_], a2_ in A2[s_]} 
            (sigma1[s, a1, a2, s_, a1_, a2_] * v1[s_, a1_, a2_]) * F_inv_alph1) 
        + x[s, a2] * mu1[s, a1, a2]
    );

subject to D2_constraint{s in S, a2 in A2[s]}:
    sum{s_ in S} (
        lambda2[s_] * ((if s = s_ then 1 else 0) - beta * P[s_, s, a2])
    ) <= sum{a1 in A1[s]} (
        f[s, a1] * (
            sum{s_ in S, a1_ in A1[s_], a2_ in A2[s_]} 
            (sigma2[s, a1, a2, s_, a1_, a2_] * v2[s_, a1_, a2_]) * F_inv_alph2) 
        + f[s, a1] * mu2[s, a1, a2]
    );

subject to Norm_v1_constraint:
    sqrt(sum{s in S, a1 in A1[s], a2 in A2[s]} (v1[s, a1, a2]^2)) <= 1;

subject to Norm_v2_constraint:
    sqrt(sum{s in S, a1 in A1[s], a2 in A2[s]} (v2[s, a1, a2]^2)) <= 1;

subject to FS{s in S}:
    sum{a1 in A1[s]} (f[s, a1]) = 1;

subject to QB{s_ in S}:
    sum{s in S, a2 in A2[s]} (x[s, a2] * ((if s = s_ then 1 else 0) - beta * P[s_, s, a2])) = (1 - beta) * gamma[s_];

# Objective Constraint
maximize objective_constraint:
    (
        (-F_inv_alph1 * sqrt(
            sum{s in S, a1 in A1[s], a2 in A2[s]} 
            (sum {s_ in S , a1_ in A1[s_],a2_ in A2[s_]}
                (sigma1[s, a1, a2, s_, a1_, a2_] * f[s_, a1_] * x[s_, a2_])
            )^2) 
            - sum{s in S, a1 in A1[s], a2 in A2[s]} (f[s, a1] * x[s, a2] * mu1[s, a1, a2])
            + sum{s in S} lambda1[s]
        ) +
        (-F_inv_alph2 * sqrt(
            sum{s in S, a1 in A1[s], a2 in A2[s]} 
            (sum {s_ in S , a1_ in A1[s_],a2_ in A2[s_]}
                (sigma2[s, a1, a2, s_, a1_, a2_] * f[s_, a1_] * x[s_, a2_])
            )^2) 
            - sum{s in S, a1 in A1[s], a2 in A2[s]} (f[s, a1] * x[s, a2] * mu2[s, a1, a2])
            + sum{s in S} (lambda2[s] * (1 - beta) * gamma[s])
        )    
    );


