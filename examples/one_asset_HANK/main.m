% Turn off the warning message from auto diff
% You should read the warning once, but turn it off after reading it.
warning('off','AutoDiff:maxmin');

ReduceDistribution = 1;  % 1 for state space reduction 0 for not
reduceV = 1;             % 1 for value function reduction 0 for not
ReduceDist_hors = [25,30,35,40];     % Dimensionality of the Krylov subspace
DisplayLev = 1;          % Determines verbosity of steady state calculation
check_consistency = 1;   % Runs Step 6: Internal consistency check

%% Step 0: Set parameters

set_parameters;

n_v = I*J + 1;          % number of choice/jump variables (value function + inflation)
n_g = I*J;              % number of state variables (distribution + monetary policy)
n_p = 5;                % number of static relations (includes observables)
n_shocks = 1;           % only monetary policy shock is considered
nEErrors = n_v;
nVars = n_v + n_g + n_p;

%% Step 1: Solve for the steady state

fprintf('Computing steady state...\n');
t0 = tic;
compute_steady_state;
fprintf('Time to compute steady state: %2.4f seconds\n\n\n',toc(t0));

%% Step 2: Linearize Model equations

fprintf('Taking derivatives of equilibrium conditions...\n');
t0 = tic;

% Prepare automatic differentiation
vars = zeros(nVars + nVars + nEErrors + n_shocks,1);
vars = myAD(vars);

% Evaluate derivatives
equilibrium_conditions;

% Extract derivative values
derivs = getderivs(v_residual);

t_derivs = toc(t0);
fprintf('Time to compute derivatives: %2.4f seconds\n\n\n',t_derivs);
if t_derivs> 1
    warning('If you do not compile mex/C files for the automatic differentiation, matrix vector multiplication will be slow');
    disp('Press any key to continue...');
    pause();
end

%% Step 3: Solve out Static Constraints or Reduce the Model

g1 = -derivs(:,1:nVars);
g0 = derivs(:,nVars+1:2*nVars);
pi = -derivs(:,2*nVars+1:2*nVars+nEErrors);
psi = -derivs(:,2*nVars+nEErrors+1:2*nVars+nEErrors+n_shocks);
constant = sparse(nVars,1);

% Solve out static constraints
fprintf('Solving Out Static Constraints ...\n');
[state_red,inv_state_red,g0,g1,constant,pi,psi] = clean_G0_sparse(g0,g1,constant,pi,psi);
n_g_red = n_g;

% Create identity matrix for code reuse below
from_spline = speye(n_g_red + n_v);
to_spline = speye(n_g_red + n_v);
n_splined = n_v;

%% Step 4: Solve Linear Systems

t0 = tic;
fprintf('Solving linear system...\n');

% Note parts of schur_solver will be swapped out in the
%    near future, so it might have a different interface. Since the codebase is
%    new, there will be interative updates to simplify interface and syntax.
%    Underlying math will stay the same, but interfact may change with updates,
%    so one should note the release number of the codebase one is using.
[G1, ~, impact, eu, F] = schur_solver(g0,g1,c,psi,pi,1,1,1,n_splined);
fprintf('...Done!\n')
fprintf('Existence and uniqueness? %2.0f and %2.0f\n',eu);
fprintf('Time to solve linear system: %2.4f seconds\n\n\n',toc(t0));

%% Step 5: Simulate Impulse Response Functions

fprintf('Simulating Model...\n');
t0 = tic;

T = 100;
N = 400;
dt = T/N;
vAggregateShock	= zeros(1,N);
vAggregateShock(1) = -1/sqrt(dt);
trans_mat = inv_state_red*from_spline;
[simulated,vTime] = simulate(G1,impact,T,N,vAggregateShock,'implicit',trans_mat,[n_v,n_v+n_g:n_v+n_g+5]);

fprintf('...Done!\n');
fprintf('Time to simulate model: %2.4f seconds\n\n\n',toc(t0));

% Define variables to be plotted below
inflation = simulated(1,:)';
monetary_shock = simulated(2,:)';
consumption = (simulated(4,:)')/vars_SS(n_v+n_g+3);
Y = (simulated(6,:)')/vars_SS(n_v+n_g+4);
lab_sup = (simulated(4,:)')/vars_SS(n_v+n_g+2);
wage = simulated(3,:)'/vars_SS(n_v+n_g+1);

%% Step 7: IRFs

fig = figure('units','normalized','outerposition',[0 0 1 1]);
line_style = '-';
color = 'blue';
plot_IRFs;

%% Step 3: Reduce the model

ReduceDist_hor = ReduceDist_hors(1);
fprintf('\n\nReduction horizon %d\n',ReduceDist_hor);
g1 = -derivs(:,1:nVars);
g0 = derivs(:,nVars+1:2*nVars);
pi = -derivs(:,2*nVars+1:2*nVars+nEErrors);
psi = -derivs(:,2*nVars+nEErrors+1:2*nVars+nEErrors+n_shocks);
constant = sparse(nVars,1);

% State Variables
% Reduce model
fprintf('Reducing distribution ...\n');
[state_red,inv_state_red,n_g_red] = krylov_reduction(g0,g1,n_v,n_g,ReduceDist_hor);
[g1,psi,pi,constant,g0] = change_basis(state_red,inv_state_red,g1,psi,pi,constant,g0);
fprintf('Reduced to %d from %d.\n',n_g_red,n_g);

% Jump Variables
% Reduce dimensionality of value function using splines
n_knots = 15;
c_power = 1;
x = a';
n_post = size(z,2);
n_prior = 1;

% Create knot points for spline (the knot points are not uniformly spaced)
knots = linspace(amin,amax,n_knots-1)';
knots = (amax-amin)/(2^c_power-1)*((knots-amin)/(amax-amin)+1).^c_power+ amin-(amax-amin)/(2^c_power-1);

% Function calls to create basis reduction
[from_spline, to_spline] = oneDquad_spline(x,knots);
[from_spline, to_spline] = extend_to_nd(from_spline,to_spline,n_prior,n_post);
from_spline(end+1,end+1) = 1;
to_spline(end+1,end+1) = 1;
n_splined = size(from_spline,2);
[from_spline, to_spline] = projection_for_subset(from_spline,to_spline,0,n_g_red);

% Reduce the decision vector
[g1,psi,~,constant,g0] = change_basis(to_spline,from_spline,g1,psi,pi,constant,g0);
pi = to_spline * pi * from_spline(1:n_v,1:n_splined);

%% Step 4: Solve linear systems

t0 = tic;
fprintf('Solving linear system...\n');

% Note parts of schur_solver will be swapped out in the
%    near future, so it might have a different interface. Since the codebase is
%    new, there will be interative updates to simplify interface and syntax.
%    Underlying math will stay the same, but interfact may change with updates,
%    so one should note the release number of the codebase one is using.
[G1, ~, impact, eu, F] = schur_solver(g0,g1,c,psi,pi,1,1,1,n_splined);
fprintf('...Done!\n')
fprintf('Existence and uniqueness? %2.0f and %2.0f\n',eu);
fprintf('Time to solve linear system: %2.4f seconds\n\n\n',toc(t0));

%% Step 5: Simulate IRFs

fprintf('Simulating Model...\n');
t0 = tic;

T = 100;
N = 400;
dt = T/N;
vAggregateShock	= zeros(1,N);
vAggregateShock(1) = -1/sqrt(dt);
trans_mat = inv_state_red*from_spline;
[simulated,vTime] = simulate(G1,impact,T,N,vAggregateShock,'implicit',trans_mat,[n_v,n_v+n_g:n_v+n_g+5]);

fprintf('...Done!\n');
fprintf('Time to simulate model: %2.4f seconds\n\n\n',toc(t0));

% Define variables to be plotted below
inflation = simulated(1,:)';
monetary_shock = simulated(2,:)';
consumption = (simulated(4,:)')/vars_SS(n_v+n_g+3);
Y = (simulated(6,:)')/vars_SS(n_v+n_g+4);
lab_sup = (simulated(4,:)')/vars_SS(n_v+n_g+2);
wage = simulated(3,:)'/vars_SS(n_v+n_g+1);

%% Step 6: Internal consistency check

g1 = -derivs(:,1:nVars);
psi = -derivs(:,2*nVars+nEErrors+1:2*nVars+nEErrors+n_shocks);
from_red = inv_state_red * from_spline;
to_red = to_spline * state_red;
[epsilon] = internal_consistency_check(G1,impact,n_g_red,from_red,to_red,g1,psi,F,n_v,n_g,1000,vars_SS,1,0.07);

%% Step 7

figure(fig);
line_style = '--';
color = 'red';
plot_IRFs;

%%

ReduceDist_hor = ReduceDist_hors(2);
fprintf('\n\nReduction horizon %d\n',ReduceDist_hor);
color = [1,0.5,0];
solve_reduced_model;

ReduceDist_hor = ReduceDist_hors(3);
fprintf('\n\nReduction horizon %d\n',ReduceDist_hor);
color = 'yellow';
solve_reduced_model;

ReduceDist_hor = ReduceDist_hors(4);
fprintf('\n\nReduction horizon %d\n',ReduceDist_hor);
color = 'green';
solve_reduced_model;

legends = cell(5,1);
legends{1} = 'full';
legends{2} = ['reduced to ',num2str(ReduceDist_hors(1))];
legends{3} = ['reduced to ',num2str(ReduceDist_hors(2))];
legends{4} = ['reduced to ',num2str(ReduceDist_hors(3))];
legends{5} = ['reduced to ',num2str(ReduceDist_hors(4))];

figure(fig);
subplot(2,3,3);
legend(legends{:});

