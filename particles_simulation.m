Npart = 300;
sigma = sqrt(sigma2);

% Feedback from the computed Schrödinger factor:
% u_t(x) = sigma * grad log(phi_t(x))
ugrid = zeros(Nx,Nt+1);
for n = 1:Nt+1
    ugrid(:,n) = sigma * log_derivative_no_endpoint(phi_k(:,n),dx);
end

%% Sample initial particles from p_in
p0 = max(p_in,0);
p0 = p0/(sum(p0)*dx);

cdf = cumsum(p0)*dx;
cdf = cdf/cdf(end);

Xpaths = zeros(Npart,Nt+1);
for ii = 1:Npart
    r = rand;
    Xpaths(ii,1) = interp1(cdf,x,r,'linear','extrap');
end

%% Euler-Maruyama simulation of the finite-N interacting system
for n = 1:Nt
    Xn = Xpaths(:,n);

    % Evaluate feedback u_t(X_i)
    Xeval = min(max(Xn,x(1)),x(end));
    uX = interp1(x,ugrid(:,n),Xeval,'linear','extrap');

    % Pairwise interaction:
    % interaction_i = (1/N) sum_j grad W(X_i - X_j)
    D = Xn - Xn.';

    regD = sqrt(D.^2 + epsReg^2);
    gradW_D = -beta*alpha*D.*(regD.^(-(alpha+2)));
    gradW_D(1:Npart+1:end) = 0;

    interaction = mean(gradW_D,2);

    % Closed-loop finite-N drift
    drift = sigma*uX - sigma2*interaction;

    % Euler-Maruyama step
    Xpaths(:,n+1) = Xn + drift*dt + sigma*sqrt(dt)*randn(Npart,1);
end

fprintf('Particles outside domain: %.2f%%\n', ...
    100*mean(Xpaths(:) < x(1) | Xpaths(:) > x(end)));
%%
idx = [1, round([1/6 2/6 3/6 4/6 5/6]*Nt) + 1, Nt+1];

% cInt  = [0.0000 0.4470 0.7410];   
colorSB  = [0.8500 0.3250 0.0980 0.7]; 
 

cInt = [0.0000 0.4470 0.7410];     
% cSB  = [0.8500 0.3250 0.0980];       
               

figure('Color','w','Position',[100 100 1150 780]);
hold on;

% Proxy handles for legend
hInt  = plot3(nan,nan,nan,'-','Color',cInt,'LineWidth',3.0);
hSB   = plot3(nan,nan,nan,'--','Color',cSB,'LineWidth',3.0);
% hPart = plot3(nan,nan,nan,'-','Color',cPart,'LineWidth',1.8);

% Plot selected interacting and noninteracting marginals
for m = 1:length(idx)
    n = idx(m);

    
    plot3(t(n)*ones(length(x)-2,1), x(2:end-1), p_k(2:end-1,n), ...
        '-', 'Color', cInt, 'LineWidth', 3);

    
    plot3(t(n)*ones(length(x)-2,1), x(2:end-1), pSB_k(2:end-1,n), ...
        '--', 'Color', cSB, 'LineWidth', 3);
end

num_show = min(7,size(Xpaths,1));
idx_show = randperm(size(Xpaths,1),num_show);

zMax = max([p_k(:); pSB_k(:)]);
zPart = 0*zMax;

pathColors = lines(num_show);
pathColors = 0.55*pathColors + 0.45*ones(size(pathColors));  

hPart = plot3(nan,nan,nan,'-','Color',[0.3 0.3 0.3],'LineWidth',1.5);

for m = 1:num_show
    Xi = Xpaths(idx_show(m),:);

    plot3(t, Xi, zPart*ones(size(t)), ...
        '-', ...
        'Color', pathColors(m,:), ...
        'LineWidth', 1.3);
end

xlabel('$t$','Interpreter','latex');
ylabel('$x$','Interpreter','latex');
zlabel('$p_t(x)$','Interpreter','latex');

set(gca,'FontSize',28, ...
    'LineWidth',1.2, ...
    'TickLabelInterpreter','latex', ...
    'Box','on');

grid on;

% Same visual proportions as your MFSB figure
view(-45,15);
pbaspect([1.25 1 1]);

xlim([0,1]);
ylim([x(1)-0.01,x(end)+0.01]);
zlim([zPart,1.05*zMax]);

legend([hInt,hSB], ...
    {'Interacting SB','Noninteracting SB'}, ...
    'Interpreter','latex','Location','northeast');

exportgraphics(gcf,'plot-MFSB-with-7-particle-paths.png','Resolution',400);
%%
function df = log_derivative_no_endpoint(u,dx)
n = numel(u);
lu = log(u);
df = zeros(size(u));

% Avoid using the artificial Dirichlet endpoint values in the
% first interior derivatives.
df(2) = (lu(3)-lu(2))/dx;
df(n-1) = (lu(n-1)-lu(n-2))/dx;

% Standard centered derivative away from the boundary.
df(3:n-2) = (lu(4:n-1)-lu(2:n-3))/(2*dx);

% Endpoints are not used dynamically.
df(1) = 0;
df(n) = 0
end