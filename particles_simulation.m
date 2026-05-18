Npart = 1000;
sigma = sqrt(sigma2);

% Feedback from the computed Schrödinger factor:
% u_t(x) = sigma * grad log(phi_t(x))
ugrid = zeros(Nx,Nt+1);
for n = 1:Nt+1
    ugrid(:,n) = sigma * log_derivative(phi_k(:,n),dx);
end

%% Sample initial particles from p_in
%% Sample initial particles from p_in
p0 = max(p_in,0);
p0(1) = 0;
p0(end) = 0;

% Trapz-consistent normalization
p0 = p0/trapz(x,p0);

cdf = cumtrapz(x,p0);
cdf = cdf/cdf(end);

% interp1 needs unique sample points
[cdfU, ia] = unique(cdf,'stable');
xU = x(ia);

% Make sure CDF spans [0,1]
if cdfU(1) > 0
    cdfU = [0; cdfU];
    xU = [x(1); xU];
end

if cdfU(end) < 1
    cdfU = [cdfU; 1];
    xU = [xU; x(end)];
end

Xpaths = zeros(Npart,Nt+1);
Xpaths(:,1) = interp1(cdfU,xU,rand(Npart,1),'linear','extrap');

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

    % gradW_D = beta*exp(-regD/ell).*(-1/ell).*(D./regD);
    % gradW_D(1:Npart+1:end) = 0;

    interaction = mean(gradW_D,2);

    % Closed-loop finite-N drift
    drift = sigma*uX - sigma2*interaction;

    % Euler-Maruyama step
    Xpaths(:,n+1) = Xn + drift*dt + sigma*sqrt(dt)*randn(Npart,1);
end

fprintf('Particles outside domain: %.2f%%\n', ...
    100*mean(Xpaths(:) < x(1) | Xpaths(:) > x(end)));
%%
idx = [1, round([1/6 2/6 3/6  4/6 5/6 5.5/6]*Nt) + 1, Nt+1];

% cInt  = [0.0000 0.4470 0.7410];   
% cSB = [0.9 0.5 0.32]; 
 

cInt = [0.0000 0.4470 0.7410];     
 cSB  = [0.8500 0.3250 0.0980];       
               
pathColors = [
    0.4940 0.1840 0.5560;   % purple
    0.4660 0.6740 0.1880;   % green
     0.3010 0.7450 0.9330;   % cyan
    0.6350 0.0780 0.1840;   % dark red
    0.2500 0.2500 0.2500;   % dark gray
     0.0000 0.5000 0.5000;   % teal
    0.3500 0.3500 0.7000;   % blue-purple
    ];


num_show = min(7,size(Xpaths,1));
idx_show = randperm(size(Xpaths,1),num_show);

pathColors = 0.7*pathColors + 0.3*ones(size(pathColors));  % muted path colors

xPlot = x(2:end-1);
zData = [p_k(2:end-1,idx); pSB_k(2:end-1,idx)];
zMax = max(zData(:));
zMin = min(zData(:));

samplePathData = Xpaths(idx_show,:);
yMin = min([xPlot(:); samplePathData(:)]);
yMax = max([xPlot(:); samplePathData(:)]);

yPad = 0;

zPart = 0*zMax;
zUpper = 1*zMax;

figure('Color','w','Position',[100 100 1150 780]);
hold on;

% Proxy handles for legend
hInt  = plot3(nan,nan,nan,'-','Color',cInt,'LineWidth',3.0);
hSB   = plot3(nan,nan,nan,'--','Color',cSB,'LineWidth',2.0);
hPart = plot3(nan,nan,nan,'-','Color',[0.45 0.45 0.45],'LineWidth',1.5);

% Plot selected interacting and noninteracting marginals
for m = 1:length(idx)
    n = idx(m);

    if m == 1 || m == length(idx)
        lw = 3.4;
    else
        lw = 2.8;
    end

    plot3(t(n)*ones(length(xPlot),1), xPlot, p_k(2:end-1,n), ...
        '-', 'Color', cInt, 'LineWidth', lw);

    plot3(t(n)*ones(length(xPlot),1), xPlot, pSB_k(2:end-1,n), ...
        '--', 'Color', cSB, 'LineWidth', lw);
end

% Plot 7 sample paths below the density curves
for m = 1:num_show
    Xi = Xpaths(idx_show(m),:);

    plot3(t, Xi, zPart*ones(size(t)), ...
        '-', 'LineWidth', 1.3, ...
         'Color', pathColors(m,:))
end

% xlabel('$t$','Interpreter','latex');
% ylabel('$x$','Interpreter','latex');
% zlabel('$p_t(x)$','Interpreter','latex');

set(gca,'FontSize',30, ...
        'LineWidth',1.2, ...
        'TickLabelInterpreter','latex', ...
        'Box','on');

grid on;

view(-45,15);
pbaspect([2 1 1.25]);

xlim([0,1]);
ylim([xMin, xMax]);
zlim([zPart,zUpper]);

legend([hInt,hSB], ...
    {'Interacting','Noninteracting'}, ...
    'Interpreter','latex','Location','best');

exportgraphics(gcf,'plot-MFSB-sample-paths-repulsive.png','Resolution',300);
%%
function df = log_derivative(u,dx)
    n = numel(u);
    idx = 2:n-1;

    lu = zeros(size(u));
    lu(idx) = log(u(idx));

    df = zeros(size(u));

    df(2)   = (-3*lu(2) + 4*lu(3) - lu(4))/(2*dx);
    df(n-1) = ( 3*lu(n-1) - 4*lu(n-2) + lu(n-3))/(2*dx);

    df(3:n-2) = (lu(4:n-1)-lu(2:n-3))/(2*dx);

    df(1) = 0;
    df(n) = 0;
end
