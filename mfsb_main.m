clear;
close;
clc;
%% Parameters
sigma2 = 0.5;

xMin = -1.25; xMax = 1.25;
Nx = 60;
Nt = 300;
x  = linspace(xMin,xMax,Nx)';
dx = x(2)-x(1);

t  = linspace(0,1,Nt+1);
dt = 1/Nt;

ItersOut   = 300;
ItersMid   = 400;
ItersInner = 300;


bcValPhi = 1e-6;
bcValHat = 1e-6;
bcValPBase = bcValPhi*bcValHat;
posTol    = 0;
metricThr = 1e-6;


tolInner = 1e-8;
tolMid   = 1e-8;
tolOut   = 1e-8;

maxSubHalvings = 30;
%% Marginals
p_in = exp(-(x+0.4).^2./0.08) + 0.5*exp(-(x-0.5).^2./0.08);
p_in = normalize_density(p_in,dx,bcValPBase);

p_fin = exp(-(x-0.4).^2./0.08);
p_fin = normalize_density(p_fin,dx,bcValPBase);

figure, plot(x, p_in, x, p_fin);
%% Interaction
S = x-x.';
R = abs(S);
epsReg = 0.05;

reg = sqrt(R.^2 + epsReg^2);

beta  = -1.5;
alpha = 0.3;
Wmat  = beta./(reg.^alpha);
dWmat = -beta*alpha*S.*(reg.^(-(alpha+2)));
dWmat(1:Nx+1:end) = 0;

% Gaussian alternative:
% beta  = -0.7;
% alpha = 0.15;
% Wmat  = beta*exp(-(R.^2)/alpha);
% dWmat = -(2*beta/alpha)*S.*exp(-(R.^2)/alpha);
% dWmat(1:Nx+1:end) = 0;

% Quadratic alternative:
% beta = 0.5;
% Wmat = 0.5*beta*(R.^2);
% dWmat = beta*S;        % derivative w.r.t. x of 0.5 beta (x-y)^2
% dWmat(1:Nx+1:end) = 0;

Wconvp     = @(p) (Wmat*p)*dx;
gradWconvp = @(p) (dWmat*p)*dx;

Win  = Wconvp(p_in);
Wfin = Wconvp(p_fin);
%% Discrete operators
evec = ones(Nx,1);
L = spdiags([evec -2*evec evec],[-1 0 1],Nx,Nx)/(dx^2);

A_dt = make_dirichlet_matrix(speye(Nx)-dt*(sigma2/2)*L);

mHeat = Nt;
dtau  = 1/mHeat;
A_heat = make_dirichlet_matrix(speye(Nx)-dtau*(sigma2/2)*L);

A_dt_dec   = decomposition(A_dt,'lu');
A_heat_dec = decomposition(A_heat,'lu');

Dx_ = @(f) centered_Dx(f,dx);
DxLog_ = @(u) log_derivative_no_endpoint(u,dx);
%% Noninteracting Sinkhorn initialization

kernel_phi = @(u,tau) heat_propagate(u,tau,dtau,sigma2,L,A_heat_dec,bcValPhi,posTol);
kernel_hat = @(u,tau) heat_propagate(u,tau,dtau,sigma2,L,A_heat_dec,bcValHat,posTol);

maxSink = 400;
tolSink = 1e-12;

f = ones(Nx,1); f(1)=bcValHat; f(end)=bcValHat;
g = ones(Nx,1); g(1)=bcValPhi; g(end)=bcValPhi;

errSink1 = nan(maxSink,1);
errSink2 = nan(maxSink,1);

for kk = 1:maxSink
    Pg = kernel_phi(g,1.0);
    require_positive(Pg,'Pg in Sinkhorn',posTol);
    f_new = p_in./Pg;
    f_new(1)=bcValHat; f_new(end)=bcValHat;
    require_positive(f_new,'f_new in Sinkhorn',posTol);

    Pf = kernel_hat(f_new,1.0);
    require_positive(Pf,'Pf in Sinkhorn',posTol);
    g_new = p_fin./Pf;
    g_new(1)=bcValPhi; g_new(end)=bcValPhi;
    require_positive(g_new,'g_new in Sinkhorn',posTol);

    errSink1(kk) = hilbert_metric(f_new,f,metricThr);
    errSink2(kk) = hilbert_metric(g_new,g,metricThr);

    f = f_new;
    g = g_new;

    if (errSink1(kk) < tolSink) && (errSink2(kk) < tolSink)
        break;
    end
end

%% Build initial factor trajectories
p_k      = zeros(Nx,Nt+1);
phi_k    = zeros(Nx,Nt+1);
hatphi_k = zeros(Nx,Nt+1);
massSB_raw = nan(Nt+1,1);
massSB_raw_noBC = nan(Nt+1,1);

for n = 1:Nt+1
    tn = t(n);
    hatphi_k(:,n) = kernel_hat(f,tn);
    phi_k(:,n)    = kernel_phi(g,1-tn);

    require_positive(hatphi_k(:,n),sprintf('initial hatphi at n=%d',n),posTol);
    require_positive(phi_k(:,n),sprintf('initial phi at n=%d',n),posTol);

    prod = phi_k(:,n).*hatphi_k(:,n);
    prod(1)=bcValPBase;
    prod(end)=bcValPBase;
    massSB_raw(n) = sum(prod)*dx;
    p_k(:,n) = normalize_density(prod,dx,bcValPBase);
    p_k(1,n)=bcValPBase;
    p_k(end,n)=bcValPBase;

end

pSB_k = p_k;
phiSB = phi_k;
hatphiSB = hatphi_k;

fprintf(['Initial boundary errors:\n', ...
    '  t=0: dH = %.3e, max abs = %.3e\n', ...
    '  t=T: dH = %.3e, max abs = %.3e\n'], ...
    hilbert_metric(pSB_k(:,1),p_in,metricThr), ...
    max(abs(pSB_k(:,1)-p_in)), ...
    hilbert_metric(pSB_k(:,end),p_fin,metricThr), ...
    max(abs(pSB_k(:,end)-p_fin)));

%% Error storage
errInner_hatphiIn = nan(ItersInner,1);
errInner_phiFin   = nan(ItersInner,1);
errMid_oplus      = nan(ItersMid,1);

ErrdHTime   = nan(ItersOut,Nt+1);
errOut_dS   = nan(ItersOut,1);
boundaryErr0 = nan(ItersOut,1);
boundaryErrT = nan(ItersOut,1);


err_hatphiIn_all = nan(ItersInner,ItersMid,ItersOut);
err_phiFin_all   = nan(ItersInner,ItersMid,ItersOut);
err_max_all      = nan(ItersInner,ItersMid,ItersOut);


errMid_oplus_all = nan(ItersMid,ItersOut);


innerStop = zeros(ItersMid,ItersOut);
midStop   = zeros(ItersOut,1);

%% Main outer loop
tic;
for k = 1:ItersOut

    pCoeff = p_k;
    Wp = zeros(Nx,Nt+1);
    q  = zeros(Nx,Nt+1);
    for n = 1:Nt+1
        require_positive(pCoeff(:,n),sprintf('pCoeff at k=%d,n=%d',k,n),posTol);
        Wp(:,n) = Wconvp(pCoeff(:,n));
        q(:,n)  = gradWconvp(pCoeff(:,n));
    end

    phi_j    = phi_k;
    hatphi_j = hatphi_k;

    errMid_oplus(:) = nan;

    for j = 1:ItersMid

        react_phi_j    = zeros(Nx,Nt+1);
        react_hatphi_j = zeros(Nx,Nt+1);

        for n = 1:Nt+1
            require_positive(phi_j(:,n),sprintf('phi_j at k=%d,j=%d,n=%d',k,j,n),posTol);
            require_positive(hatphi_j(:,n),sprintf('hatphi_j at k=%d,j=%d,n=%d',k,j,n),posTol);

            ptn = pCoeff(:,n);
            dlog_phi_j = DxLog_(phi_j(:,n));
            react_phi_j(:,n) = -sigma2*gradWconvp(ptn.*dlog_phi_j);

            dlog_hatphi_j = DxLog_(hatphi_j(:,n));
            react_hatphi_j(:,n) = -sigma2*gradWconvp(ptn.*dlog_hatphi_j);
        end

        phi1_i    = phi_j(:,end);
        hatphi0_i = hatphi_j(:,1);

        phi_i    = phi_j;
        hatphi_i = hatphi_j;

        errInner_hatphiIn(:) = nan;
        errInner_phiFin(:)   = nan;

        for i = 1:ItersInner
            phi1_prev_i    = phi1_i;
            hatphi0_prev_i = hatphi0_i;

            %% Backward solve for phi
            phi_next = zeros(Nx,Nt+1);
            phi_next(:,end) = phi1_i;
            phi_next(1,end) = bcValPhi;
            phi_next(end,end) = bcValPhi;
            require_positive(phi_next(:,end),sprintf('phi terminal at k=%d,j=%d,i=%d',k,j,i),posTol);

            for n = Nt+1:-1:2
                phi_next(:,n-1) = positive_imex_step( ...
                    phi_next(:,n), q(:,n), react_phi_j(:,n), ...
                    dt, dx, sigma2, L, A_dt_dec, bcValPhi, posTol, maxSubHalvings, ...
                    +1, sprintf('phi step k=%d,j=%d,i=%d,n=%d',k,j,i,n));
            end

            %% Initial boundary scaling for hatphi
            phi0 = phi_next(:,1);
            require_positive(phi0,sprintf('phi0 before hatphi update k=%d,j=%d,i=%d',k,j,i),posTol);

            hatphi0_i = zeros(Nx,1);
            idx = 2:Nx-1;
            hatphi0_i(idx) = (p_in(idx)./phi0(idx)).*exp(2*Win(idx));
            hatphi0_i(1) = bcValHat;
            hatphi0_i(end) = bcValHat;
            require_positive(hatphi0_i,sprintf('hatphi0 update k=%d,j=%d,i=%d',k,j,i),posTol);

            %% Forward solve for hatphi
            hatphi_next = zeros(Nx,Nt+1);
            hatphi_next(:,1) = hatphi0_i;
            hatphi_next(1,1) = bcValHat;
            hatphi_next(end,1) = bcValHat;

            for n = 1:Nt
                hatphi_next(:,n+1) = positive_imex_step( ...
                    hatphi_next(:,n), q(:,n), react_hatphi_j(:,n), ...
                    dt, dx, sigma2, L, A_dt_dec, bcValHat, posTol, maxSubHalvings, ...
                    -1, sprintf('hatphi step k=%d,j=%d,i=%d,n=%d',k,j,i,n));
            end

            %% Final boundary scaling for phi
            hatphiT = hatphi_next(:,end);
            require_positive(hatphiT,sprintf('hatphiT before phi update k=%d,j=%d,i=%d',k,j,i),posTol);

            phi1_i = zeros(Nx,1);
            phi1_i(idx) = (p_fin(idx)./hatphiT(idx)).*exp(2*Wfin(idx));
            phi1_i(1) = bcValPhi;
            phi1_i(end) = bcValPhi;
            require_positive(phi1_i,sprintf('phi1 update k=%d,j=%d,i=%d',k,j,i),posTol);

            errInner_hatphiIn(i) = hilbert_metric(hatphi0_i,hatphi0_prev_i,metricThr);
            errInner_phiFin(i)   = hilbert_metric(phi1_i,phi1_prev_i,metricThr);

            err_hatphiIn_all(i,j,k) = errInner_hatphiIn(i);
            err_phiFin_all(i,j,k)   = errInner_phiFin(i);
            err_max_all(i,j,k)      = max(errInner_hatphiIn(i),errInner_phiFin(i));

            phi_i    = phi_next;
            hatphi_i = hatphi_next;

            if (errInner_hatphiIn(i) < tolInner) && (errInner_phiFin(i) < tolInner)
                break;
            end
        end
        innerStop(j,k) = i;

        phi_jp1    = phi_i;
        hatphi_jp1 = hatphi_i;
        phi_jp1(:,end)    = phi1_i;
        hatphi_jp1(:,1)   = hatphi0_i;

        err_phi_traj = 0;
        err_hat_traj = 0;
        for n = 1:Nt+1
            err_phi_traj = max(err_phi_traj, ...
                hilbert_metric(phi_jp1(:,n),phi_j(:,n),metricThr));
            err_hat_traj = max(err_hat_traj, ...
                hilbert_metric(hatphi_jp1(:,n),hatphi_j(:,n),metricThr));
        end
        errMid_oplus(j) = max(err_phi_traj,err_hat_traj);
        errMid_oplus_all(j,k) = errMid_oplus(j);

        phi_j    = phi_jp1;
        hatphi_j = hatphi_jp1;

        if errMid_oplus(j) < tolMid
            break;
        end
    end
    midStop(k) = j;

    phi_kp1    = phi_j;
    hatphi_kp1 = hatphi_j;

    %% Density reconstruction
    p_kp1 = zeros(Nx,Nt+1);


    for n = 1:Nt+1
        cand = reconstruct_density_product(Wp(:,n),phi_kp1(:,n),hatphi_kp1(:,n),bcValPhi,bcValHat,posTol, ...
            sprintf('p candidate k=%d,n=%d',k,n));

       if n >=1 && n<=Nt+1
        cand = normalize_density(cand,dx,density_bc_from_weight(Wp(:,n),bcValPhi,bcValHat));
       end

        p_kp1(:,n) = cand;
        pBCn = density_bc_from_weight(Wp(:,n),bcValPhi,bcValHat);
        p_kp1(1,n) = pBCn(1);
        p_kp1(end,n) = pBCn(2);

        require_positive(p_kp1(:,n),sprintf('damped p k=%d,n=%d',k,n),posTol);
    end

    boundaryErr0(k) = hilbert_metric(p_kp1(:,1),p_in,metricThr);
    boundaryErrT(k) = hilbert_metric(p_kp1(:,end),p_fin,metricThr);

    fprintf('k=%3d, i=%3d, j=%3d, b0=%.3e, bT=%.3e\n', ...
        k,i,j,boundaryErr0(k),boundaryErrT(k));

    for n = 1:Nt+1
        ErrdHTime(k,n) = hilbert_metric(p_kp1(:,n),p_k(:,n),metricThr);
    end

    errOut_dS(k) = max(ErrdHTime(k,2:Nt));

    p_k      = p_kp1;
    phi_k    = phi_kp1;
    hatphi_k = hatphi_kp1;

    if errOut_dS(k) < tolOut
        break;
    end
end
elapsed_time = toc;
fprintf('Elapsed time: %.2f seconds\n',elapsed_time);

%% Plots
set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

idx = [1, round([1/6 2/6 3/6 4/6 5/6]*Nt) + 1, Nt+1];

colorInt = [0.0000 0.4470 0.7410];      
colorSB  = [0.9300 0.5000 0.2500];      

plotIdx = (x >= -1) & (x <= 1);
xPlot = x(plotIdx);
zData = [p_k(plotIdx,idx); pSB_k(plotIdx,idx)];

zMax = max(zData(:));
zMin = min(zData(:));

zPad = 0.05*(zMax-zMin);
if zPad == 0
    zPad = 0.05*zMax;
end

figure('Color','w','Position',[100 100 1150 780]);
hold on;

hInt = plot3(nan,nan,nan,'-','Color',colorInt,'LineWidth',3.0);
hSB  = plot3(nan,nan,nan,'--','Color',colorSB,'LineWidth',3.0);

for m = 1:length(idx)
    n = idx(m);

   plot3(t(n)*ones(length(xPlot),1), xPlot, p_k(plotIdx,n), ...
    '-', 'Color', colorInt, 'LineWidth', 3);

plot3(t(n)*ones(length(xPlot),1), xPlot, pSB_k(plotIdx,n), ...
    '--', 'Color', colorSB, 'LineWidth', 3);
end



set(gca,'FontSize',30, ...
        'LineWidth',1.2, ...
        'TickLabelInterpreter','latex', ...
        'Box','on');

grid on;
view(-45,15);
pbaspect([1.5 1 1]);


xlim([0,1]);
ylim([-1,1]);
zlim([max(0,zMin-zPad), zMax+zPad]);

legend([hInt,hSB], {'Interacting','Noninteracting SB'}, ...
    'Location','northeast');
exportgraphics(gcf,'plot-Ex1-3d-two-method-colors-repulsive.png','Resolution',300);



kFinal = k;
jFinal = midStop(kFinal);
iFinal = innerStop(jFinal,kFinal);

fprintf('Final active indices: iFinal=%d, jFinal=%d, kFinal=%d\n', ...
    iFinal,jFinal,kFinal);

% Last inner loop at final j,k.
figure('Color','w','Position',[100 100 950 650]);
y1 = err_phiFin_all(1:iFinal,jFinal,kFinal);
y2 = err_hatphiIn_all(1:iFinal,jFinal,kFinal);
y3 = err_max_all(1:iFinal,jFinal,kFinal);

semilogy(1:iFinal,y1,'-o','LineWidth',3);
hold on;
semilogy(1:iFinal,y2,'-s','LineWidth',3);
semilogy(1:iFinal,y3,'-^','LineWidth',3);

grid on;
set(gca,'FontSize',28, ...
        'LineWidth',1.2, ...
        'TickLabelInterpreter','latex', ...
        'Box','on');

xlabel('$i$');
ylabel('Inner error');
legend({'$\varphi_1$ update','$\hat\varphi_0$ update','max'}, ...
    'Location','best');



yy = [y1(:); y2(:); y3(:)];
yy = yy(isfinite(yy) & yy > 0);
ylim([0.8*min(yy), 1.2*max(yy)]);

% Last middle loop at final k.
figure('Color','w','Position',[100 100 950 650]);
yMid = errMid_oplus_all(1:jFinal,kFinal);
semilogy(1:jFinal,yMid,'-o','LineWidth',3);
grid on;
set(gca,'FontSize',28, ...
        'LineWidth',1.2, ...
        'TickLabelInterpreter','latex', ...
        'Box','on');

xlabel('$j$');
ylabel('$d_H^\oplus$');


% Outer loop.
figure('Color','w','Position',[100 100 950 650]);
yOut = errOut_dS(1:kFinal);
semilogy(1:kFinal,yOut,'-o','LineWidth',6);
grid on;
set(gca,'FontSize',50, ...
        'LineWidth',1.2, ...
        'TickLabelInterpreter','latex', ...
        'Box','on', 'YAxisLocation','right');

% xlabel('$k$');
% ylabel('$d_H^{\mathcal S}(p^{(k)}, p^{(k+1)})$');

xlim([1,kFinal]);
ylim([1E-7,1E1]);
yticks([1E-6 1E-3 1E1]);
yticklabels({'$10^{-6}$','$10^{-3}$','$10^{0}$'});
set(gca,'TickLabelInterpreter','latex');
exportgraphics(gcf,'plot-Ex2-k-attracting.pdf','Resolution',300);

figure('Color','w','Position',[100 100 900 650]);
Wplot = beta./((sqrt(x.^2+epsReg.^2).^alpha));
plot(x,Wplot,'LineWidth',6);

grid on;
set(gca,'FontSize',50, ...
        'LineWidth',1.5, ...
        'TickLabelInterpreter','latex', ...
        'Box','on');

% xlabel('$x$');
% ylabel('$W(x)$');
xPad = 0.03*(x(end)-x(1));
yPad = 0.08*(max(Wplot)-min(Wplot));
xlim([x(1)-xPad, x(end)+xPad]);
ylim([min(Wplot)-yPad, max(Wplot)+yPad]);
exportgraphics(gcf,'plot-Ex1-W-attracting.pdf','Resolution',300);
%% Local functions
function bc = endpoint_bc_pair(bcVal)
    if isscalar(bcVal)
        bc = [bcVal; bcVal];
    else
        bc = bcVal(:);
        if numel(bc) ~= 2
            error('Boundary value must be a scalar or a two-entry vector.');
        end
    end
end

function pBC = density_bc_from_weight(Wp,bcValPhi,bcValHat)
    pBC = exp(-2*[Wp(1); Wp(end)]).*bcValPhi.*bcValHat;
end

function A = make_dirichlet_matrix(A)
    n = size(A,1);
    A(1,:) = 0;      A(1,1) = 1;
    A(n,:) = 0;      A(n,n) = 1;
end

function u = impose_dirichlet(u,bcVal)
    u(1) = bcVal;
    u(end) = bcVal;
end

function p = normalize_density(p,dx,bcVal)
    p = p(:);
    bc = endpoint_bc_pair(bcVal);
    p(1) = bc(1);
    p(end) = bc(2);
    mass = sum(p)*dx;
    if ~isfinite(mass) || mass <= 0
        error('Cannot normalize density: mass=%g',mass);
    end
    p = p/mass;
    p(1) = bc(1);
    p(end) = bc(2);

    % Correct the interior mass after imposing endpoint values.
    endpointMass = (p(1)+p(end))*dx;
    interiorMass = sum(p(2:end-1))*dx;
    targetInteriorMass = 1 - endpointMass;
    if targetInteriorMass <= 0 || interiorMass <= 0
        error('Bad Dirichlet normalization: endpointMass=%g, interiorMass=%g',endpointMass,interiorMass);
    end
    p(2:end-1) = p(2:end-1)*(targetInteriorMass/interiorMass);
end

function require_positive(u,name,posTol)
    if any(~isfinite(u(:)))
        error('%s has nonfinite entries.',name);
    end
    if any(u(2:end-1) <= posTol)
        [val,idx0] = min(u(2:end-1));
        error('%s lost interior positivity: min interior=%e at interior index %d.', ...
            name,val,idx0+1);
    end
end

function u = heat_propagate(u0,tau,dtau,sigma2,L,A_full_dec,bcVal,posTol)
    u = u0(:);
    u = impose_dirichlet(u,bcVal);
    require_positive(u,'heat initial data',posTol);

    if tau <= 0
        return;
    end

    nfull = floor(tau/dtau);
    trem  = tau - nfull*dtau;

    for kk = 1:nfull
        rhs = impose_dirichlet(u,bcVal);
        u = A_full_dec\rhs;
        u = impose_dirichlet(u,bcVal);
        require_positive(u,'heat propagation',posTol);
    end

    if trem > 1e-14
        A = make_dirichlet_matrix(speye(numel(u))-trem*(sigma2/2)*L);
        rhs = impose_dirichlet(u,bcVal);
        u = A\rhs;
        u = impose_dirichlet(u,bcVal);
        require_positive(u,'heat propagation remainder',posTol);
    end
end

function uPrev = positive_imex_step(u, q, react, dt, dx, sigma2, L, A_dt_dec, bcVal, posTol, maxHalvings, modeSign, name)
    % modeSign = +1 reproduces: rhs = u + h*( advect(u,-sigma2*q) - react.*u )
    % modeSign = -1 reproduces: rhs = u + h*(-advect(u, sigma2*q) - react.*u )

    uCur = impose_dirichlet(u(:),bcVal);
    require_positive(uCur,[name ' input'],posTol);

    rem = dt;
    while rem > 1e-15
        h = rem;
        accepted = false;

        for hh = 1:maxHalvings
            if modeSign == +1
                drift = advect_upwind(uCur,-sigma2*q,dx);
                rhs = uCur + h*(drift - react.*uCur);
            else
                drift = advect_upwind(uCur, sigma2*q,dx);
                rhs = uCur + h*(-drift - react.*uCur);
            end

            rhs = impose_dirichlet(rhs,bcVal);

            if all(isfinite(rhs)) && all(rhs(2:end-1) > 0)
                if abs(h-dt) <= 10*eps(dt)
                    uTrial = A_dt_dec\rhs;
                else
                    A = make_dirichlet_matrix(speye(numel(uCur))-h*(sigma2/2)*L);
                    uTrial = A\rhs;
                end
                uTrial = impose_dirichlet(uTrial,bcVal);

                if all(isfinite(uTrial)) && all(uTrial(2:end-1) > 0)
                    accepted = true;
                    break;
                end
            end

            h = 0.5*h;
        end

        if ~accepted
            error('%s: could not find positive explicit/implicit substep. Try smaller dt/Nt or weaker beta.',name);
        end

        uCur = uTrial;
        rem = rem - h;
    end

    uPrev = uCur;
end

function p = reconstruct_density_product(Wp,phi,hatphi,bcValPhi,bcValHat,posTol,name)
    require_positive(phi,[name ' phi'],posTol);
    require_positive(hatphi,[name ' hatphi'],posTol);

    logp = -2*Wp + log(phi) + log(hatphi);

    if any(~isfinite(logp))
        error('%s: log-density product has nonfinite entries.',name);
    end
    if max(logp(2:end-1)) > log(realmax)
        error('%s: density product overflows. This is a real scaling/gauge problem.',name);
    end
    if min(logp(2:end-1)) < log(realmin)
        error('%s: density product underflows. This is a real scaling/gauge problem.',name);
    end

    p = exp(logp);
    pBC = density_bc_from_weight(Wp,bcValPhi,bcValHat);
    p(1) = pBC(1);
    p(end) = pBC(2);
    require_positive(p,name,posTol);
end

function df = centered_Dx(f,dx)
    n = numel(f);
    df = zeros(size(f));
    df(2:n-1) = (f(3:n)-f(1:n-2))/(2*dx);
    df(1) = 0;
    df(n) = 0;
end

function adv = advect_upwind(v,a,dx)
    n = numel(v);
    adv = zeros(size(v));

    for ii = 2:n-1
        if a(ii) >= 0
            adv(ii) = a(ii)*(v(ii)-v(ii-1))/dx;
        else
            adv(ii) = a(ii)*(v(ii+1)-v(ii))/dx;
        end
    end

    adv(1) = 0;
    adv(n) = 0;
end

function df = log_derivative_no_endpoint(u,dx)
n = numel(u);
lu = log(u);
df = zeros(size(u));


df(2) = (lu(3)-lu(2))/dx;
df(n-1) = (lu(n-1)-lu(n-2))/dx;


df(3:n-2) = (lu(4:n-1)-lu(2:n-3))/(2*dx);


df(1) = 0;
df(n) = 0;
end

function d = hilbert_metric(u,v,thr)
    u = u(:);
    v = v(:);

    cond = isfinite(u) & isfinite(v) & (u > thr) & (v > thr);

    if nnz(cond) < 5
        cond = isfinite(u) & isfinite(v) & (u > 0) & (v > 0);
    end

    if nnz(cond) == 0
        d = Inf;
        return;
    end

    lr = log(u(cond)) - log(v(cond));
    d = max(lr) - min(lr);
end
