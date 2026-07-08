%%  Singular Spectrum Analysis (SSA) with Matlab
% This script demonstrates step by step the Singular Spectrum Analysis
% (SSA), following the same four-step structure as the classic SSA
% literature (embedding, decomposition via SVD, grouping, and diagonal
% averaging), and adds two extras: an automatic, principled way to
% choose how many components are "signal" (Gavish & Donoho, 2014), and
% a cross-check against the covariance-matrix formulation of SSA used
% in the climate-science literature (Vautard & Ghil, 1989; Groth &
% Ghil, 2015).

%% Resources and references used to build this script
% [1] Rodrigues, P.C., "Hybrid Strategies for Time Series Forecasting
% using Singular Spectrum Analysis, Classical Time Series Models, and
% Recurrent Neural Networks", invited seminar talk. Paulo Canas
% Rodrigues is Professor of Statistics at the Federal University of
% Bahia (Brazil) and Head & Principal Investigator of the Statistical
% Learning Laboratory (SaLLy).
% <https://www.youtube.com/watch?v=muWEfZTrVrE&t=2264s>

% [2] Gavish, M., and Donoho, D.L., 2014: The optimal hard threshold for
% singular values is 4/sqrt(3), arXiv:1305.5870.
% <https://arxiv.org/pdf/1305.5870>]

% Worked Matlab tutorial this script drew on and adapted (the
% covariance-matrix / Toeplitz-vs-trajectory / PC-RC formulation
% adapted into ssa_cov_decompose, ssa_principal_components, and
% ssa_rc_from_pc below comes from this tutorial, which references
% Vautard & Ghil, 1989, and Groth & Ghil, 2015):

% [3] "Singular Spectrum Analysis - A Beginner's Guide", Mathworks File
% Exchange.
% <https://www.mathworks.com/matlabcentral/fileexchange/58967-singular-spectrum-analysis-beginners-guide>


%% Create (or load) a signal
% To keep things concrete, we generate a synthetic series made of a
% slow trend, a clean seasonal oscillation, and additive white noise.
% If you have your own data, simply replace this block with
% y = yourDataVector;

rng(1);
t = (1:200)';
trend    = 0.02 * t;
seasonal = 3 * sin(2*pi*t/12);
noise    = 0.7 * randn(size(t));
y = trend + seasonal + noise;

figure(1);
set(gcf, 'name', 'Time series y');
clf;
plot(t, y, 'b-');
xlabel('Time'); ylabel('Value');
title('Original series');


%% Choose the window length L
% The window length L is the one real design choice SSA asks of the
% analyst. Following Gavish and Donoho (2014), the optimal-threshold
% theory used below is derived for a square matrix, so we pick L as
% close as possible to K = n - L + 1. So, L is about half the series
% length. This makes the trajectory matrix roughly square.

n = length(y);
L = round(n/2);


%% Embed the series and compute the SVD
% We slide a window of length L across y to build the trajectory
% matrix, then decompose it into eigentriples via the singular value
% decomposition. Each eigentriple is a candidate building block of the
% series. Basically some will turn out to represent signal, rest is noise.

[Y, U, S, V, d] = ssa_decompose(y, L);

fprintf('Trajectory matrix size: %d x %d\n', size(Y,1), size(Y,2));
fprintf('Number of nonzero eigentriples: %d\n', d);


%% Inspect the singular value spectrum
% Rather than eyeballing the spectrum for an elbow or choose to retain 90 
% percent of the signal energy, we use the Gavish-Donoho optimal hard 
% threshold to draw a line separating eigentriples that are likely signal
% (above the noise floor) from values which are noise

[r_opt, tau] = ssa_optimal_rank(S, Y);

figure(2);
set(gcf, 'name', 'Singular value spectrum');
clf;
plot(1:d, S, 'o-'); hold on;
yline(tau, 'r--', 'LineWidth', 1.5);
xlabel('Eigentriple index'); ylabel('Singular value');
title(sprintf('Optimal signal rank r = %d', r_opt));
legend('Singular values', 'Gavish-Donoho threshold \tau^*');

fprintf('Gavish-Donoho optimal threshold tau* = %.4f\n', tau);
fprintf('Automatically selected signal rank r = %d\n', r_opt);


%% Group the eigentriples and reconstruct the signal
% In the grouping step we decide which eigentriples belong together.
% Here we simply take all r_opt eigentriples identified above as one
% "signal" group, but you are free to split this group further: Ex.
% groups = {1, 2:r_opt} to separate a trend eigentriple from a
% seasonal pair, guided by the singular value plot and the
% w-correlation matrix computed a little further down.

groups = {1:r_opt};
[signalRecon, components] = ssa_reconstruct(U, S, V, groups);
noiseRecon = y - signalRecon;

figure(3);
set(gcf, 'name', 'SSA reconstruction');
clf;
plot(t, y, 'Color', [0.7 0.7 0.7]); hold on;
plot(t, signalRecon, 'b', 'LineWidth', 1.5);
legend('Original series', 'Reconstructed signal');
xlabel('Time'); ylabel('Value');
title('Original series and its SSA reconstruction');


%% Check separability with the w-correlation matrix
% Two reconstructed components that are genuinely distinct should have
% a w-correlation close to zero; large values suggest the components
% are entangled and might belong in the same group. Here we compare
% the signal group against the noise, but if you split groups into
% several sub-groups above, each will appear as its own row and column.

allComponents = [components; {noiseRecon}];
Wcorr = ssa_wcorrelation(allComponents, L);
disp('W-correlation matrix (signal group(s), noise):');
disp(Wcorr);


%% Forecast forward with Recurrent SSA
% Because the reconstructed signal satisfies (approximately) a linear
% recurrent formula, we can derive coefficients from the signal
% eigenvectors and use them to extend the series forward in time,
% one step at a time.

a = ssa_forecast_coeffs(U, groups);
nAhead = 24;
yForecast = ssa_recurrent_forecast(signalRecon, a, nAhead);

figure(4);
set(gcf, 'name', 'Recurrent SSA forecast');
clf;
plot(t, y, 'Color', [0.7 0.7 0.7]); hold on;
plot(t, signalRecon, 'b', 'LineWidth', 1.5);
tFuture = (t(end)+1 : t(end)+nAhead)';
plot(tFuture, yForecast, 'r--', 'LineWidth', 1.5);
legend('Original series', 'Reconstructed signal', 'SSA forecast');
xlabel('Time'); ylabel('Value');
title('Original series, reconstruction, and out-of-sample forecast');


%% Cross-check with the covariance-matrix approach
% SSA can equally be formulated through the eigendecomposition of a
% covariance matrix rather than the SVD of the trajectory matrix
% directly. This is the classic route in the climate-science
% literature (Broomhead & King, 1986; Vautard & Ghil, 1989), where the
% eigenvectors are called RHO, the eigenvalues LAMBDA, and the
% projected series are called principal components (PC) and
% reconstructed components (RC). The two formulations are
% mathematically equivalent, so we use this section simply as a sanity
% check on the SVD-based reconstruction above.

[signalRecon_cov, LAMBDA] = ssa_cov_reconstruct(y, L, r_opt, 'trajectory');

fprintf('\nCross-check: max abs difference between SVD-based and\n');
fprintf('covariance-based signal reconstruction = %.3e\n', ...
        max(abs(signalRecon - signalRecon_cov)));

figure(5);
set(gcf, 'name', 'Eigenvalue spectrum (covariance approach)');
clf;
plot(1:d, LAMBDA(1:d), 'o-');
xlabel('Component index'); ylabel('Eigenvalue \lambda');
title('Eigenvalue spectrum from the covariance-matrix formulation');


%%  Local functions

% ---- Core decomposition ------------------------------------------------

function Y = ssa_embed(y, L)
% Builds the trajectory matrix from the series y using window length
% L. We slide a window of length L across y, one step at a time, and
% stack the resulting lagged vectors as the columns of Y.

    y = y(:);
    n = length(y);
    K = n - L + 1;

    if L <= 1 || L >= n
        error('Window length L must satisfy 1 < L < n');
    end

    Y = zeros(L, K);
    for i = 1:K
        Y(:, i) = y(i:i+L-1);
    end
end


function [Y, U, S, V, d] = ssa_decompose(y, L)
% Embeds the series and computes the singular value decomposition of
% the resulting trajectory matrix. The columns of U and V are the left
% and right singular vectors, and S holds the singular values, sorted
% in descending order. Together, a triple (singular value, left
% vector, right vector) is often called an eigentriple.

    Y = ssa_embed(y, L);
    [U, D, V] = svd(Y, 'econ');
    S = diag(D);

    % We keep only the numerically significant (nonzero) singular
    % values, since a finite series only has finitely many.
    tol = max(size(Y)) * eps(max(S));
    d = sum(S > tol);

    U = U(:, 1:d);
    S = S(1:d);
    V = V(:, 1:d);
end


function [r, tau] = ssa_optimal_rank(S, Y, sigma)
% Selects the number of signal eigentriples r using the optimal hard
% threshold of Gavish and Donoho (2014), rather than eyeballing the
% singular value spectrum for an elbow. If the noise level sigma is
% known, the classical threshold tau* = (4/sqrt(3)) * sqrt(n) * sigma
% applies for a square trajectory matrix; a general beta-dependent
% correction is used for non-square matrices. If sigma is unknown, the
% threshold is instead estimated from the data itself using the median
% singular value, which is the version most commonly used in practice.
% L and K are read directly from the size of the trajectory matrix Y.

    if nargin < 3
        sigma = [];
    end

    L = size(Y, 1);
    K = size(Y, 2);

    m = min(L, K);
    nDim = max(L, K);
    beta = m / nDim;

    omega_beta = 0.56*beta^3 - 0.95*beta^2 + 1.82*beta + 1.43;

    if isempty(sigma)
        yMed = median(S);
        tau = omega_beta * yMed;
    else
        lambda_beta = sqrt( (2*(beta+1) + 8*beta) / ...
                             ((beta+1) + sqrt(beta^2 + 14*beta + 1)) );
        tau = lambda_beta * sqrt(nDim) * sigma;
    end

    r = sum(S > tau);
    r = max(r, 1);
end


% ---- Reconstruction ------------------------------------------------

function y = ssa_diag_average(X)
% Converts a matrix X back into a one-dimensional series by averaging
% over its anti-diagonals. This is the diagonal-averaging (or
% Hankelization) step: it reverses the embedding step, returning a
% reconstructed variant of the trajectory matrix to a proper time
% series.

    [L, K] = size(X);
    if L > K
        X = X';
        [L, K] = size(X);
    end

    n = L + K - 1;
    y = zeros(n, 1);
    counts = zeros(n, 1);

    for i = 1:L
        for j = 1:K
            k = i + j - 1;
            y(k) = y(k) + X(i, j);
            counts(k) = counts(k) + 1;
        end
    end

    y = y ./ counts;
end


function [recon, components] = ssa_reconstruct(U, S, V, groups)
% Reconstructs one or more series components from a chosen grouping of
% eigentriples. Each group is summed as a rank-|group| approximation of
% the trajectory matrix, then diagonal-averaged back into a series. The
% sum of all requested groups is returned as the overall reconstruction.

    numGroups = length(groups);
    components = cell(numGroups, 1);

    L = size(U, 1);
    K = size(V, 1);
    n = L + K - 1;

    recon = zeros(n, 1);

    for g = 1:numGroups
        idx = groups{g};
        Xg = zeros(L, K);
        for i = idx
            Xg = Xg + S(i) * U(:, i) * V(:, i)';
        end
        comp = ssa_diag_average(Xg);
        components{g} = comp;
        recon = recon + comp;
    end
end


function Wcorr = ssa_wcorrelation(components, L)
% Computes the pairwise weighted (w-)correlation matrix between
% reconstructed components. A near-zero w-correlation between two
% components indicates that they are well separated by the SSA
% decomposition; a large value suggests they are still entangled and
% may belong in the same group.

    m = length(components);
    n = length(components{1});
    K = n - L + 1;

    w = zeros(n, 1);
    Lstar = min(L, K);
    Kstar = max(L, K);
    for k = 1:n
        if k < Lstar
            w(k) = k;
        elseif k <= Kstar
            w(k) = Lstar;
        else
            w(k) = n - k + 1;
        end
    end

    normF = zeros(m, 1);
    for i = 1:m
        Xi = components{i};
        normF(i) = sqrt(sum(w .* Xi.^2));
    end

    Wcorr = zeros(m, m);
    for i = 1:m
        for j = 1:m
            num = sum(w .* components{i} .* components{j});
            Wcorr(i, j) = num / (normF(i) * normF(j));
        end
    end
end


% ---- Forecasting ------------------------------------------------

function a = ssa_forecast_coeffs(U, groups)
% Computes the linear recurrent formula (LRF) coefficients used for
% Recurrent SSA forecasting. These coefficients come directly from the
% eigenvectors of the signal group: each eigenvector is split into its
% last component and its remaining components, and the coefficients
% are a particular weighted combination of the latter. The eigentriple
% indices to use are taken directly from groups, so the caller does not
% need to select the signal columns of U beforehand.

    signalIdx = [groups{:}];
    Usig = U(:, signalIdx);

    pi_j = Usig(end, :)';
    Uv   = Usig(1:end-1, :);

    v2 = sum(pi_j.^2);
    if v2 >= 1
        error(['Verticality coefficient v^2 >= 1: the series is not ' ...
               'suitable for recurrent forecasting with this L/r choice.']);
    end

    a = (1/(1 - v2)) * (Uv * pi_j);
end


function yForecast = ssa_recurrent_forecast(reconSignal, a, nAhead)
% Produces out-of-sample forecasts using the Recurrent SSA algorithm.
% Each new point is predicted as a linear combination of the preceding
% reconstructed (or previously forecasted) values, using the LRF
% coefficients a, and the process is repeated forward one step at a
% time until nAhead future points have been produced.

    d = length(a);
    y = reconSignal(:);
    n = length(y);

    yExtended = [y; zeros(nAhead, 1)];

    for h = 1:nAhead
        idx = n + h;
        window = yExtended(idx-d:idx-1);
        yExtended(idx) = a' * window;
    end

    yForecast = yExtended(n+1:end);
end


% ---- Covariance-matrix alternative ------------------------------------------------

function [RHO, LAMBDA, C] = ssa_cov_decompose(y, M, method)
% Provides an alternative route into SSA via the eigendecomposition of
% a covariance matrix, following the classic climate-science
% formulation. Two estimators of the covariance matrix are offered:
% the 'trajectory' approach of Broomhead and King, which builds C
% directly from the embedded series, and the 'toeplitz' approach of
% Vautard and Ghil, which builds C from the lagged autocovariance
% function and so is symmetric by construction. The two SSA routes
% (this one and the SVD-based one above) are mathematically equivalent.

    if nargin < 3
        method = 'trajectory';
    end

    y = y(:);
    N = length(y);

    switch lower(method)
        case 'trajectory'
            Y = zeros(N-M+1, M);
            for m = 1:M
                Y(:, m) = y((1:N-M+1) + m - 1);
            end
            C = (Y' * Y) / (N - M + 1);

        case 'toeplitz'
            covX = xcorr(y - mean(y), M-1, 'unbiased');
            C = toeplitz(covX(M:end));

        otherwise
            error('method must be ''trajectory'' or ''toeplitz''');
    end

    [RHO, LAMBDA] = eig(C);
    LAMBDA = diag(LAMBDA);
    [LAMBDA, ind] = sort(LAMBDA, 'descend');
    RHO = RHO(:, ind);
end


function PC = ssa_principal_components(Y, RHO)
% Projects the embedded series onto the eigenvectors RHO to obtain the
% principal components (PC), following Groth and Ghil (2015). Y is
% expected in the L-by-K convention used by ssa_embed, so it is
% transposed internally to match the K-by-M convention of the
% covariance-matrix formulation before projecting.

    PC = Y' * RHO;
end


function RC = ssa_rc_from_pc(PC, RHO, N)
% Reconstructs the components (RC) from the principal components (PC)
% by inverting the projection and then averaging along anti-diagonals,
% mirroring the diagonal-averaging step of the SVD-based formulation.
% Summing all columns of RC recovers the full series; summing only the
% first few columns recovers the signal alone.

    M = size(RHO, 1);
    RC = zeros(N, M);

    for m = 1:M
        buf = PC(:, m) * RHO(:, m)';
        buf = buf(end:-1:1, :);
        for i = 1:N
            RC(i, m) = mean(diag(buf, -(N-M+1) + i));
        end
    end
end


function [signalRecon_cov, LAMBDA] = ssa_cov_reconstruct(y, L, r, method)
% Runs the whole covariance-matrix route in a single call: computes RHO
% and LAMBDA via ssa_cov_decompose, embeds the series and projects it
% onto RHO to obtain the principal components, reconstructs all
% components via ssa_rc_from_pc, and sums the first r of them into one
% signal reconstruction. This keeps the covariance-based cross-check to
% a single function call from the script, rather than chaining several
% steps together by hand.

    if nargin < 4
        method = 'trajectory';
    end

    [RHO, LAMBDA, ~] = ssa_cov_decompose(y, L, method);
    Y = ssa_embed(y, L);
    PC = ssa_principal_components(Y, RHO);
    N = length(y);
    RC = ssa_rc_from_pc(PC, RHO, N);

    signalRecon_cov = sum(RC(:, 1:r), 2);
end