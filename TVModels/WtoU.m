function [U] = WtoU(W, m,n, mu, beta, C)
% TV Solver
% Input£º
% ---- U: current image
% ---- Bn: 2D blurred and noisy image
% ---- H:  filter matrix
% ---- mu: paramters
% ---- beta: penalty parameters
% ---- C: solver matrix
% ---- TvType: TV discretization type.
%              1 - anisotropic: |U_x|+|U_y|; 
%              2 - isotropic: sqrt(|U_x|^2+|U_y|^2)

% Output:
% ---- U: deblurred image

% [m, n] = size(Bn);
Wx=W(1:m*n);
Wy=W(m*n+1:end);
%W=reshape(W,2*m,n);
Wx=reshape(Wx,m,n);
Wy=reshape(Wy,m,n);

gamma = beta / mu;
Denom = C.Denom1 + gamma * C.Denom2;  
% u-subproblem
Nomin2 = C.conjoDx .* fft2(Wx) + C.conjoDy.*fft2(Wy);
FU = (C.Nomin1 + gamma * Nomin2) ./ Denom;
U = real(ifft2(FU));
U = reshape(U, m * n, 1);
        

end