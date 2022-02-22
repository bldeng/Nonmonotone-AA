function [gw] = TV_solver_w(W, m,n, mu, beta, C, TvType)
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

%w-subproblem
Ux = [diff(U, 1, 2), U(:, 1) - U(:, end)]; %%  filter [1, -1]
Uy = [diff(U, 1, 1); U(1, :) - U(end, :)]; %%  filter [1; -1]

switch TvType
     case 1;
         %|Du|_1=|u_x| + |u_y| 
         Wx = sign(Ux) .* max(abs(Ux) - 1 / beta, 0);
         Wy = sign(Uy) .* max(abs(Uy) - 1 / beta, 0);
      case 2;  
          %\|Du\|=\sqrt(u_x^2+u_y^2)
          V = sqrt(Ux.*Ux + Uy.*Uy);
          S = max(V - 1/beta, 0);
          V(V==0) = 1; 
          S = S./V;
          Wx = S.*Ux; 
          Wy = S.*Uy;
       otherwise; error('TVtype must be 1 or 2');    
end
Wx=reshape(Wx,m*n,1);
Wy=reshape(Wy,m*n,1);
gw=[Wx;Wy];
%gw=reshape(gw,m*n,1);

        

end