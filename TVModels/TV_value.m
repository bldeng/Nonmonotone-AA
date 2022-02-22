function [value] = TV_value(U, Wx, Wy, Bn, mu, beta, otfH, TvType)
% Compute TV function value
% Input£º
% ---- U: current image
% ---- Wx, Wy
% ---- Bn: 2D blurred and noisy image
% ---- mu: paramter
% ---- beta: penalty parameter
% ---- otfH :fourier transform for H
% ---- TvType: TV discretization type.
%              1 - anisotropic: |U_x|+|U_y|; 
%              2 - isotropic: sqrt(|U_x|^2+|U_y|^2)

% Output:
% ---- value: function value

Ux = [diff(U, 1, 2), U(:, 1) - U(:, end)]; %%  filter [1, -1]
Uy = [diff(U, 1, 1); U(1, :) - U(end, :)]; %%  filter [1; -1]
KXF = real(ifft2(otfH .* fft2(U))) - Bn;



switch TvType
    case 1;
        value = sum(sum(abs(Wx) + abs(Wy))) +  beta / 2 * sum(sum((Wx - Ux).^2 + (Wy - Uy).^2)) + mu / 2 * norm(KXF,'fro')^2;
    case 2;
        value = sum(sum(sqrt(Wx.^2 + Wy.^2))) + beta / 2 * sum(sum((Wx - Ux).^2 + (Wy - Uy).^2)) +  mu / 2 * norm(KXF,'fro')^2;
end

