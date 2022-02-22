function [grad_value] = TV_gradient(image, mu, image0, H)
% compute TV gradient
% Input
% image :current image // column vector
% mu : regularized parameters
% image0: observed image

 [m, n] = size(image0);
 image = reshape(image, m, n);

 Dx = [diff(image, 1, 2), image(:, 1) - image(:, end)]; %% filter [1, -1]
 Dy = [diff(image, 1, 1); image(1, :) - image(end, :)]; %% filter [1; -1]

 Dxy_norm = sqrt(Dx.^2 + Dy.^2);

 Dx_r = [Dx(:, end), Dx(:, 1:end - 1)];
 Dy_r = [Dy(:, end), Dy(:, 1:end - 1)];

 Dx_d = [Dx(end, :); Dx(1:end - 1, :)];
 Dy_d = [Dy(end, :); Dy(1:end - 1, :)];

 Dxy_r_norm = sqrt(Dx_r.^2 + Dy_r.^2);
 Dxy_d_norm = sqrt(Dx_d.^2 + Dy_d.^2);

 grad_value = Dx_r ./ Dxy_r_norm + Dy_d ./ Dxy_d_norm - (Dx + Dy) ./ Dxy_norm + mu * imfilter(imfilter(image, H) - image0, H);

 grad_value = reshape(grad_value, m*n,1);

end