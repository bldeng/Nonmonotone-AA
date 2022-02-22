function [C] =getC(Bn, H)
% compute solver matrix 
% Input:
% ---- Bn: 2D blurred and noisy image
% ---- H:  filter matrix

% Output
% C.conjoDx
% C.conjoDy
% C.Nomin1, 
% C.Denom1
% C.Denom2
sizeB = size(Bn);
otfDx = psf2otf([1,-1], sizeB);
otfDy = psf2otf([1;-1], sizeB);
C.conjoDx = conj(otfDx);
C.conjoDy = conj(otfDy);
otfH  = psf2otf(H,sizeB);
C.Nomin1 = conj(otfH).*fft2(Bn);
C.Denom1 = abs(otfH).^2;
C.Denom2 = abs(otfDx).^2 + abs(otfDy ).^2;
C.otfH = otfH;

end

