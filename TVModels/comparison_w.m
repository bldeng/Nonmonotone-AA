%% TV model
% Original Problem  min TV(u) + mu/2*||H * u - f||^2 
% Sub Problem min TV(w) + beta/2 * \sum ||w_i-D_i u||^2 + mu/2*||Hs*u-f}}^2

%% Load image
sigma = 5e-2;
H=1;
Bn=imread('man_noise.jpg');
Bn = im2double(Bn);
%% Solver Parameters
[m, n] = size(Bn);
mu=4;
beta = 1e2;
TvType = 2;   % Isotropic TV 
% TVType = 1;   % Anisotropic TV
C = getC(Bn, H);

MaxIters = 2000;

%% Mapping g
f = @(x, Wx, Wy) TV_value(x, Wx, Wy, Bn, mu, beta, C.otfH, TvType);
g = @(x) TV_solver_w(x,m,n, mu, beta, C, TvType);
gu=@(x) TV_solver(x,m,n, mu, beta, C, TvType);
%% Alternating Minimization Method
U0 = Bn;
U0 = reshape(U0, m * n, 1);
W0=UtoW(U0, m,n, beta, TvType);
% U1=gu(U0);
% U2=gu(U1);

W=W0;
% W1=g(W0);
error_g = zeros(1, MaxIters);
time = zeros(1, MaxIters);
total_time = 0;
% uW=UtoW(U0, m,n, beta, TvType);
% fprintf("test1=%d\n",norm(UtoW(WtoU(W0,m,n, mu, beta, C), m,n, beta, TvType)-W0));
% fprintf("test2=%d\n",norm(g(W0)-W0));
% fprintf("test1=%d\n",norm(WtoU(W0,m,n, mu, beta, C)-U1));

for i = 1:2*MaxIters
   
    tic;
    pre_W=W;
    W=g(W);
    t2 = toc;
    
    total_time = total_time + t2;
    
    time(i) = total_time;
  %  fprintf("iteration=%d,res_u=%d\n",i,norm(WtoU(W,m,n, mu, beta, C)-WtoU(pre_W,m,n, mu, beta, C)));
    error_g(i) = norm(W-pre_W);
 %   fprintf('Iterations: %d, Error: %f \n', i, error_g(i));
end
% figure;
% U=reshape(U,m,n);
% imshow(U);
%% LM_AA Method
 

c = 1-1/(1+4*beta/mu);
andersonm = 1;
C1 = 1;


[~, time_LM_AA1, error_g_LM_AA1] = LM_AA(g, W0, c, MaxIters, andersonm, C1);
 
%U = reshape(Bn, m * n, 1);
[~, time_LM_AA2, error_g_LM_AA2] = LM_AA(g, W0, c, MaxIters, andersonm+2, C1);
 
[W, time_LM_AA3, error_g_LM_AA3] = LM_AA(g, W0, c, MaxIters, andersonm+4, C1);

 
U=WtoU(W, m,n, mu, beta, C);

U=reshape(U,m,n);

imwrite(U,strcat('result_','beta=',num2str(beta),'_man.jpg'));


%%
error0 = norm(g(W0) - W0);

time = [0, time];
time_LM_AA1 = [0, time_LM_AA1];
time_LM_AA2 = [0, time_LM_AA2];
time_LM_AA3 = [0, time_LM_AA3];
 
error_g = [error0, error_g];
error_g_LM_AA1 = [error0, error_g_LM_AA1];
error_g_LM_AA2 = [error0, error_g_LM_AA2];
error_g_LM_AA3 = [error0, error_g_LM_AA3];
 
%% Plot

figure
semilogy(0:MaxIters,  error_g(1:MaxIters+1), 'r');
hold on
semilogy(0:MaxIters, error_g_LM_AA1, 'g');
hold on
semilogy(0:MaxIters, error_g_LM_AA2, 'b');
hold on
semilogy(0:MaxIters, error_g_LM_AA3, 'c');
 
legend({'Alternating Direction','LM-AA m=1', 'LM-AA m=3','LM-AA m=5'},'location','NE')
xlabel('Iterations')
ylabel('||f(w)||')
%title('beta = 10')
saveas(gcf,strcat('beta=',num2str(beta),'_iteration.pdf'))
 
figure
semilogy(time,  error_g, 'r');
hold on
semilogy(time_LM_AA1, error_g_LM_AA1, 'g');
hold on
semilogy(time_LM_AA2, error_g_LM_AA2, 'b');
hold on
semilogy(time_LM_AA3, error_g_LM_AA3, 'c');
 
legend({'Alternating Direction','LM-AA m=1', 'LM-AA m=3','LM-AA m=5'},'location','NE')
xlabel('Time(s)')
ylabel('||f(w)||')
saveas(gcf,strcat('beta=',num2str(beta),'_times.pdf'))

save(strcat('beta=',num2str(beta),'.mat'));
%title('beta = 10')





