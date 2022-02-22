function  [x_result,time_t, error_g] = LM_AA(g, x0, c, N, andersonm,C1)
% Input:
% gradval: gradient of objective function
% x0: initial x
% c: Lipschitz constant
% step_length:
% xstar: ground truth
% N :max iteration

% Output:
% x_result: final result
% error: norm(x - xstar)

n_dim = length(x0);

% Solver Parameters
 delta = 2;
 mu_min = 0;
 mu = 1;
 
 p1 = 0.01;
 p2 = 0.25;
 
 
 constant=1;

 eta0 = 2;
 eta1 = 0.25;
 
 m = andersonm+1;
 
 % Initialization
 X = zeros(n_dim, m);
 G = zeros(n_dim, m);
 F = zeros(n_dim, m);  % F = G - X
 M = zeros(m , m);     % M = F' * F
 Fnorm=zeros(1,m);
 x = x0;
 total_time = 0;
 count = 0;
 idx = 0;
 
 for i = 1:N
     tic;
     if(idx == 0)
         X(:, 1) = x;
         curr_g = g(x);
         G(:, 1) = curr_g;
         tempf=curr_g-x;
         temp=sum(tempf.^2);
         Fnorm(1)=temp ;
         F(:, 1) =(tempf)/(sqrt(Fnorm(1)));
         M(1, 1) = 1;
         x = G(:, 1);
         curr_g = g(x);
         idx=idx+1;
         m_hat = min(idx + 1, m);
         id = mod(idx, m) + 1;
         X(:, id) = x;
         G(:, id) = curr_g;
         tempf=curr_g-x;
         temp=sum(tempf.^2);
         Fnorm(id)=temp ;
         F(:, id) = (tempf)/(sqrt(Fnorm(id))); 
         t2=toc;
         total_time=total_time+t2;
         time_t(i)=total_time;
         error_g(i)=norm(curr_g-x);
         continue;
     end     
     if id>1
        M(1:id-1, id) = F(:, 1:id-1)' * F(:, id);
     end
     if id<m_hat
        M(id+1:m_hat, id) = F(:, id+1:m_hat)' * F(:, id);
     end
     M(id,id) = 1;
     M(id, 1:m_hat) = M(1:m_hat, id)';  
     % Find k0      
  %   tempF = sum(F(:, 1:m_hat).^2, 1);
     [tF_norm, index] = sort(Fnorm(:, 1:m_hat));
     %normal_Fnorm=||f_{k_i}||/||f_{k_0}||

     
     k0 = index(1);
     kmax = index(1);
     normal_Fnorm=sqrt(Fnorm)/sqrt(Fnorm(k0));
     % set lambda_k
     %f_k0 = F(:, k0);
  %   lambda = mu * min(constant*(tF_norm(1))^(delta/2), C1);
     lambda=mu; 
     tM=diag(normal_Fnorm)*M*diag(normal_Fnorm);
     % solve alpha
     bb = tM(:, k0);
     B = repmat(bb(1:m_hat), 1, m_hat);
     D = tM(1:m_hat, 1:m_hat) +1- B - B';
     D(:, k0) = [];
     D(k0, :) = [];
     
     b = 1 - bb(1: m_hat);
     b(k0) = [];
     
     A = D + lambda * eye(m_hat - 1);
    %alpha = A \ b;  
     alpha = lsqminnorm(A, b);

     % set gamma_k
     gamma = 1e-4 * ones(m_hat, 1);
     gamma(kmax) = 1 - (m_hat - 1) * 1e-4;
     
     % compute pred & ared & rho = ared / pred
     sum_f = Fnorm(:, 1:m_hat) * gamma;
    
     g_k0 = G(:, k0);
     
     temp_alpha = [alpha(1:k0 - 1); 1 - sum(alpha); alpha(k0:end)];
     %x_hat = X(:, 1:m_hat) * temp_alpha;
     g_hat = G(:, 1:m_hat) * temp_alpha;
     %caculate the norm of f_hat using some tricks
     descent=alpha'*D*alpha-2*b'*alpha;
     normf_hat=tF_norm(1)*(1+descent);
     
     trial_g = g(g_hat);
     trial_f=trial_g - g_hat;
     trial_Fnorm=sum((trial_g - g_hat).^2);
     pred = sum_f - c * c * normf_hat;
     ared = sum_f - trial_Fnorm;
     rho = ared / pred;
     
     
     
     
     % update mu
     if(rho < p1)
         mu = eta0 * mu;
     elseif (rho > p2)
         mu = max(eta1 * mu, mu_min);
     end
    % fprintf("iteration %d, rho=%d, residual=%d,lambda=%d,conditional numer=%d\n",i,rho,norm(trial_g-g_hat),lambda,sqrt(max(F_norm)/min(F_norm)));
     % update x
     if(rho <= p1)
         x = g_k0;
         curr_g = g(g_k0);
       %  idx=idx+1;
          idx=idx+1;
         m_hat = min(idx + 1, m);
         id = mod(idx, m) + 1;
         X(:, id) = x;
         G(:, id) = curr_g;
         tempf=curr_g-x;
         temp=sum(tempf.^2);
         Fnorm(id)=temp ;
         F(:, id) = (tempf)/(sqrt(Fnorm(id))); 
     else
         x = g_hat; 
         curr_g = trial_g;
         idx = idx + 1;
         m_hat = min(idx + 1, m);
         id = mod(idx, m) + 1;
         X(:, id) = x;
         G(:, id) = curr_g;
         Fnorm(id)=trial_Fnorm ;
         F(:, id) =trial_f/sqrt(Fnorm(id));
         count = count + 1;
     end
     t2=toc;
     total_time=total_time+t2;
     time_t(i)=total_time;
     error_g(i) = norm(curr_g - x);
%      
%      if i > 20 && abs(error(i) - error(i-20)) < 1e-20
%          break;
%      end
 end
% fprintf("successful stpes=%d\n",count);
 x_result = x;
end


