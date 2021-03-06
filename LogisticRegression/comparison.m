clear all;
close all;
clc;

%% Parameters
% dim = 100;
% solution_radius = 1000; % how far is x^* from x0
N = 5000; %  Max Iteration

prob_name       ={'covtype','sido0'};
for i=1:2
%for i=1:length(prob_name)
         dataset_name=prob_name{i};
         s1=load(strcat('./datasets/',dataset_name,'/',dataset_name,'_train_scale.mat'));
         s2=load(strcat('./datasets/',dataset_name,'/',dataset_name,'_labels.mat'));
         A=s1.A;
         b=s2.b;
         m=size(A,1);
         n=size(A,2);
         if issparse(A) 
             nrmA=normest(A);
         else
             nrmA=norm(A);
         end 
         condnum=1e9;
         L_f= nrmA^2/4/m;
         tao = L_f/condnum;
         f = @(w) sum(log(1 + exp(-b .* (A * w))))/m+ tao / 2 * norm(w)^2;
         gradient = @(w) A' * (-b+ b./(1 + exp(-b .* (A*w))))/m +  tao * w;
         L=L_f+tao;
         mu=tao;
         x0 =zeros(n, 1);
         stepSize = 2 / (L+mu);
         c=(L-mu)/(L+mu);
         andersonm=10;
 
        [x,time_gd,error_gd,error_gd_g]=gradient_descent(x0,f,gradient,stepSize,2*N);
          g=@(x) x-stepSize*gradient(x); 
        [x, time_LM_AA1, error_LM_AA1,error_LM_AA_g1,index_LM_fun] =LM_AA(f,gradient, x0, c, stepSize,N,andersonm,1);
 
 
       [x,time_rna5,error_rna5,error_rna_g5]=rna(x0,f,gradient, stepSize, 5, N/5);
       [x,time_rna10,error_rna10,error_rna_g10]=rna(x0,f,gradient, stepSize, 10, N/10);
       [x,time_rna20,error_rna20,error_rna_g20]=rna(x0,f,gradient, stepSize, 20, N/20);
 

optimal_f=min([min(error_LM_AA1),min(error_rna5),min(error_rna10),min(error_rna20)]);
error_gd=(error_gd-optimal_f)/optimal_f;       
error_LM_AA1=(error_LM_AA1-optimal_f)/optimal_f;    
error_rna5=(error_rna5-optimal_f)/optimal_f;    
error_rna10=(error_rna10-optimal_f)/optimal_f;    
error_rna20=(error_rna20-optimal_f)/optimal_f;    
figure

semilogy(time_gd,error_gd,'r');
hold on
semilogy(time_LM_AA1,error_LM_AA1,'g');
hold on
semilogy(time_rna5,error_rna5,'b');
hold on
semilogy(time_rna10,error_rna10,'c');
hold on
semilogy(time_rna20,error_rna20,'y');
hold on
legend({'Gradient Descend', 'LM-AA','RNA k=5','RNA k=10','RNA k=20'},'location','SW')
xlabel('time(s)');
ylabel('(F-F*)/F*');
saveas(gcf,strcat(dataset_name,'_',num2str(condnum),'_times.pdf'))
figure

semilogy(1:N,error_gd(1:N),'r');
hold on
semilogy(1:length(error_LM_AA1),error_LM_AA1,'g');
hold on
semilogy(1:length(error_rna5),error_rna5,'b');
hold on
semilogy(1:length(error_rna10),error_rna10,'c');
hold on
semilogy(1:length(error_rna20),error_rna20,'y');
legend({'Gradient Descend', 'LM-AA','RNA k=5','RNA k=10','RNA k=20'},'location','SW')
xlabel('Iteration');
ylabel('(F-F*)/F*');
saveas(gcf,strcat(dataset_name,'_',num2str(condnum),'_iteration.pdf'));
save(strcat(dataset_name,'condnum=',num2str(condnum) ,'.mat'));
end
 