load('beta=100.mat');
thre=250;
MaxIters=2000;

figure
semilogy(0:MaxIters,  error_g(1:MaxIters+1), 'r');
hold on
semilogy(0:MaxIters, error_g_LM_AA1, 'g');
hold on
semilogy(0:MaxIters, error_g_LM_AA2, 'b');
hold on
semilogy(0:MaxIters, error_g_LM_AA3, 'c');
 
%legend({'Alternating Direction','LM-AA m=1', 'LM-AA m=3','LM-AA m=5'},'location','NE')
xlabel('Iterations')
ylabel('||f(w)||')
%title('beta = 10')
saveas(gcf,strcat('beta=',num2str(beta),'_iteration.pdf'))

figure
for i=1:length(time)
    if time(i)>thre
        index=i;
        break;
    end
end
semilogy(time(1:index),error_g(1:index),'r');
hold on
index=length(time_LM_AA1);
for i=1:length(time_LM_AA1)
    if time_LM_AA1(i)>thre
        index=i;
        break;
    end
end
semilogy(time_LM_AA1(1:index),error_g_LM_AA1(1:index),'g');
hold on
for i=1:length(time_LM_AA2)
    if time_LM_AA2(i)>thre
        index=i;
        break;
    end
end
semilogy(time_LM_AA2(1:index),error_g_LM_AA2(1:index),'b');
hold on
for i=1:length(time_LM_AA3)
    if time_LM_AA3(i)>thre
        index=i;
        break;
    end
end
semilogy(time_LM_AA3(1:index),error_g_LM_AA3(1:index),'c');
xlabel('Time(s)')
ylabel('||f(w)||')
axis([0,thre,1e-15,1e1]);
saveas(gcf,strcat('beta=',num2str(beta),'_times.pdf'))
 