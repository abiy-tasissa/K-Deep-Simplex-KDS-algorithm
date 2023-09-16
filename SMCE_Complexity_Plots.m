% -------------------------------------------------------------------------
% This script saves complexity results for SMCE on the two moon
% data and plots visualizations. n is the number of data points,
% t1 is time to obtain the coefficients in SMCE and t2 is
% the time to do clustering
% -------------------------------------------------------------------------
figure
subplot(2,2,1)
x = 1000:1000:12000;
t1 = [3.04;5.54;11.15;19.40;28.39;43.22;62.93;93.19;125.02;171.90;241.11;401.77];
plot(x,t1,'r','LineWidth',2)
grid on
xlabel('Number of points','fontsize',10)
ylabel('Time to obtain coefficients with SMCE','fontsize',10)
subplot(2,2,2)
t2 = [0.18;1.70;6.86;16.61;32.23;54.25;85.34;126.78;178.40;244.93;324.31; 417.40];
plot(x,t2,'r','LineWidth',2)
grid on
xlabel('Number of points','fontsize',10)
ylabel('Time to cluster using spectral clustering','fontsize',10)
subplot(2,2,3)
x = 1000:1000:12000;
t1 = [3.04;5.54;11.15;19.40;28.39;43.22;62.93;93.19;125.02;171.90;241.11;401.77];
loglog(x,t1,'r','LineWidth',2)
grid on
xlabel('log(Number of points)','fontsize',10)
ylabel('log(Time to obtain coefficients with SMCE)','fontsize',10)
subplot(2,2,4)
t2 = [0.18;1.70;6.86;16.61;32.23;54.25;85.34;126.78;178.40;244.93;324.31; 417.40];
loglog(x,t2,'r','LineWidth',2)
grid on
xlabel('log(Number of points)','fontsize',10)
ylabel('log(Time to cluster using spectral clustering)','fontsize',10)
