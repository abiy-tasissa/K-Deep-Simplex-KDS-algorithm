% -------------------------------------------------------------------------
% This script saves complexity results for KDS on the two moon
% data and plots visualizations. n is the number of data points,
% t1 is time to obtain the coefficients in KDS and t2 is
% the time to do clustering
% -------------------------------------------------------------------------
figure
subplot(2,2,1)
x = 10000:10000:100000;
t1 = [14.16;27.92;39.7;54.5;66.5;79.91;92.49;107.52;116.13;130.18];
plot(x,t1,'r','LineWidth',2)
grid on
xlabel('Number of points','fontsize',10)
ylabel('Time to obtain coefficients with KDS','fontsize',10)
subplot(2,2,2)
t2 = [0.25;0.41;0.54;0.66;0.85;0.96;1.11;1.24;1.41;1.5];
plot(x,t2,'r','LineWidth',2)
grid on
xlabel('Number of points','fontsize',10)
ylabel('Time to cluster using spectral clustering','fontsize',10)
subplot(2,2,3)
x = 10000:10000:100000;
t1 = [14.16;27.92;39.7;54.5;66.5;79.91;92.49;107.52;116.13;130.18];
loglog(x,t1,'r','LineWidth',2)
grid on
xlabel('log(Number of points)','fontsize',10)
ylabel('log(Time to obtain coefficients with KDS)','fontsize',10)
subplot(2,2,4)
t2 = [0.25;0.41;0.54;0.66;0.85;0.96;1.11;1.24;1.41;1.5];
loglog(x,t2,'r','LineWidth',2)
grid on
xlabel('log(Number of points)','fontsize',10)
ylabel('log(Time to cluster using spectral clustering)','fontsize',10)
