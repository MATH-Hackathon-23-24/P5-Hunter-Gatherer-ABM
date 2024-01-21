%PARAMETERS
%random seed
rng('shuffle')
%landscape
dV=@(x,y)[4.*x.^3-4.*x,7.*y];
V=@(x,y)((x.^2-1).^2+3.5.*y.^2);
%noise parameter
sigma=0.7;
%number of agents
nA=100;
%number of time steps
nT=10000;
%time step size
dt=0.01;
%time
t=0;
%initialize agents
A=4*rand(nA,2)-2;
%variables for saving snapshots
snapshotfreq=100;
Asnapshots=zeros(floor(nT/snapshotfreq),nA,2);
filename='DWABM1'
for countsteps=1:nT
    %POSITION UPDATE
    potforce=-dV(A(:,1),A(:,2));
    randforce=sigma*randn(nA,2);
    poschange=potforce*dt+randforce*sqrt(dt);
    A(:,1:2)=A(:,1:2)+poschange;
    %TIME UPDATE
    t=t+dt;
    %SAVE SYSTEM SNAPSHOT
    if mod(countsteps,snapshotfreq)==0
        Asnapshots(countsteps/snapshotfreq,:,:)=A;
        T(countsteps/snapshotfreq)=t;
    end
end
save(filename,"T","Asnapshots")