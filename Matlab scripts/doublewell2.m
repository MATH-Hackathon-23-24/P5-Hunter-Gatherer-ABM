clear
%PARAMETERS
%random seed
rng('shuffle')
%landscape
V=@(x,y)((x.^2-1).^2+3.5.*y.^2);
x=-2:0.01:2;
y=-1.5:0.01:1.5;
lx=length(x);
ly=length(y);
U=zeros(lx,ly);
insideBound=ones(lx,ly);
for i=1:lx
    for j=1:ly
        U(i,j)=V(x(i),y(j));
        if i==1 || i==lx || j==1 || j==ly
            insideBound(i,j)=0;
        end
    end
end
[dVdx,dVdy]=createfinitediffmatrix(U,insideBound);
dVsize=size(dVdx);
sizeV=dVsize;
%noise parameter
sigma=7;
%number of agents
nA=100;
%number of time steps
nT=10000;
%time step size
dt=0.01;
%time
t=0;
%initialize agents
A=round(200*rand(nA,2))+100;
%variables for saving snapshots
snapshotfreq=100;
Asnapshots=zeros(floor(nT/snapshotfreq),nA,2);
filename='DWABM2'
%scaling paramters
scaletime=1;
scalesuit=100;

for countsteps=1:nT
    %POSITION UPDATE
    roundA=round(A);
    potforce=-(scalesuit*[dVdx(sub2ind(dVsize,roundA(:,1),roundA(:,2))) ,dVdy(sub2ind(dVsize,roundA(:,1),roundA(:,2) ))])*scaletime^2;
    randforce=sigma*randn(nA,2)*scaletime;
    poschange=potforce*dt+randforce*sqrt(dt);
     oldA=A(:,1:2);
    A(:,1:2)=A(:,1:2)+poschange;
    %BOUNDARY CHECK
    %check boundary conditions
    roundA=round(A(1:nA,1:2));
    indices=[find(roundA(:,1)>sizeV(1))',find(roundA(:,1)<=0)',find(roundA(:,2)>sizeV(2))',find(roundA(:,2)<=0)',find(isnan(roundA(:,1)))',find(isnan(roundA(:,2)))']';
    A(indices,1:2)=oldA(indices,:); %reject out of bound steps    
    %check inside bound
    roundA=round(A(1:nA,1:2));
    indicesb=find(insideBound(sub2ind(size(insideBound), roundA(:,1),roundA(:,2) ))==0);
    A(indicesb,1:2)=oldA(indicesb,:); %reject out of bound steps
 
    %TIME UPDATE
    t=t+dt;
    %SAVE SYSTEM SNAPSHOT
    if mod(countsteps,snapshotfreq)==0
        Asnapshots(countsteps/snapshotfreq,:,:)=A;
        T(countsteps/snapshotfreq)=t;
    end
end
save(filename,"T","Asnapshots")