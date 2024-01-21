clear
%PARAMETERS
%random seed
rng('shuffle')
%landscape
load("suitlandscapes.mat")
potid=1;
dVdx=squeeze(dVdxs(potid,:,:));
dVdy=squeeze(dVdys(potid,:,:));
insideBound=squeeze(insideBounds(potid,:,:));
dVsize=size(dVdx);
sizeV=dVsize;
%noise parameter
sigma=1;
%number of agents
nA=300;
%time step size
dt=1/12;
%number of time steps
nT=10000*(1/dt);
%time
t=0;
changetimes=[2500 5000 7500];
%initialize agents
%define legal starting positions inside boundary
[rId,cId]=find(insideBound);
%initial position distribution
r=ceil(rand(nA,1)*length(rId)); 
A(1:nA,1:2)=[rId(r),cId(r)];
%variables for saving snapshots
snapshotfreq=10*(1/dt);
Asnapshots=zeros(floor(nT/snapshotfreq),nA,2);
filename='HGABM2'
%scaling paramters
scaletime=1;
scalesuit=8;

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
    %LANDSCAPE UPDATE
    if potid <4
        if t>changetimes(potid)
            potid=potid+1;
            dVdx=squeeze(dVdxs(potid,:,:));
            dVdy=squeeze(dVdys(potid,:,:));
        end
    end
    %SAVE SYSTEM SNAPSHOT
    if mod(countsteps,snapshotfreq)==0
        Asnapshots(countsteps/snapshotfreq,:,:)=A;
        T(countsteps/snapshotfreq)=t;
        potIDs(countsteps/snapshotfreq)=potid;
    end
end
save(filename,"T","Asnapshots","potIDs")