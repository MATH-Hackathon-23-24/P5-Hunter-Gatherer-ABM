clear
%random seed
rng('shuffle')
%init agents
nA=50;
arealength=30;
A=arealength*rand(nA,2);
sigma=0.1;
%scaling parameters     
scaleint=1;
%interaction potential parameters
%attraction force
CA=0;%1;
%repulsion force
CR=2;
%CR/CA>1
%attraction rate
lA=5;
%repulsion rate
lR=1;
%lR/lA<1
scalex=1;
%interaction force potential
funintforce = @(x)((CA/lA).*exp(-(scalex.*x)/lA)-(CR/lR).*exp(-(scalex.*x)./lR));
%interaction radius
interactiondist=5;
%time step size
dt=0.01;
%time variable
t=0;
%number of time steps
nT=20000;
%variables for saving snapshots
snapshotfreq=100;
Asnapshots=zeros(floor(nT/snapshotfreq),nA,2);
filename='intpot2'
for countsteps=1:nT
    %calculate neighbor lists
    [H,D]=rangesearch(A(1:nA,1:2),A(1:nA,1:2),interactiondist);
    %init interaction force
    intforce=zeros(nA,2);
    %neighbor computations
    for j=1:nA
        %assign neighbor list
        others=H{j};
        lothers=length(others);
        if lothers>2
            intforce(j,:)=sum(funintforce(vecnorm(A(j,1:2)-A(others(2:lothers),1:2),2,2)).*(A(j,1:2)-A(others(2:lothers),1:2)));
        end
    end
    randforce=sigma*randn(nA,2);
    poschange=-scaleint*intforce*dt+randforce*sqrt(dt);
    A(:,1:2)=A(:,1:2)+poschange;
    A=mod(A,arealength);
    %TIME UPDATE
    t=t+dt;
    %SAVE SYSTEM SNAPSHOT
    if mod(countsteps,snapshotfreq)==0
        Asnapshots(countsteps/snapshotfreq,:,:)=A;
        T(countsteps/snapshotfreq)=t;
    end
end
save(filename,"T","Asnapshots","arealength")
