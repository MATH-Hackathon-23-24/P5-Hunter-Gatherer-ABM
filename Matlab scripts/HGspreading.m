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
%interaction potential parameters
%attraction force
CA=0;
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
%spreading rate
intrate=0.002;
%number of agents
nA=300;
%time step size
dt=1/12;
%number of time steps
nT=10100*(1/dt);
%time
t=0;
changetimes=[2500 5000 7500];
%initialize agents
A=ones(nA,3);
%define legal starting positions inside boundary
[rId,cId]=find(insideBound);
%initial position distribution
r=ceil(rand(nA,1)*length(rId)); 
A(1:nA,1:2)=[rId(r),cId(r)];
%variables for saving snapshots
snapshotfreq=10*(1/dt);
Asnapshots=zeros(floor(nT/snapshotfreq),nA,3);
filename='HGspread'
%scaling paramters
scaletime=1;
scalesuit=8;
scaleint=2;
%DRAW INITIAL POSITION DISTRIBUTION
delay=4000*12;
for delaysteps=1:delay
    if mod(delaysteps,(delay/100))==0
        delayprogress=delaysteps/delay
    end
    %calculate neighbor lists
    [H,D]=rangesearch(A(1:nA,1:2),A(1:nA,1:2),interactiondist);
    %init interaction force
    intforce=zeros(nA,2);
    %neighbor computations
    for j=1:nA
        %assign neighbor list
        others=H{j};
        lothers=length(others);
        intforce(j,:)=sum(funintforce(vecnorm(A(j,1:2)-A(others(2:lothers),1:2),2,2)).*(A(j,1:2)-A(others(2:lothers),1:2)));
        %interaction with neighbors
        % for i=2:lothers
        %    %calculate interaction potential for mobility
        %    intforce(j,:)=intforce(j,:)+funintforce(norm(A(j,1:2)-A(others(i),1:2))).*(A(j,1:2)-A(others(i),1:2));
        % end
    end
    %POSITION UPDATE
    roundA=round(A);
    potforce=-(scalesuit*[dVdx(sub2ind(dVsize,roundA(:,1),roundA(:,2))) ,dVdy(sub2ind(dVsize,roundA(:,1),roundA(:,2) ))])*scaletime^2;
    randforce=sigma*randn(nA,2)*scaletime;
    poschange=potforce*dt-scaleint*intforce*scaletime*dt+randforce*sqrt(dt);
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
end

start=1
%SIMULATION START
A(1,3)=2; %introduce second status
tau=log(1/rand);
for countsteps=1:nT
    if mod(countsteps,(nT/100))==0
        progress=countsteps/nT
    end
    %calculate neighbor lists
    [H,D]=rangesearch(A(1:nA,1:2),A(1:nA,1:2),interactiondist);
    %init interaction force
    intforce=zeros(nA,2);
    %init lambda
    lambda=zeros(1,nA);
    %neighbor computations
    for j=1:nA
        %assign neighbor list
        others=H{j};
        lothers=length(others);
        if lothers>2
            intforce(j,:)=sum(funintforce(vecnorm(A(j,1:2)-A(others(2:lothers),1:2),2,2)).*(A(j,1:2)-A(others(2:lothers),1:2)));
            %adoption rates
            if A(j,3)==1
                nS2neighbors=length(find(A(others(2:lothers),3)==2));
                if nS2neighbors>0
                    lambda(j)=intrate*nS2neighbors;
                end
            end
        end
    end
    %total adoption rate
    lambdasum=sum(lambda);
    if dt*lambdasum<tau
        %POSITION CHANGE
        roundA=round(A(:,1:2));
        potforce=-(scalesuit*[dVdx(sub2ind(dVsize,roundA(:,1),roundA(:,2))) ,dVdy(sub2ind(dVsize,roundA(:,1),roundA(:,2) ))])*scaletime^2;
        randforce=sigma*randn(nA,2)*scaletime;
        poschange=potforce*dt-scaleint*intforce*scaletime*dt+randforce*sqrt(dt);
        %TIME UPDATE
        t=t+dt;
        tau=tau-lambdasum*dt;
    else
        %SPREADING
        %transition kernel
        Q=cumsum(lambda);
        %draw agent for status update
        r=rand*lambdasum;
        chosenagent=find(Q>r,1);
        %status update
        A(chosenagent,3)=2;
        %POSITION CHANGE
        dtsplit=(tau/lambdasum);
        roundA=round(A(:,1:2));
        potforce=-(scalesuit*[dVdx(sub2ind(dVsize,roundA(:,1),roundA(:,2))) ,dVdy(sub2ind(dVsize,roundA(:,1),roundA(:,2) ))])*scaletime^2;
        randforce=sigma*randn(nA,2)*scaletime;
        poschange=potforce*dtsplit-scaleint*intforce*scaletime*dtsplit+randforce*sqrt(dtsplit);
        %TIME UPDATE
        t=t+dtsplit;
        %draw new waiting time for next event
        tau=log(1/rand);
    end
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