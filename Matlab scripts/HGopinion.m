clear
%PARAMETERS
%random seed
rng('shuffle')
%landscape
load("suitlandscapes.mat")
%id of first landscape
potid=1;
%first derivatives
dVdx=squeeze(dVdxs(potid,:,:));
dVdy=squeeze(dVdys(potid,:,:));
%boundary variable
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
intrate=0.01;
%number of agents
nA=300;
%number of agent variables
properties=5;
%number of traits
ntraits=3;
%first order event rates
FOrate=0.001;
lambda1=FOrate*ones(nA,ntraits);
lambda1sum=sum(sum(lambda1));
%time step size
dt=1/12;
%number of time steps
nT=12000*(1/dt);
%time
t=0;
%times for landscape change
changetimes=[2500 5000 7500];
%initialize agents
A=ones(nA,5);
%number of possible opinions per trait
nOp=10;
%define legal starting positions inside boundary
[rId,cId]=find(insideBound);
%initial position distribution
r=ceil(rand(nA,1)*length(rId));
A(1:nA,1:2)=[rId(r),cId(r)];
%initial status distribution
A(:,3:5)=randi([1 nOp],nA,3);
%variables for saving snapshots
snapshotfreq=10*(1/dt);
Asnapshots=zeros(floor(nT/snapshotfreq),nA,5);
filename='HGop1'
%scaling paramters
scaletime=1;
scalesuit=8;
scaleint=1;
%DRAW INITIAL POSITION/STATUS DISTRIBUTION
delay=4000*12;
%draw first waiting time
tau=log(1/rand);
tic
for delaysteps=1:delay
    if mod(delaysteps,(delay/100))==0
        delayprogress=delaysteps/delay
    end
    %calculate neighbor lists
    [H,D]=rangesearch(A(1:nA,1:2),A(1:nA,1:2),interactiondist);
    %init interaction force
    intforce=zeros(nA,2);
     %init lambda2
    lambda2=zeros(nA,ntraits);
    %neighbor computations
    for j=1:nA
        %assign neighbor list
        others=H{j};
        lothers=length(others);
        if lothers>2
            intforce(j,:)=sum(funintforce(vecnorm(A(j,1:2)-A(others(2:lothers),1:2),2,2)).*(A(j,1:2)-A(others(2:lothers),1:2)));
            %adoption rates
            for i=2:lothers
                lambda2(j,1:ntraits)=lambda2(j,1:ntraits)+intrate.*(1-eq(A(others(i),(1+properties-ntraits):properties),A(j,(1+properties-ntraits):properties)));
            end
        end
    end
    %total adoption rates
    lambda2sum=sum(sum(lambda2));
    lambdasum=lambda1sum+lambda2sum;
    if dt*lambdasum<tau
        %POSITION CHANGE
        roundA=round(A(:,1:2));
        potforce=-(scalesuit*[dVdx(sub2ind(dVsize,roundA(:,1),roundA(:,2))) ,dVdy(sub2ind(dVsize,roundA(:,1),roundA(:,2) ))])*scaletime^2;
        randforce=sigma*randn(nA,2)*scaletime;
        poschange=potforce*dt-scaleint*intforce*scaletime*dt+randforce*sqrt(dt);
        %TIME UPDATE
        tau=tau-lambdasum*dt;
    else
        %STATUS UPDATE
        %determine event type
        r0=rand;
        if r0>lambda2sum/lambdasum
            chosenagent=randi([1 nA],1,1);
            chosentrait=randi([3 5],1,1);
            chosenstatus=randi([1 nOp],1,1);
        else
            %transition kernel for agent selection
            Q1=cumsum(sum(lambda2(:,:),2));
            %agent selection
            r1=rand*Q1(nA);
            chosenagent=find(Q1>r1,1);
            %transition kernel for trait selection
            Q2=cumsum(lambda2(chosenagent,:));
            %trait selection
            r2=rand*Q2(3);
            chosentrait=find(Q2>r2,1)+2;
            %transition kernel for status selection
            others=H{chosenagent};
            lothers=length(others);
            statd=statdist(A(others(2:lothers),chosentrait),1:nOp);
            Q3=cumsum(statd);
            %status selection
            r3=rand*Q3(nOp);
            chosenstatus=find(Q3>r3,1);
        end
        %status update
        A(chosenagent,chosentrait)=chosenstatus;
        %POSITION CHANGE
        dtsplit=(tau/lambdasum);
        roundA=round(A(:,1:2));
        potforce=-(scalesuit*[dVdx(sub2ind(dVsize,roundA(:,1),roundA(:,2))) ,dVdy(sub2ind(dVsize,roundA(:,1),roundA(:,2) ))])*scaletime^2;
        randforce=sigma*randn(nA,2)*scaletime;
        poschange=potforce*dtsplit-scaleint*intforce*scaletime*dtsplit+randforce*sqrt(dtsplit);
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
    %init lambda2
    lambda2=zeros(nA,ntraits);
    %neighbor computations
    for j=1:nA
        %assign neighbor list
        others=H{j};
        lothers=length(others);
        if lothers>2
            intforce(j,:)=sum(funintforce(vecnorm(A(j,1:2)-A(others(2:lothers),1:2),2,2)).*(A(j,1:2)-A(others(2:lothers),1:2)));
            %adoption rates
            for i=2:lothers
                lambda2(j,1:ntraits)=lambda2(j,1:ntraits)+intrate.*(1-eq(A(others(i),(1+properties-ntraits):properties),A(j,(1+properties-ntraits):properties)));
            end
        end
    end
    %total adoption rates
    lambda2sum=sum(sum(lambda2));
    lambdasum=lambda1sum+lambda2sum;
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
        %STATUS UPDATE
        %determine event type
        r0=rand;
        if r0>lambda2sum/lambdasum
            chosenagent=randi([1 nA],1,1);
            chosentrait=randi([3 5],1,1);
            chosenstatus=randi([1 nOp],1,1);
        else
            %transition kernel for agent selection
            Q1=cumsum(sum(lambda2(:,:),2));
            %agent selection
            r1=rand*Q1(nA);
            chosenagent=find(Q1>r1,1);
            %transition kernel for trait selection
            Q2=cumsum(lambda2(chosenagent,:));
            %trait selection
            r2=rand*Q2(3);
            chosentrait=find(Q2>r2,1)+2;
            %transition kernel for status selection
            others=H{chosenagent};
            lothers=length(others);
            statd=statdist(A(others(2:lothers),chosentrait),1:nOp);
            Q3=cumsum(statd);
            %status selection
            r3=rand*Q3(nOp);
            chosenstatus=find(Q3>r3,1);
        end
        %status update
        A(chosenagent,chosentrait)=chosenstatus;
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
toc
save(filename,"T","Asnapshots","potIDs")