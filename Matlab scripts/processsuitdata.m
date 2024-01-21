%load landscape data
V1=csvread("landscape1.csv");
V2=csvread("landscape2.csv");
V3=csvread("landscape3.csv");
V4=csvread("landscape4.csv");
sizeV=size(V1);
Vs=zeros(4,sizeV(1),sizeV(2));
sizeVs=size(Vs);
dVdxs=zeros(sizeVs);
dVdys=zeros(sizeVs);
%switch "mountains" and "valleys" and combine them into 1 variable
Vs(1,:,:)=1-V1;
Vs(2,:,:)=1-V2;
Vs(3,:,:)=1-V3;
Vs(4,:,:)=1-V4;
%set numeric value to assign to boundary values
m=2;
insideBound=ones(sizeV);
%define boundary and set boundary values to maximum 
for i=1:sizeVs(1)
    for j=1:sizeVs(2)
        for k=1:sizeVs(3)
            %grid boundary
            if j==1 || j== sizeV(1) || k==1 || k==sizeV(2) || isnan(Vs(i,j,k))
                Vs(i,j,k) = m;
                insideBound(j,k)=0;
            end
        end
    end
    [dVdx,dVdy]=createfinitediffmatrix(squeeze(Vs(i,:,:)),insideBound);
    dVdxs(i,:,:)=dVdx;
    dVdys(i,:,:)=dVdy;
end


save("suitlandscapes","insideBounds","Vs","dVdxs","dVdys")



