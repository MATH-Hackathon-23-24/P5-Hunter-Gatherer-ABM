function [dVdx,dVdy]=createfinitediffmatrix(V,insideBound)
    sizeV=size(V);
    %init variables
    dVdx=zeros(sizeV);
    dVdy=zeros(sizeV);
    for i=1:sizeV(1)
        for j=1:sizeV(2)
            if insideBound(i,j)==1
                dVdx(i,j)=(V(i+1,j)-V(i-1,j))/2;
                dVdy(i,j)=(V(i,j+1)-V(i,j-1))/2;
            end
        end
    end
end