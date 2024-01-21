function [D]= statdist(A,strat)
nA = length(A);
D = zeros(length(strat),1);
for i=1:nA
    D(A(i))= D(A(i)) + 1;
end

D= (1/nA)*D;
end