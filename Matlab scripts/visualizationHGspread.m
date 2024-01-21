%snapshot data
load("HGspread.mat")
sizeAsnap=size(Asnapshots);
%load landscapes
load("suitlandscapes.mat")
V=squeeze(Vs(1,:,:));
sizeV=size(V);
x=1:sizeV(1);
y=1:sizeV(2);
%create plots
figure
for i=1:sizeAsnap(1)
    %plot frame background
    contour(x,y,squeeze(Vs(potIDs(i),:,:))',0.2:0.1:1)
    
    colormap jet
    hold on
    %plot agents
    s1agents=find(Asnapshots(i,:,3)==1);
    s2agents=find(Asnapshots(i,:,3)==2);
    scatter(squeeze(Asnapshots(i,s1agents,1)),squeeze(Asnapshots(i,s1agents,2)),"blue","filled")
    scatter(squeeze(Asnapshots(i,s2agents,1)),squeeze(Asnapshots(i,s2agents,2)),"red","filled")
    %labels
    xlabel('x')
    ylabel('y')
    title(['t=' num2str(T(i))])
    hold off
    colorbar
    %save frame for video
    drawnow
    F=getframe(gcf);
    %frames for avi
    Frames{i}=F;
end
close
%create avi
vid = VideoWriter('HGspreadvid.avi');
vid.FrameRate = 10;
open(vid)
for i = 1:sizeAsnap(1)
    writeVideo(vid,Frames{i}) 
end
close(vid)