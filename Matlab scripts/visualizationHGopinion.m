%snapshot data
load("HGop1.mat")
load('gray4.mat')
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
    colormap(gray4)
    hold on
    %plot agents
    colormat=squeeze(Asnapshots(i,1:nA,3:5))./10;
    scatter(squeeze(Asnapshots(i,:,1)),squeeze(Asnapshots(i,:,2)),30,colormat,'filled')
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
vid = VideoWriter('HGop1vid.avi');
vid.FrameRate = 10;
open(vid)
for i = 1:sizeAsnap(1)
    writeVideo(vid,Frames{i}) 
end
close(vid)