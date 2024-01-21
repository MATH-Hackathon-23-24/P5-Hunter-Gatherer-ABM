clear
%snapshot data
load("HGABM1.mat")
filename='HGMvid1.avi'
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
    contour(x,y,V',0.2:0.1:1)
    colormap jet
    hold on
    %plot agents
    scatter(squeeze(Asnapshots(i,:,1)),squeeze(Asnapshots(i,:,2)),"red","filled")
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
    %images for gif
    im{i}=frame2im(F);
end
close
%create gif
start=0;
for i= 1:sizeAsnap(1)
    [B,map]=rgb2ind(im{i},256);
    if start==0
        imwrite(B,map,'ABMvid1.gif','gif','LoopCount',Inf,'DelayTime',0.01);
        start=1;
    else
        imwrite(B,map,'ABMvid1.gif','gif','WriteMode','append','DelayTime',0.01);
    end
end
%create avi
vid = VideoWriter(filename);
vid.FrameRate = 10;
open(vid)
for i = 1:sizeAsnap(1)
    writeVideo(vid,Frames{i}) 
end
close(vid)