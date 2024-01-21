clear
%snapshot data
load("DWABM1.mat")
sizeAsnap=size(Asnapshots);
%create background from function
V=@(x,y)((x.^2-1).^2+3.5.*y.^2);
x=-2:0.01:2;
y=-1.5:0.01:1.5;
lx=length(x);
ly=length(y);
U=zeros(lx,ly);
for i=1:lx
    for j=1:ly
        U(i,j)=V(x(i),y(j));
    end
end
%create plots
figure
for i=1:sizeAsnap(1)
    %plot frame background
    contour(x,y,U',0.5:0.5:5)
    %imagesc(x,y,U')
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
vid = VideoWriter('DWABMvid1.avi');
vid.FrameRate = 10;
open(vid)
for i = 1:sizeAsnap(1)
    writeVideo(vid,Frames{i}) 
end
close(vid)