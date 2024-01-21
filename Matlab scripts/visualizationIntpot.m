clear
%snapshot data
load("intpot1.mat")
filename='ABMvidInt1.avi'
sizeAsnap=size(Asnapshots);
%create plots
figure
for i=1:sizeAsnap(1)
    colormap jet
    %plot agents
    scatter(squeeze(Asnapshots(i,:,1)),squeeze(Asnapshots(i,:,2)),"red","filled")
    %labels
    xlabel('x')
    ylabel('y')
    title(['t=' num2str(T(i))])
    xlim([0 arealength])
    ylim([0 arealength])
    %save frame for video
    drawnow
    F=getframe(gcf);
    %frames for avi
    Frames{i}=F;
end
close
%create avi
vid = VideoWriter(filename);
vid.FrameRate = 10;
open(vid)
for i = 1:sizeAsnap(1)
    writeVideo(vid,Frames{i}) 
end
close(vid)