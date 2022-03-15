function des = RIFT_descriptor_no_rotation_invariance(im, kps,eo, patch_size, s,o,x)

KPS=kps'; %keypoints  ת��
[yim,xim,~] = size(im);

% ����ͼ��I��x��y�����������Ƚ�I��x��y����2D-LGF����Ի����Ӧ����Eso��x��y����Oso��x��y���� Ȼ�������s�ͷ���o�����Aso��x��y��
%%%%%%%%%%%%%
% CS = zeros(yim, xim, o); %convolution sequence   o=6 ��6�������������  s=4
% if(x==1)
%     for j=1:o
% 
% %             CS(:,:,j)=CS(:,:,j)+abs(eo{1,j}); %eo1{s,o} = convolution result for scale s and orientation o.
% %            if (j==1)
% %               imwrite(CS(:,:,j),'I:\2020loggabor\CS11.tif');
% %            end
% %            if (j==2)
% %               imwrite(CS(:,:,j),'I:\2020loggabor\CS12.tif');
% %            end
% %            if (j==3)
% %               imwrite(CS(:,:,j),'I:\2020loggabor\CS13.tif');
% %            end
% %            if (j==4)
% %               imwrite(CS(:,:,j),'I:\2020loggabor\CS14.tif');
% %            end
% %            if (j==5)
% %               imwrite(CS(:,:,j),'I:\2020loggabor\CS15.tif');
% %            end
% %            if (j==6)
% %               imwrite(CS(:,:,j),'I:\2020loggabor\CS16.tif');
% %            end
%            
%             CS(:,:,j)=CS(:,:,j)+abs(eo{4,j}); %eo1{s,o} = convolution result for scale s and orientation o.
%            if (j==1)
%               imwrite(CS(:,:,j)/255,'I:\2020loggabor\pairCS41.tif');
%            end
%            if (j==2)
%               imwrite(CS(:,:,j)/255,'I:\2020loggabor\pairCS42.tif');
%            end
%            if (j==3)
%               imwrite(CS(:,:,j)/255,'I:\2020loggabor\pairCS43.tif');
%            end
%            if (j==4)
%               imwrite(CS(:,:,j)/255,'I:\2020loggabor\pairCS44.tif');
%            end
%            if (j==5)
%               imwrite(CS(:,:,j)/255,'I:\2020loggabor\pairCS45.tif');
%            end
%            if (j==6)
%               imwrite(CS(:,:,j)/255,'I:\2020loggabor\pairCS46.tif');
%            end
% 
%     end
% 
% end
% if(x==2)
%     for j=1:o
% 
% %             CS(:,:,j)=CS(:,:,j)+abs(eo{1,j}); %eo1{s,o} = convolution result for scale s and orientation o.
% %            if (j==1)
% %               imwrite(CS(:,:,j),'I:\2020loggabor\CS11.tif');
% %            end
% %            if (j==2)
% %               imwrite(CS(:,:,j),'I:\2020loggabor\CS12.tif');
% %            end
% %            if (j==3)
% %               imwrite(CS(:,:,j),'I:\2020loggabor\CS13.tif');
% %            end
% %            if (j==4)
% %               imwrite(CS(:,:,j),'I:\2020loggabor\CS14.tif');
% %            end
% %            if (j==5)
% %               imwrite(CS(:,:,j),'I:\2020loggabor\CS15.tif');
% %            end
% %            if (j==6)
% %               imwrite(CS(:,:,j),'I:\2020loggabor\CS16.tif');
% %            end
%            
%             CS(:,:,j)=CS(:,:,j)+abs(eo{4,j}); %eo1{s,o} = convolution result for scale s and orientation o.
%            if (j==1)
%               imwrite(CS(:,:,j)/255,'I:\2020loggabor\pairCS41_.tif');
%            end
%            if (j==2)
%               imwrite(CS(:,:,j)/255,'I:\2020loggabor\pairCS42_.tif');
%            end
%            if (j==3)
%               imwrite(CS(:,:,j)/255,'I:\2020loggabor\pairCS43_.tif');
%            end
%            if (j==4)
%               imwrite(CS(:,:,j)/255,'I:\2020loggabor\pairCS44_.tif');
%            end
%            if (j==5)
%               imwrite(CS(:,:,j)/255,'I:\2020loggabor\pairCS45_.tif');
%            end
%            if (j==6)
%               imwrite(CS(:,:,j)/255,'I:\2020loggabor\pairCS46_.tif');
%            end
% 
%     end
% 
% end
%%%%%%%%%%%%%%%%%%
CS = zeros(yim, xim, o); %convolution sequence   o=6 ��6�������������  s=4
for j=1:o
    for i=1:s
        CS(:,:,j)=CS(:,:,j)+abs(eo{i,j}); %eo1{s,o} = convolution result for scale s and orientation o.
    end
end


[~, MIM] = max(CS,[],3); % MIM maximum index map  %max�ǰ�1->�������ֵ����д���൱����ÿ�е����ֵ %2->�� MIMΪ������
%[C,I]=max(a,[],dim) ���dim=1ʱ��������[c,i]=max(a)��һ���� ��dim=2ʱ i���ص��Ǿ���a���к�
% display(MIM);
%  figure(8), imshow(MIM(:,:)), hold on;

%display(size(CS));

%  figure, imshow(CS), hold on;

des = zeros(36*o, size(KPS,2)); %descriptor (size: 6��6��o)
kps_to_ignore = zeros(1,size(KPS,2));

for k = 1: size(KPS,2)
    x = round(KPS(1, k)); %��������
    y = round(KPS(2, k)); %ȡ��ÿ�е�x��y����
    
    x1 = max(1,x-floor(patch_size/2));  %patch_size=96
    y1 = max(1,y-floor(patch_size/2));
    x2 = min(x+floor(patch_size/2),size(im,2));
    y2 = min(y+floor(patch_size/2),size(im,1));
%     display(x1);
%     display(y1);
%     display(x2);
%     display(y2);
    
    if y2-y1 ~= patch_size || x2-x1 ~= patch_size
        kps_to_ignore(i)=1;
        continue;
    end
  
    patch = MIM(y1:y2,x1:x2); %local MIM patch for feature description
    [ys,xs]=size(patch);
    
    %�ֲ���������Ϊ6��6�������񣬲�Ϊÿ�������񹹽���Nobin�ķֲ�ֱ��ͼ����ΪMIM��ֵ��Χ��1��No. 
    %ͨ����������ֱ��ͼ������������� ��ˣ�����������ά��Ϊ6��6��No��
    ns=6;
    RIFT_des = zeros(ns,ns,o);  %descriptor vector
    
    % histogram vectors
    for j = 1:ns
        for i = 1:ns
            clip = patch(round((j-1)*ys/ns+1):round(j*ys/ns),round((i-1)*xs/ns+1):round(i*xs/ns));  %round ��������
            RIFT_des(j,i,:) = permute(hist(clip(:), 1:o), [1 3 2]);   %���ڽ�������ά�͵ڶ�ά    %N = hist(Y,M),M��һ������������ʹ��M������
        end
    end
   % hist(clip(:), 1:o);
    RIFT_des=RIFT_des(:);
    
    if norm(RIFT_des) ~= 0  %��������һ��
        RIFT_des = RIFT_des /norm(RIFT_des);  
    end
    
    des(:,k)=RIFT_des;
end
des = struct('kps', KPS(:,kps_to_ignore ==0)', 'des', des(:,kps_to_ignore==0)');


