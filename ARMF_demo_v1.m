
%% This is the main code ARMF_demo of the proposed ARMF algorithm.
clc;clear;close all;
warning('off')

%-------
%APAP: Paths.
%-------
addpath('modelspecific');
addpath('mexfiles'); 
addpath('multigs');

%-------------------
%APAP: Compile Mex files.
%-------------------

cd multigs;
if exist('computeIntersection','file')~=3
    mex computeIntersection.c; % <-- for multigs
end
cd ..;

cd mexfiles;
if exist('imagewarping','file')~=3
    mex ../imagewarping.cpp; 
end
if exist('wsvd','file')~=3
    mex ../wsvd.cpp; % We make use of eigen3's SVD in this file.
end
cd ..;

%----------------------
% Setup VLFeat toolbox.
%----------------------
cd vlfeat-0.9.14/toolbox;
feval('vl_setup');
cd ../..;

%---------------------------------------------
% Check if we are already running in parallel.
%---------------------------------------------
% poolsize = matlabpool('size');
% if poolsize == 0 %if not, we attempt to do it:
%     matlabpool open;
% end

%-------------------------
% User defined parameters.
%-------------------------
% Global model specific function handlers.
clear global;
global fitfn resfn degenfn psize numpar
fitfn = 'homography_fit';
resfn = 'homography_res';
degenfn = 'homography_degen';
psize   = 4;
numpar  = 9;

M     = 500;  % Number of hypotheses for RANSAC.
thr   = 0.1;  % RANSAC threshold.

% APAP:Resolution/grid-size for the mapping function in MDLT (C1 x C2).
C1 = 100; 
C2 = 100;

fprintf('images/n');    
gamma = 0.0015; 
sigma = 12; 
    
% Load images and SIFT matches for railtracks data.
load 'SIFTdata/railtracks.mat'    
% ARMF_demo.m parameter
error_region=100;
thr_matching_error=10;
num_trans=8;
restricted_region_size=1400; 
ARMF=1;

I1_h=0;
I2_h=0;
I1_w=0;
I2_w=0;
row=0;
col=0;
%%%%%%%%%%%%%%%%%%%
% *** IMPORTANT ***
%%%%%%%%%%%%%%%%%%%
% If you want to try with your own images and make use of the VLFEAT
% library for SIFT keypoint detection and matching, **comment** the 
% previous IF/ELSE STATEMENT and **uncomment** the following code:
% 
 gamma = 0.01; % Normalizer for Moving DLT. (0.0015-0.1 are usually good numbers).
 sigma = 8.5;  % Bandwidth for Moving DLT. (Between 8-12 are good numbers).   
 scale = 0.5;    % Scale of input images (maybe for large images you would like to use a smaller scale).

str1='/home/fly/Desktop/pic/hhjc2013.idx_106719_22482_17.tif';   % image pair
str2='/home/fly/Desktop/pic/hhjc2015.idx_106719_22482_17.tif';

str3=str1;


I2=imread(str2);
I1=imread(str1);
I3=imread(str3);
display(size(I2));
display(size(I1));
[I1_h,I1_w,I1_c]=size(I1);
[I2_h,I2_w,I2_c]=size(I2);
%%
%ARMF_prepare:copyMakeBorder
[I1,I2] = ARMF_prepare(I1_h,I2_h,I1_w,I2_w,I1,I2);

I_1=I1;
I_2=I2;
I2_global=I2;
I1_global=I1;

%% RIFT 

AA = max(size(I1,1),size(I2,1));
BB = max(size(I1,2),size(I2,2));

im1 = im2uint8(I1);
im2 = im2uint8(I2);
im3 = im2uint8(I3);

if size(im1,3)==1
    temp=im1;
    im1(:,:,1)=temp;
    im1(:,:,2)=temp;
    im1(:,:,3)=temp;
end

if size(im2,3)==1
    temp=im2;
    im2(:,:,1)=temp;
    im2(:,:,2)=temp;
    im2(:,:,3)=temp;
end
%%
%%%%%   save global matching feature  %%%%%
ref_total_phase_feature=[];
tar_total_phase_feature=[];
ref_total_sift_feature=[];
tar_total_sift_feature=[];
ref_total_feature=[];
tar_total_feature=[];
ref_total_phaseout_feature=[];
tar_total_phaseout_feature=[];
ref_total_siftout_feature=[];
tar_total_siftout_feature=[];
ref_totalout_feature=[];
tar_totalout_feature=[];
ref_total_lsd_feature=[];
tar_total_lsd_feature=[];
ref_sift1_out_sec=[];
tar_sift1_out_sec=[];
cleanedPoints3_lsd_out_sec=[];
cleanedPoints4_lsd_out_sec=[];
cleanedPoints1_lsd_out_sec=[];
cleanedPoints2_lsd_out_sec=[];
ref_total_lsdout_feature=[];
tar_total_lsdout_feature=[];
ref_lsdout_feature=[];
tar_lsdout_feature=[];
ref_phase_out_sec=[];
tar_phase_out_sec=[];
cleanedPoints1_e_inliers=[];
cleanedPoints2_e_inliers=[];
cleanedPoints1_lsd_e_inliers=[];
cleanedPoints2_lsd_e_inliers=[];
outcleanedPoints3_sec=[];
outcleanedPoints4_sec=[];
ref_phaseout_feature=[];
tar_phaseout_feature=[];
ref_siftout_feature=[];
tar_siftout_feature=[];
ref_new_region_sift=[];
ref_new_region_phase=[];
ref_new_region_lsd=[];
tar_new_region_sift=[];
tar_new_region_phase=[];
tar_new_region_lsd=[];
ref_new_region_sift_x=[];
ref_new_region_phase_x=[];
ref_new_region_lsd_x=[];
tar_new_region_sift_x=[];
tar_new_region_phase_x=[];
tar_new_region_lsd_x=[];
ref_new_region_sift_y=[];
ref_new_region_phase_y=[];
ref_new_region_lsd_y=[];
tar_new_region_sift_y=[];
tar_new_region_phase_y=[];
tar_new_region_lsd_y=[];
ref_new_region_phase_feature=[];
tar_new_region_phase_feature=[];
ref_new_region_lsd_feature=[];
tar_new_region_lsd_feature=[];
ref_new_region_sift_feature=[];
tar_new_region_sift_feature=[];
total_ref_new_region_sift=[];
total_tar_new_region_sift=[];
total_ref_new_region_phase=[];
total_tar_new_region_phase=[];
total_ref_new_region_lsd=[];
total_tar_new_region_lsd=[];
ref_total_=0;
tar_total_=0;
total_new_region_sift=[];
total_new_region_phase=[];
total_new_region_lsd=[];
ref_total_sift=[];
ref_total_phase=[];
tar_total_sift=[];
tar_total_phase=[];
seg_new_l=[];
seg_new_r=[];
%%    blocks    

L = size(I1);
R = size(I2);
%ARMF:Automatically adjust the area block size according to the size of the
%image
if(((I1_h>20000)||(I1_w>20000)||(I2_h>20000)||(I2_w>20000)))

    height=3*1600;

    width=3*1600;
else
    height=2*1200;

    width=2*1200;
end
%%
%ARMF:overlapping regions parameter
x=0;
%ARMF:Region blocks extartion for target and reference images
[max_row,max_col,r_max_row,r_max_col,w_val,wr_val,h_val,hr_val] = ARMF_blocks(height,width,L(1),L(2),R(1),R(2),x);

%ARMF:cell for reference and target images
[seg] = ARMF_cell(max_row,max_col,width,height,w_val,h_val,L(1),L(2),I1);
[seg_r] = ARMF_cell_r(r_max_row,r_max_col,width,height,wr_val,hr_val,R(1),R(2),I2,L(1));

imshow([I1 I2]);

hold on

for i_r=1:r_max_row

for j_r=1:r_max_col

end

end


for row_r = 1:r_max_row

for col_r = 1:r_max_col

c=rand(1,3);

rectangle ('Position',[wr_val*(col_r-1),hr_val*(row_r-1),width,height],'LineWidth',2,'LineStyle','-','EdgeColor',c );

end

end
%%
riftnum=0;
num_figure=0;
apapnum=0;
num_plot=0;
num_sift_in_out=0;
ref_sift1_e_inliers_num=0;
num_fsc_ref_sift1=0;
num_bf_fsc_ref_sift=0;
num_ref_phase=0;
num_ref_phase_fsc=0;
ref_phase_e_inliers_num=0;
num_ref_lsd=0;
ref_lsd_e_inliers_num=0;
error_res_lsd=0;
error_res_sift=0;
error_res_phase=0;
ref_phase_e_inliers_num_kuozeng=0;
ref_sift1_e_inliers_num_kuozeng=0;

%%

for i=1:max_row

for j=1:max_col

end

end

% ARMF: blocks
for i_ = 1:max_row

for j_ = 1:max_col
    
    
resize_I1=seg{i_,j_};
resize_I2=seg_r{i_,j_};


c=rand(1,3);

rectangle ('Position',[w_val*(col-1),h_val*(row-1),width,height],'LineWidth',2,'LineStyle','-','EdgeColor',c );

% ARMF:nums
first_num=0;
num=0;

[num,first_num,seg{i_,j_},seg_r{i_,j_}] = ARMF_down(num,first_num,seg{i_,j_},seg_r{i_,j_},restricted_region_size);


% ARMF:Multiple features extration for image regions.
% SIFT and RIFT has robust and excellent feature extraction and feature
% matching capabilities. The algorithm references are from the following:
% Li J, Hu Q, Ai M. RIFT: Multi-modal image matching based on radiation-variation insensitive feature transform[J]. 
% IEEE Transactions on Image Processing, 2019, 29: 3296-3310.
% Lowe D G. Distinctive image features from scale-invariant keypoints[J]. 
% International journal of computer vision, 2004, 60(2): 91-110.

[inliers,outliers,data_orig,cleanedPoints1,cleanedPoints2,cleanedPoints3,cleanedPoints4,outedPoints1,outedPoints2,outedPoints3,outedPoints4,cleanedPoints1_lsd,cleanedPoints2_lsd,cleanedPoints3_lsd,cleanedPoints4_lsd,outedPoints1_lsd,outedPoints2_lsd,outedPoints3_lsd,outedPoints4_lsd,ref_sift1,tar_sift1,ref_sift_1,tar_sift_1,ref_sift1_out,tar_sift1_out,ref_sift1_out_1,tar_sift1_out_1]=ARMF_Multifeatures(seg{i_,j_},seg_r{i_,j_},thr,M,num);


    [mm_a,mm_b]=size(cleanedPoints1);
    [nn_a,nn_b]=size(cleanedPoints2);
    [mm_a_lsd,mm_b_lsd]=size(cleanedPoints1_lsd);
    [nn_a_lsd,nn_b_lsd]=size(cleanedPoints2_lsd);

    [asift,bsift]=size(ref_sift1);
    [aphase,cphase]=size(cleanedPoints3);
    [alsd,clsd]=size(cleanedPoints3_lsd);
     
      if (size(cleanedPoints2,1)>num_trans)
          [tform_phase]=estimateGeometricTransform(cleanedPoints2,cleanedPoints1,'similarity');
          squarephasesum=[];
          ref_phase_save=[];
          tar_phase_save=[];
          phase_after_tar_save=[];

        % ARMF:FME
          [squarephasesum,error_res_phase] = ARMF_FME(aphase,cleanedPoints1,cleanedPoints2,tform_phase,ref_phase_save,tar_phase_save,phase_after_tar_save);
      end
  
    
      if (size(ref_sift1,1)>num_trans)
         [tform_sift]=estimateGeometricTransform(ref_sift1,tar_sift1,'affine'); 
         % cal sift error
         squaresiftsum=[];
         ref_sift_save=[];
         tar_sift_save=[];
         sift_after_tar_save=[];
                
        % ARMF:FME
        [squaresiftsum,error_res_sift] = ARMF_FME(asift,ref_sift1,tar_sift1,tform_sift,ref_sift_save,tar_sift_save,sift_after_tar_save);        
     end

       if (size(cleanedPoints2_lsd,1)>num_trans)
          [tform_lsd]=estimateGeometricTransform(cleanedPoints2_lsd,cleanedPoints1_lsd,'similarity');
     
        % cal
          squarelsdsum=[];
          ref_lsd_save=[];
          tar_lsd_save=[];
          lsd_after_tar_save=[];
        % ARMF:FME
        [squarelsdsum,error_res_lsd] = ARMF_FME(alsd,cleanedPoints1_lsd,cleanedPoints2_lsd,tform_lsd,ref_lsd_save,tar_lsd_save,lsd_after_tar_save);
       end
       
        %ARMF:show matching result
        [fig,~] = ARMF_show(figure,seg{i_,j_},seg_r{i_,j_},asift,ref_sift1,tar_sift1,mm_a_lsd,cleanedPoints1_lsd,cleanedPoints2_lsd,mm_a,cleanedPoints1,cleanedPoints2,error_res_phase,error_res_sift,error_res_lsd);
        

        if ARMF==1
            %ARMF:AR
             [cleanedPoints3,cleanedPoints4,ref_sift_1,tar_sift_1,cleanedPoints3_lsd,cleanedPoints4_lsd,num,ref_new_region_phase_feature,tar_new_region_phase_feature,ref_new_region_sift_feature,tar_new_region_sift_feature,ref_new_region_lsd_feature,tar_new_region_lsd_feature] = ARMF_AR(thr_matching_error,error_res_phase,cleanedPoints1,cleanedPoints2,error_res_sift,ref_sift1,tar_sift1,error_res_lsd,cleanedPoints1_lsd,cleanedPoints2_lsd,num,max_col,max_row,i_,j_,seg,seg_r,first_num,M,thr,w_val,h_val,wr_val,hr_val,width,height,x,num_trans,error_region,thr_matching_error,total_ref_new_region_phase,total_tar_new_region_phase,total_ref_new_region_sift,total_tar_new_region_sift,total_ref_new_region_lsd,total_tar_new_region_lsd);
        end
        

        [ref_total_phase_feature,tar_total_phase_feature,ref_total_sift_feature,tar_total_sift_feature,ref_total_lsd_feature,tar_total_lsd_feature]=ARMF_AR_re(num,first_num,cleanedPoints3,cleanedPoints4,ref_sift_1,tar_sift_1,cleanedPoints3_lsd,cleanedPoints4_lsd,w_val,h_val,wr_val,hr_val,width,height,i_,j_,x,ref_total_phase_feature,tar_total_phase_feature,ref_total_sift_feature,tar_total_sift_feature,ref_total_lsd_feature,tar_total_lsd_feature);

    
end

end


    %%%%%%   plot total feature on %%%%%%

    total_ref_feature=[ref_total_phase_feature;ref_total_sift_feature;ref_total_lsd_feature;total_ref_new_region_sift;total_ref_new_region_phase;total_ref_new_region_lsd];%;
    total_tar_feature=[tar_total_phase_feature;tar_total_sift_feature;tar_total_lsd_feature;total_tar_new_region_sift;total_tar_new_region_phase;total_tar_new_region_lsd];%;
% 
%                    
%% ARMF: transform result
        if ARMF==1   
        atotal=size(total_ref_feature,1);
        % cal 
        squaretotalsum=[];
        ref_total_save=[];
        tar_total_save=[];
        total_after_tar_save=[];
        [total_ref_feature,total_tar_feature]=ARMF_cor(atotal,total_ref_feature,total_tar_feature,ref_total_save,tar_total_save,total_after_tar_save,total_ref_feature,total_tar_feature);
        end        
        %ARMF_show
        %%%%%   total feature on global %%%%%%%  
        fig1= figure(1111);

        ARMF_finalres(ref_total_phase_feature,tar_total_phase_feature,ref_total_sift_feature,tar_total_sift_feature,ref_total_lsd_feature,tar_total_lsd_feature,total_ref_new_region_sift,total_tar_new_region_sift,total_ref_new_region_phase,total_tar_new_region_phase,total_ref_new_region_sift,total_tar_new_region_sift,I1_global,I2_global);
     
        disp('ARMF registration result')
        
%% ARMF: transform result
         if (ARMF==1) 

            ref_total_sift=[ref_total_sift_feature;total_ref_new_region_sift];
            tar_total_sift=[tar_total_sift_feature;total_tar_new_region_sift];
             atotalsift=size(ref_total_sift,1);
             % cal
             squaretotalsiftsum=[];
             ref_totalsift_save=[];
             tar_totalsift_save=[];
             totalsift_after_tar_save=[];

             [ref_total_sift,tar_total_sift]=ARMF_FMEres(atotalsift,ref_total_sift,tar_total_sift,ref_totalsift_save,tar_totalsift_save,totalsift_after_tar_save);

         end

%% ARMF: transform result
         if (ARMF==1) 
             %%%%%%%%%%

            ref_total_phase=[ref_total_phase_feature;total_ref_new_region_phase];
            tar_total_phase=[tar_total_phase_feature;total_tar_new_region_phase];
            atotalphase=size(ref_total_phase,1);

            squaretotalphasesum=[];
            ref_totalphase_save=[];
            tar_totalphase_save=[];
            totalphase_after_tar_save=[];

            [ref_total_phase,tar_total_phase]=ARMF_FMEres(atotalphase,ref_total_phase,tar_total_phase,ref_totalphase_save,tar_totalphase_save,totalphase_after_tar_save);

         end
        

%% ARMF + APAP
    % The APAP has strong image stitching performance.
    % The algorithm reference from the following:
    % Zaragoza, J., Chin, T. J., Brown, M. S., & Suter, D. (2013). As-projective-as-possible image stitching with moving DLT. 
    % In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2339-2346).
  %%%%%%%%%%%   ARMF  + APAP    %%%%%%%%%%%%%%

    total_sift_dorm=[total_ref_feature';ones(1,size(total_ref_feature,1));total_tar_feature';ones(1,size(total_tar_feature,1))];
    [ dat_norm_I1,T_1 ] = normalise2dpts(total_sift_dorm(1:3,:));
    [ dat_norm_I2,T_2 ] = normalise2dpts(total_sift_dorm(4:6,:));
    sift_data_norm = [ dat_norm_I1 ; dat_norm_I2 ];
    %-----------------------
    % Global homography (H).
    %-----------------------
    fprintf('DLT (projective transform) on inliers/n');
    % Refine homography using DLT on inliers.
    fprintf('> Refining homography (H) using DLT...');tic;
    [ h,A,D1,D2 ] = feval(fitfn,sift_data_norm);
    Hg = T_2\(reshape(h,3,3)*T_1);
    fprintf('done (%fs)/n',toc);

    %----------------------------------------------------
    % Obtaining size of canvas (using global Homography).
    %----------------------------------------------------
    fprintf('Canvas size and offset (using global Homography)/n');
    fprintf('> Getting canvas size...');tic;
    % Map four corners of the right image.
    TL = Hg\[1;1;1];
    TL = round([ TL(1)/TL(3) ; TL(2)/TL(3) ]);
    BL = Hg\[1;size(I2_global,1);1];
    BL = round([ BL(1)/BL(3) ; BL(2)/BL(3) ]);
    TR = Hg\[size(I2_global,2);1;1];
    TR = round([ TR(1)/TR(3) ; TR(2)/TR(3) ]);
    BR = Hg\[size(I2_global,2);size(I2_global,1);1];
    BR = round([ BR(1)/BR(3) ; BR(2)/BR(3) ]);

    % Canvas size.
    cw = max([1 size(I1_global,2) TL(1) BL(1) TR(1) BR(1)]) - min([1 size(I1_global,2) TL(1) BL(1) TR(1) BR(1)]) + 1;
    ch = max([1 size(I1_global,1) TL(2) BL(2) TR(2) BR(2)]) - min([1 size(I1_global,1) TL(2) BL(2) TR(2) BR(2)]) + 1;
    fprintf('done (%fs)/n',toc);

    % Offset for left image.
    fprintf('> Getting offset...');tic;
    off = [ 1 - min([1 size(I1_global,2) TL(1) BL(1) TR(1) BR(1)]) + 1 ; 1 - min([1 size(I1_global,1) TL(2) BL(2) TR(2) BR(2)]) + 1 ];
    fprintf('done (%fs)/n',toc);

    %--------------------------------------------
    % Image stitching with global homography (H).
    %--------------------------------------------
    % Warping source image with global homography 
    fprintf('Image stitching with global homography (H) and linear blending/n');
    fprintf('> Warping images by global homography...');tic;
    warped_img1 = uint8(zeros(ch,cw,3));
    warped_img1(off(2):(off(2)+size(I1_global,1)-1),off(1):(off(1)+size(I1_global,2)-1),:) = I1_global;
    warped_img2 = imagewarping(double(ch),double(cw),double(I2_global),Hg,double(off));
    warped_img2 = reshape(uint8(warped_img2),size(warped_img2,1),size(warped_img2,2)/3,3);
    fprintf('done (%fs)/n',toc);

    % Blending images by simple average (linear blending)
    fprintf('  Homography linear image blending (averaging)...');tic;
    fprintf('done (%fs)/n',toc);                         

    %-------------------------
    % Moving DLT (projective).
    %-------------------------
    fprintf('As-Projective-As-Possible Moving DLT on inliers/n');

    % Image keypoints coordinates.
    Kp = [total_sift_dorm(1,:)' total_sift_dorm(2,:)'];

    % Generating mesh for MDLT.
    fprintf('> Generating mesh for MDLT...');tic;
    [ X,Y ] = meshgrid(linspace(1,cw,C1),linspace(1,ch,C2));  
    
    % grid show
   % figure(100),mesh(Y, X);view(1);axis equal;hold on;
    
    fprintf('done (%fs)/n',toc);

    % Mesh (cells) vertices' coordinates.
    Mv = [X(:)-off(1), Y(:)-off(2)];

    % Perform Moving DLT
    fprintf('  Moving DLT main loop...');tic;
    Hmdlt = zeros(size(Mv,1),9);
    parfor i=1:size(Mv,1)

        % Obtain kernel    
        Gki = exp(-pdist2(Mv(i,:),Kp)./sigma^2);   

        % Capping/offsetting kernel
        Wi = max(gamma,Gki); 

        % This function receives W and A and obtains the least significant 
        % right singular vector of W*A by means of SVD on WA (Weighted SVD).
        v = wsvd(Wi,A);%image_fusion(warped_img2,warped_img1,warped_img1,double(Hg));
        h = reshape(v,3,3)';        

        % De-condition
        h = D2\h*D1;

        % De-normalize
        h = T_2\h*T_1;

        Hmdlt(i,:) = h(:);
    end
    fprintf('done (%fs)/n',toc);

    %---------------------------------
    % Image stitching with Moving DLT.
    %---------------------------------
    fprintf('As-Projective-As-Possible Image stitching with Moving DLT and linear blending/n');
    % Warping images with Moving DLT.
    fprintf('> Warping images with Moving DLT...');tic;
    warped_img1_dlt = uint8(zeros(ch,cw,3));
    warped_img1_dlt(off(2):(off(2)+size(I1_global,1)-1),off(1):(off(1)+size(I1_global,2)-1),:) = I1_global;

    [warped_img2_dlt] = imagewarping(double(ch),double(cw),double(I2_global),Hmdlt,double(off),X(1,:),Y(:,1)');
    warped_img2_dlt = reshape(uint8(warped_img2_dlt),size(warped_img2_dlt,1),size(warped_img2_dlt,2)/3,3);
   
    fprintf('done (%fs)/n',toc);

    % Blending images by averaging (linear blending)
    fprintf('  Moving DLT linear image blending (averaging)...');tic;
    fprintf('> Finished!./n');
    
    %ARMF+APAP:Thumbnail checkerboard view
    
      while (1)
          warped_img1=warped_img1(1:2:end,1:2:end,:);
          warped_img2=warped_img2(1:2:end,1:2:end,:);
          new_warped_img1_h1=size(warped_img1,2);
          new_warped_img1_w1=size( warped_img1,1);
          new_warped_img2_h2=size( warped_img2,2);
          new_warped_img2_w2=size( warped_img2,1);

           if (new_warped_img1_h1 < 1500)||(new_warped_img1_w1 < 1500)||(new_warped_img2_h2 < 1500)||(new_warped_img2_w2 < 1500)
                 display('fusion down success');
                 break;
          end
      end

     
     title('ARMF matching result fused image MDLT');
     grid_num=5;
     grid_size=floor(min(size(warped_img1,1),size(warped_img2,2))/grid_num);
     [~,~,f_mdlt]=mosaic_map(warped_img1,warped_img2,grid_size);  
     figure;
     imshow(f_mdlt);
     hold on;

% end

%%%%%%%% ARMF blocks over %%%%%%%%%%
%%