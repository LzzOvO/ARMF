function [outputArg1,outputArg2] = ARMF_finalres(inputArg1,inputArg2,inputArg3,inputArg4,inputArg5,inputArg6,inputArg7,inputArg8,inputArg9,inputArg10,inputArg11,inputArg12,inputArg13,inputArg14)
%ARMF_FINALRES Summary of this function goes here
%   Detailed explanation goes here

        imshow([inputArg13 inputArg14]);
        hold on;
     for total_i_phase=1:size(inputArg1,1)
            plot(inputArg1(total_i_phase,1),inputArg1(total_i_phase,2),'y+','LineWidth',1);
            plot(inputArg2(total_i_phase,1)+size(inputArg13,2),inputArg2(total_i_phase,2),'y+','LineWidth',1);
            plot([(inputArg1(total_i_phase,1)) inputArg2(total_i_phase,1)+size(inputArg13,2)],[(inputArg1(total_i_phase,2)) (inputArg2(total_i_phase,2))],'y-');
     end

      for total_i_sift=1:size(inputArg3,1)
            plot(inputArg3(total_i_sift,1),inputArg3(total_i_sift,2),'b+','LineWidth',1);
            plot(inputArg4(total_i_sift,1)+size(inputArg13,2),inputArg4(total_i_sift,2),'b+','LineWidth',1);
            plot([(inputArg3(total_i_sift,1)) inputArg4(total_i_sift,1)+size(inputArg13,2)],[(inputArg3(total_i_sift,2)) (inputArg4(total_i_sift,2))],'b-');
      end
     
      for total_i_lsd=1:size(inputArg5,1)
            plot(inputArg5(total_i_lsd,1),inputArg5(total_i_lsd,2),'c+','LineWidth',1);
            plot(inputArg6(total_i_lsd,1)+size(inputArg13,2),inputArg6(total_i_lsd,2),'c+','LineWidth',1);
            plot([(inputArg5(total_i_lsd,1)) inputArg6(total_i_lsd,1)+size(inputArg13,2)],[(inputArg5(total_i_lsd,2)) (inputArg6(total_i_lsd,2))],'c-');
      end
        title('total feature matching');
        
              for total_i_sift_region=1:size(inputArg7,1)
                    plot(inputArg7(total_i_sift_region,1),inputArg7(total_i_sift_region,2),'w+','LineWidth',4);
                    plot(inputArg8(total_i_sift_region,1)+size(inputArg13,2),inputArg8(total_i_sift_region,2),'w+','LineWidth',4);
                    plot([(inputArg7(total_i_sift_region,1)) inputArg8(total_i_sift_region,1)+size(inputArg13,2)],[(inputArg7(total_i_sift_region,2)) (inputArg8(total_i_sift_region,2))],'w-','LineWidth',4);
              end
      
            for total_i_lsd_region=1:size(inputArg11,1)
            plot(inputArg11(total_i_lsd_region,1),inputArg11(total_i_lsd_region,2),'w+','LineWidth',4);
            plot(inputArg12(total_i_lsd_region,1)+size(inputArg13,2),inputArg12(total_i_lsd_region,2),'w+','LineWidth',4);
            plot([(inputArg11(total_i_lsd_region,1)) inputArg12(total_i_lsd_region,1)+size(inputArg13,2)],[(inputArg11(total_i_lsd_region,2)) (inputArg12(total_i_lsd_region,2))],'w-','LineWidth',4);
            end
      
            for total_i_phase_region=1:size(inputArg9,1)
            plot(inputArg9(total_i_phase_region,1),inputArg9(total_i_phase_region,2),'w+','LineWidth',4);
            plot(inputArg10(total_i_phase_region,1)+size(inputArg13,2),inputArg10(total_i_phase_region,2),'w+','LineWidth',4);
            plot([(inputArg9(total_i_phase_region,1)) inputArg10(total_i_phase_region,1)+size(inputArg13,2)],[(inputArg9(total_i_phase_region,2)) (inputArg10(total_i_phase_region,2))],'w-','LineWidth',4);
            end
outputArg1 = inputArg1;
outputArg2 = inputArg2;
end

