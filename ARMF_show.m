function [outputArg1,outputArg2] = ARMF_show(inputArg1,inputArg2,inputArg3,inputArg4,inputArg5,inputArg6,inputArg7,inputArg8,inputArg9,inputArg10,inputArg11,inputArg12,inputArg13,inputArg14,inputArg15)
%ARMF_SHOW Summary of this function goes here
%   Detailed explanation goes here

         fig=inputArg1;
%         imshow([inputArg2,inputArg3]);
%         hold on;
        for isift=1:inputArg4
            plot(inputArg5(isift,1),inputArg5(isift,2),'b*','LineWidth',3);
            plot(inputArg6(isift,1)+size(inputArg2,2),inputArg6(isift,2),'b*','LineWidth',3);
            plot([(inputArg5(isift,1)) inputArg6(isift,1)+size(inputArg2,2)],[(inputArg5(isift,2)) (inputArg6(isift,2))],'b-','LineWidth',4);
        end

        for p_lsd=1:inputArg7
            plot(inputArg8(p_lsd,1),inputArg8(p_lsd,2),'c+','LineWidth',2);
            plot(inputArg9(p_lsd,1)+size(inputArg2,2),inputArg9(p_lsd,2),'c+','LineWidth',2);
            plot([(inputArg8(p_lsd,1)) inputArg9(p_lsd,1)+size(inputArg2,2)],[(inputArg8(p_lsd,2)) (inputArg9(p_lsd,2))],'c-','LineWidth',4);
        end

         for p=1:inputArg10
            plot(inputArg11(p,1),inputArg11(p,2),'y+','LineWidth',2);
            plot(inputArg12(p,1)+size(inputArg2,2),inputArg12(p,2),'y+','LineWidth',2);
            plot([(inputArg11(p,1)) inputArg12(p,1)+size(inputArg2,2)],[(inputArg11(p,2)) (inputArg12(p,2))],'y-','LineWidth',4);
         end


        
        
outputArg1 = fig;
outputArg2 = inputArg2;
end

