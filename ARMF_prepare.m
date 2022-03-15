function [outputArg1,outputArg2] = ARMF_prepare(inputArg1,inputArg2,inputArg3,inputArg4,inputArg5,inputArg6)
%ARMF_PREPARE Summary of this function goes here
%   Detailed explanation goes here

Ih_res=abs(inputArg1-inputArg2);
Iw_res=abs(inputArg3-inputArg4);

if ((inputArg1-inputArg2)>0)&&((inputArg3-inputArg4)>0)
     display('inputArg5_h>inputArg6_h,inputArg5_w>inputArg6_w');
    inputArg6=padarray(inputArg6, [Ih_res Iw_res], 'post');
elseif((inputArg1-inputArg2)>0)&&((inputArg3-inputArg4)<0)
    display('inputArg5_h>inputArg6_h,inputArg5_w<inputArg6_w');
    inputArg6=padarray(inputArg6, [Ih_res 0], 'post');
    inputArg5=padarray(inputArg5, [0 Iw_res], 'post');
elseif((inputArg1-inputArg2)<0)&&((IinputArg3-inputArg4)>0)
    display('inputArg5_h<inputArg6_h,inputArg5_w>inputArg6_w');
    inputArg5=padarray(inputArg5, [Ih_res 0], 'post');
    inputArg6=padarray(inputArg6, [0 Iw_res], 'post');
elseif((inputArg1-inputArg2)<0)&&((inputArg3-inputArg4)<0)
    display('inputArg5_h<inputArg6_h,inputArg5_w<inputArg6_w');
    inputArg5=padarray(inputArg5, [Ih_res Iw_res], 'post');
elseif((inputArg1-inputArg2)==0)&&((inputArg3-inputArg4)<0)
    display('inputArg5_h=inputArg6_h,inputArg5_w<inputArg6_w');
    inputArg5=padarray(inputArg5, [Ih_res Iw_res], 'post');
elseif((inputArg1-inputArg2)==0)&&((inputArg3-inputArg4)>0)
    display('inputArg5_h=inputArg6_h,inputArg5_w>inputArg6_w');
    inputArg6=padarray(inputArg6, [Ih_res Iw_res], 'post');


end 
display('resize over');
% display(inputArg1);
% display(inputArg3);
% display(inputArg2);
% display(inputArg4);

outputArg1 = inputArg5;
outputArg2 = inputArg6;



end

