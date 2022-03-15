function [outputArg1,outputArg2,outputArg3,outputArg4,outputArg5,outputArg6,outputArg7,outputArg8] = ARMF_blocks(inputArg1,inputArg2,inputArg3,inputArg4,inputArg5,inputArg6,inputArg7)
%ARMF_BLOCKS Summary of this function goes here
%   Detailed explanation goes here




h_val=inputArg1*(1-inputArg7);
hr_val=inputArg1*(1-inputArg7);

w_val=inputArg2*(1-inputArg7);
wr_val=inputArg2*(1-inputArg7);


max_row = (inputArg3-inputArg1)/h_val+1;
r_max_row = (inputArg5-inputArg1)/hr_val+1;

max_col = (inputArg4-inputArg2)/w_val+1;
r_max_col = (inputArg6-inputArg2)/wr_val+1;

if max_row==fix(max_row)

    max_row=max_row;

else

    max_row=fix(max_row+1);

end

if max_col==fix(max_col)

    max_col=max_col;

else

    max_col=fix(max_col+1);

end
if r_max_row==fix(r_max_row)%�ж��Ƿ��ܹ�����

    r_max_row=r_max_row;

else

    r_max_row=fix(r_max_row+1);

end

if r_max_col==fix(r_max_col)%�ж��Ƿ��ܹ�����

    r_max_col=r_max_col;

else

    r_max_col=fix(r_max_col+1);

end


outputArg1 = max_row;
outputArg2 = max_col;
outputArg3 = r_max_row;
outputArg4= r_max_col;
outputArg5 = h_val;
outputArg6 = hr_val;
outputArg7 = w_val;
outputArg8= wr_val;
end

