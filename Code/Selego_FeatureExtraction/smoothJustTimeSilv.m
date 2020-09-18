function [Y]=smoothJustTimeSilv(I,sigma)
% convolution of time
% class(I)
Y = imsmooth(I,sigma) ;
Y = [zeros(size(Y)), Y ,zeros(size(Y))];
% py_I = mat2np(I);
% gauss_kernel =  py.astropy.convolution.Gaussian1DKernel(sigma);
% smoothed_data_gauss = py.astropy.convolution.convolve(I, gauss_kernel);
% Y = double(py.array.array('d',py.numpy.nditer(smoothed_data_gauss)));
% Y = reshape(Y,numel(Y),1);
% Y = [zeros(size(Y)), Y ,zeros(size(Y))];
end

% function npary = mat2np(mat)
% 
% % convert matlab matrix to python (Numpy) ndarray 
% sh = fliplr(size(mat));
% mat2 = reshape(mat,numel(mat),1);  % [1, n] vector
% npary = py.numpy.array(mat2);
% npary = npary.reshape(int32(sh)).transpose();  % python ndarray
% 
% end