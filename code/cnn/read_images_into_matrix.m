function data_matrix = read_images_into_matrix()

imagefiles = dir('*.jpg');      
nfiles = length(imagefiles);


for ii=1:nfiles
   currentfilename = imagefiles(ii).name;
   currentimage = imread(currentfilename);
   image_vec = convert_image_to_vector(currentimage);
   data_matrix(ii, :) = image_vec;
end
