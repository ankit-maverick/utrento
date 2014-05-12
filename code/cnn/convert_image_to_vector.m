function image_vec = convert_image_to_vector(image)
	num_elements = prod(size(image));
	image_vec = reshape(image, num_elements, 1);
end
