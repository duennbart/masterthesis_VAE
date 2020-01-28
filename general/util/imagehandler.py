import numpy as np
from PIL import Image

def save_reconimgs_as_grid(input_imgs,output_imgs,path):
    '''

    :param input_imgs: orginal input images np.array (N,H,B,C) N: number of images
    :param output_imgs:
    :param epoch:
    :param prefix:
    :return:
    '''
    input_imgs = np.asarray(input_imgs)
    if input_imgs.shape[0] == 1:
        input_imgs = np.squeeze(input_imgs, axis=0)
    input_imgs = input_imgs[:10, :]
    input_imgs = input_imgs * 255
    input_imgs = input_imgs.astype(np.uint8)
    input_imgs1, input_imgs2 = np.split(input_imgs, 2, axis=0)

    output_imgs = output_imgs[:10, :]
    output_imgs = output_imgs * 255
    output_imgs = output_imgs.astype(np.uint8)
    output_imgs1, output_img2 = np.split(output_imgs, 2, axis=0)

    grid = np.concatenate((input_imgs1, output_imgs1, input_imgs2, output_img2))

    grid = python_image_grid(grid, [4, 5])
    grid = np.squeeze(grid)

    im = Image.fromarray(grid)

    im.save(path)




def python_image_grid(input_array, grid_shape):
    """This is a pure python version of tfgan.eval.image_grid.
    Args:
      input_array: ndarray. Minibatch of images to format. A 4D numpy array
          ([batch size, height, width, num_channels]).
      grid_shape: Sequence of int. The shape of the image grid,
          formatted as [grid_height, grid_width].
    Returns:
      Numpy array representing a single image in which the input images have been
      arranged into a grid.
    Raises:
      ValueError: The grid shape and minibatch size don't match.
      ValueError: The input array isn't 4D.
    """
    if grid_shape[0] * grid_shape[1] != int(input_array.shape[0]):
        raise ValueError("Grid shape %s incompatible with minibatch size %i." %
                         (grid_shape, int(input_array.shape[0])))
    if len(input_array.shape) != 4:
        raise ValueError("Unrecognized input array format.")
    image_shape = input_array.shape[1:3]
    num_channels = input_array.shape[3]
    height, width = (
        grid_shape[0] * image_shape[0], grid_shape[1] * image_shape[1])
    input_array = np.reshape(
        input_array, tuple(grid_shape) + tuple(image_shape) + (num_channels,))
    input_array = np.transpose(input_array, [0, 1, 3, 2, 4])
    input_array = np.reshape(
        input_array, [grid_shape[0], width, image_shape[0], num_channels])
    input_array = np.transpose(input_array, [0, 2, 1, 3])
    input_array = np.reshape(input_array, [height, width, num_channels])
    return input_array