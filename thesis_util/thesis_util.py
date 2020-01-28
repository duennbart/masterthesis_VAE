import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image
from thesis_util.PDFTex import PDFTex


def pre_image_4thesis(input_path,save_path,name_even_row= 'Training data set',name_odd_rows='Sampled',title=None):
    '''
    add red and green boarder for orignal and recon or sampled
    :param input_path: input path of image
    :param save_path: save path of image
    :param title: add a title if supported
    :return:
    '''
    im = cv2.imread(input_path)
    delta_y = int(im.shape[0] / 4)

    cv2.rectangle(im, (0, 0), (im.shape[1], delta_y - 1), (0, 255, 0), 2)
    cv2.rectangle(im, (0, delta_y + 1), (im.shape[1], delta_y * 2 - 1), (0, 0, 255), 2)
    cv2.rectangle(im, (0, delta_y * 2 + 1), (im.shape[1], delta_y * 3 - 1), (0, 255, 0), 2)
    cv2.rectangle(im, (0, delta_y * 3 + 1), (im.shape[1], delta_y * 4), (0, 0, 255), 2)
    im = add_text(image=im,text_to_show=name_even_row,position=(10, 5),fill=(0,255,0,255))
    im = add_text(image=im, text_to_show=name_odd_rows, position=(10, 5 + delta_y ), fill=(255, 0, 0, 255))
    im = add_text(image=im, text_to_show=name_even_row, position=(10, 5+delta_y*2), fill=(0, 255, 0, 255))
    im = add_text(image=im, text_to_show=name_odd_rows, position=(10, 5 + delta_y*3), fill=(255, 0, 0, 255))

    if title != None:
        im = cv2.copyMakeBorder(im, 60, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        im = add_text(image=im, text_to_show=title, position=(500, 10), fill=(0, 0, 0, 255))
    cv2.imwrite(save_path,im)
    cv2.imshow('image', im)
    # cv2.imshow('bottom', bottom)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def add_text(image,text_to_show,position,fill):
    # Convert the image to RGB (OpenCV uses BGR)
    cv2_im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Pass the image to PIL
    pil_im = Image.fromarray(cv2_im_rgb)

    draw = ImageDraw.Draw(pil_im)
    # use a truetype font
    font = ImageFont.truetype("lmroman10-regular.otf", 32)

    # Draw the text
    draw.text(position, text_to_show, font=font,fill=fill)

    cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    return cv2_im_processed

def stack_trials(result_paths):
    x = np.load(result_paths[0])
    for i in range(1, len(result_paths)):
        x = np.vstack((x, np.load(result_paths[i])))
    return x

def get_min_max_from_results(result_paths):
    dic_minmax = {}
    x = stack_trials(result_paths)
    mins = x.min(1)
    maxs = x.max(1)

    dic_minmax['min_mean'] = mins.mean()
    dic_minmax['max_mean'] = maxs.mean()
    dic_minmax['min_std'] = mins.std()
    dic_minmax['max_std'] = maxs.std()

    return dic_minmax

def create_result_plot(result_paths_train,result_paths_test, plot_title,  ylabel,xlabel=r'$\textrm{Epoch}$',figure_path=None,legend_position='upper right'):
    '''
    create plot for history of training included in the thesis e.g. MSE, MS-SSIM or FID
    :param result_paths_train: list of npy files of the training results
    :param result_paths_test: list of npy files of the test results
    :param plot_title: title of the figure
    :param ylabel: y label name
    :param xlabel: x label name, normally Epoch
    :param figure_path: path where to save the figure
    :param legend_position: position of the legend
    :return:
    '''
    #plt.style.use('ggplot')
    matplotlib.rcParams['text.usetex'] = True
    # merge train results

    y_values_train = stack_trials(result_paths_train)
    mean_train = y_values_train.mean(axis=0)
    std_train = y_values_train.std(axis=0)

    epochs = np.arange(0, mean_train.shape[0])

    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot(epochs, mean_train, color='red', label=r'$\textrm{Train data}$')

    ax.fill_between(epochs, mean_train - std_train, mean_train + std_train, color='red', alpha=0.2)

    # merge test results, no test result for FID score, because it is sampled
    if result_paths_test != None:
        y_values_test = stack_trials(result_paths_test)

        mean_test = y_values_test.mean(axis=0)
        std_test = y_values_test.std(axis=0)
        ax.plot(epochs, mean_test, color='blue', label=r'$\textrm{Test data}$')
        ax.fill_between(epochs, mean_test - std_test, mean_test + std_test, color='blue', alpha=0.2)
        ax.legend(loc=legend_position)

    ax.grid()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(plot_title)
    # tikzplotlib.save("C:\\Users\\ga45tis\\GIT\\masterthesisgeneral\\latex\\900 Report\\images\\experiments\\VAE\\mse_history_VAE50.tex")
    if figure_path != None:
        plt.savefig(figure_path,bbox_inches='tight')
    plt.show()

    return True


def eval_experiment(save_directory,model_name,pathes_2_experiments,title,sample_img_path,recon_test_img_path,recon_train_img_path):
    # MSE error
    figure_path = save_directory + "mse_history_" + model_name + ".pdf"
    result_path_train = []
    result_path_test = []
    for element in pathes_2_experiments:
        result_path_train.append(element + r'\results\mse_train300.npy')
        result_path_test.append(element + r'\results\mse_test300.npy')




    ylabel = r'$\textrm{MSE}$'

    # mse error
    # create_result_plot(result_paths_train=result_path_train, result_paths_test=result_path_test, plot_title=title,
    #                     ylabel=ylabel, xlabel=r'$\textrm{Epoch}$', figure_path=figure_path)

    train_minmax = get_min_max_from_results(result_path_train)
    print('Train min, max MSE: ' + str(train_minmax))
    test_minmax = get_min_max_from_results(result_path_test)
    print('Test min, max MSE: ' + str(test_minmax))

    # MS-SSIM
    figure_path = save_directory + "msssim_history_" + model_name + ".pdf"
    result_path_train = []
    result_path_test = []
    for element in pathes_2_experiments:
        result_path_train.append(element +  r'\results\msssim_train300.npy')
        result_path_test.append(element + r'\results\msssim_test300.npy')


    ylabel = r'$\textrm{MS-SSIM}$'
    # msssim error
    # create_result_plot(result_paths_train=result_path_train, result_paths_test=result_path_test, plot_title=title,
    #                     ylabel=ylabel, xlabel=r'$\textrm{Epoch}$', figure_path=figure_path,
    #                     legend_position='lower right')
    train_minmax = get_min_max_from_results(result_path_train)
    print('Train min, max MSSIM: ' + str(train_minmax))
    test_minmax = get_min_max_from_results(result_path_test)
    print('Test min, max MSSIM: ' + str(test_minmax))

    # FID

    figure_path = save_directory + "fid_history_" + model_name + ".pdf"
    result_path_train = []
    for element in pathes_2_experiments:
        result_path_train.append(element +  r'\results\fid_score300.npy')
    #
    #
    #
    ylabel = r'$\textrm{FID}$'
    # fid error
    # create_result_plot(result_paths_train=result_path_train, result_paths_test=None, plot_title=title, ylabel=ylabel,
    #                    xlabel=r'$\textrm{Epoch}$', figure_path=figure_path, legend_position='lower right')
    train_minmax = get_min_max_from_results(result_path_train)
    print('min, max FID: ' + str(train_minmax))

    # create thesis images

    # generated samples
    # figure_path = save_directory + "sample_" + model_name + ".png"
    # pre_image_4thesis(input_path=sample_img_path, save_path=figure_path, title=None)
    #
    # # reconstrution test
    # figure_path = save_directory + "recon_test_" + model_name + ".png"
    # pre_image_4thesis(input_path=recon_test_img_path, save_path=figure_path, name_even_row='Test data set',
    #                   name_odd_rows='Reconstructed', title=None)
    #
    # # reconstrution train
    # figure_path = save_directory + "recon_train_" + model_name + ".png"
    # pre_image_4thesis(input_path=recon_train_img_path, save_path=figure_path, name_odd_rows='Reconstructed', title=None)



def concat_v(im1, im2):
    '''
    concatenate two pillow images row by row -> vertical
    :param im1: pillow image 1
    :param im2: pillow image 2
    :return:
    '''
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

# concate imgs horizontal
def concat_h(im1, im2):
    '''
    concatenate two pillow images column by column -> horizontal
    :param im1: pillow image 1
    :param im2: pillow image 2
    :return: concatenate pillow image
    '''
    dst = Image.new('RGB', (im1.width+ im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width,0))
    return dst

def get_img_pair(img,col=0,row=0, ncol = 5, nrow=4):
    '''
    gets from a image prduced during training a pair of original and recon image
    :param img: the complete input iamge
    :param col: the column of the original image you want
    :param row:  thr row of the original image, ether 0 or 2
    :param ncol:  number of total columns
    :param nrow: number of total rows
    :return: returns croped image of orgiginal and reconstructed
    '''
    assert row % 2 ==0
    width, height = img.size
    img_height = height / nrow
    img_width = width / ncol
    top = row*img_height
    bottom = top + img_height*2
    left = col *img_width
    right = left + img_width
    crop_img = img.crop((left, top, right, bottom))
    return crop_img

def get_single_img(img,col=0,row=0, ncol = 5, nrow=4):
    '''
    gets from a image produced during training a single image
    :param img: the complete input iamge
    :param col: the column of the original image you want
    :param row:  thr row of the original image
    :param ncol:  number of total columns
    :param nrow: number of total rows
    :return: returns croped image of orgiginal and reconstructed
    '''
    width, height = img.size
    img_height = height / nrow
    img_width = width / ncol
    top = row*img_height
    bottom = top + img_height
    left = col *img_width
    right = left + img_width
    crop_img = img.crop((left, top, right, bottom))
    return crop_img


def restack_imgs_4recon(path):
    '''
    restack the images created during training to, 1. row original
    KL class 0-4, second row the corresponding reconstructed images
    :param path:
    :return:
    '''
    im = Image.open(path)
    # get img class 0
    im_1 = get_img_pair(im, col=0, row=0)
    # get img class 1
    im_2 = get_img_pair(im, col=2, row=0)
    # concate cl 0 and cl 1
    im_1 = concat_h(im_1,im_2)
    # get img class 2
    im_2 = get_img_pair(im, col=3, row=0)
    # concate
    im_1 = concat_h(im_1, im_2)
    # get img class 3
    im_2 = get_img_pair(im, col=1, row=2)
    # concate
    im_1 = concat_h(im_1, im_2)
    # get img class 4
    im_2 = get_img_pair(im, col=3, row=2)
    im_1 = concat_h(im_1, im_2)
    return im_1

def restack_imgs_4random(path):
    '''
    restack the images created during training to, 1. row original
    KL class 0-4, second row the corresponding reconstructed images
    :param path:
    :return:
    '''
    im = Image.open(path)
    # get img class 0
    im_1 = get_single_img(im, col=0, row=1)
    # get img class 1
    im_2 = get_single_img(im, col=1, row=1)
    # concate cl 0 and cl 1
    im_1 = concat_h(im_1,im_2)
    # get img class 2
    im_2 = get_single_img(im, col=2, row=1)
    # concate
    im_1 = concat_h(im_1, im_2)
    # get img class 3
    im_2 = get_single_img(im, col=3, row=1)
    # concate
    im_1 = concat_h(im_1, im_2)
    # get img class 4
    im_2 = get_single_img(im, col=4, row=1)
    im_1 = concat_h(im_1, im_2)
    return im_1



def create_eval_recon_imgs(recon_img_path,title,pdf_file_name,save_directory,prefix_4include=r"images/experiments/VAE/"):
    '''
    create reconstruction images for thesis, first row original, second row reconstuction
    :param recon_img_path:
    :param title:
    :param pdf_file_name:
    :param save_directory:
    :return:
    '''
    pdf_file_name = pdf_file_name
    single_img_width = 0.2
    pdf_hight = 0.46
    # Opens a image and restack it
    img = restack_imgs_4recon(recon_img_path)
    # img.show()
    pdftex = PDFTex(img=img, save_path=save_directory, pdf_file_name=pdf_file_name, pdf_hight=pdf_hight, img_y_pos=0.03,
                    prefix_4include=prefix_4include)  #
    # put title
    pdftex.put_text(text=title, x_position=0.51, y_position=0.45, text_color='black', font_size=r'\large',
                    box_alignment='t')
    # put KL classes
    for i in range(0, 5):
        x_position = i * single_img_width + 0.1
        pdftex.put_text(text="KL-" + str(i), x_position=x_position, y_position=0, text_color='black',
                        font_size=r'\normalsize',
                        box_alignment='t')
    pdftex.create_pdf_tex()

def create_eval_random_sample_imgs(recon_img_path,title,pdf_file_name,save_directory,prefix_4include=r"images/experiments/VAE/"):
    '''
    create imags of 5 random generated images in a row
    :param recon_img_path:
    :param title:
    :param pdf_file_name:
    :param save_directory:
    :return:
    '''
    pdf_file_name = pdf_file_name
    single_img_width = 0.2
    pdf_hight = 0.26
    # Opens a image and restack it
    img = restack_imgs_4random(recon_img_path)
    # img.show()
    pdftex = PDFTex(img=img, save_path=save_directory, pdf_file_name=pdf_file_name, pdf_hight=pdf_hight, img_y_pos=0.03,
                    prefix_4include=prefix_4include)  #
    # put title
    pdftex.put_text(text=title, x_position=0.51, y_position=0.25, text_color='black', font_size=r'\large',
                    box_alignment='t')
    # put KL classes
    for i in range(0, 5):
        x_position = i * single_img_width + 0.1
        pdftex.put_text(text=str(i), x_position=x_position, y_position=0, text_color='black',
                        font_size=r'\normalsize',
                        box_alignment='t')
    pdftex.create_pdf_tex()





def create_eval_recon_all_imgs(data,title,pdf_file_name,save_directory,prefix_4include=r"images/experiments/VAE/",add_kl_class=True):
    '''
    create reconstruction images for thesis, first row original, second row reconstuction
    :param recon_img_path:
    :param title:
    :param pdf_file_name:
    :param save_directory:
    :return:
    '''
    pdf_file_name = pdf_file_name
    single_img_width = 0.2
    pdf_hight = len(data)*0.2 + 0.1
    h = 255
    for element in data:
        # Opens a image and restack it
        img = restack_imgs_4recon(element[0])

        # Size of the image in pixels (size of orginal image)
        width, height = img.size
        # Setting the points for cropped image
        left = 0
        top = h * element[2]
        right = width
        bottom = h + h * element[2]
        # Cropped image of above dimension
        crop_img = img.crop((left, top, right, bottom))
        #crop_img.show()
        try:
            total_im = concat_v(total_im, crop_img)

        except NameError:
            total_im = crop_img
    #total_im.show()
    pdftex = PDFTex(img=total_im, save_path=save_directory, pdf_file_name=pdf_file_name, pdf_hight=pdf_hight, pdf_width=1.22,img_y_pos=0.03,
                    prefix_4include=prefix_4include,img_x_pos=0.22)  #
    # put title
    pdftex.put_text(text=title, x_position=0.51 + 0.22, y_position=len(data)*0.2 + 0.05, text_color='black', font_size=r'\large',
                    box_alignment='t')
    # put KL classes
    for i in range(0, len(data)):
        # set experiment name
        element = data[i]
        y_position = len(data)*0.2 - i * single_img_width -0.09
        pdftex.put_text(text=element[1], x_position=0.22, y_position=y_position, text_color='black',
                        font_size=r'\normalsize',
                        box_alignment='mr')

        # put KL classes
    if add_kl_class == True:
        for i in range(0, 5):
            x_position = i * single_img_width + 0.1 + 0.22
            pdftex.put_text(text="KL-" + str(i), x_position=x_position, y_position=0, text_color='black',
                            font_size=r'\normalsize',
                            box_alignment='t')
    pdftex.create_pdf_tex()



