import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, laplace  
import scipy.fft as fft
import scipy.signal as signal
from PIL import Image

#create a 2d ricker wavelet from a laplacian of gaussian
def ricker(a, size):

    gaussian = signal.gaussian(size, std=a)
    gaussian_2d = np.outer(gaussian, gaussian)
    laplacian = -laplace(gaussian_2d, mode='reflect')
    # laplacian = -signal.convolve2d(gaussian_2d, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]), mode='same', boundary='symm')
    
    return laplacian

def rotate(x, y, theta):

    '''Rotate arrays of coordinates x and y by theta radians '''

    theta_rad = np.radians(theta)
    x_rotated = x * np.cos(theta_rad) + y*np.sin(theta_rad)
    y_rotated = y * np.cos(theta_rad) - x*np.sin(theta_rad)

    return x_rotated, y_rotated

 

def mask_2d(num_points, aperture, angle):

    '''Create a 2D mask with a given aperture and angle of incidence'''

    x_values = np.linspace(-1, 1, num_points)
    y_values = np.linspace(-1, 1, num_points)
    aperture = np.deg2rad(90-aperture)
    x_mesh, y_mesh,= np.meshgrid(x_values, y_values)
    x_mesh, y_mesh = rotate(x_mesh, y_mesh, angle)
    mask = np.where((np.abs(np.tan(aperture)*x_mesh) > np.abs(y_mesh)) , 0,1)

    return mask

 

def create_psf(size, a, aperture, angle, amp=1,  mask_on = True, print_on = False, smooth_mask = False,sigma=0.5):

    #create a mask
    mask = mask_2d(size, aperture=aperture, angle=angle)
    if smooth_mask:
        mask = gaussian_filter(mask, sigma=sigma)

    #create a  2D wavelet
    wavelet_2D = ricker(a,size)

    #create a 2D wavelet in the frequency domain
    wavelet_2D_f = fft.fft2(wavelet_2D)
    wavelet_2D_f = fft.fftshift(wavelet_2D_f)

 
    #apply the mask
    if mask_on:
        wavelet_2D_f_masked = wavelet_2D_f*mask

    else:
        wavelet_2D_f_masked = wavelet_2D_f


    if print_on:
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
        axs[0,0].imshow(mask, cmap = 'gray')
        axs[0,0].set_title('Iluminação')
        axs[0,1].imshow(wavelet_2D)
        axs[0,1].set_title('Prof. iluminação total ')
        axs[1,0].imshow(np.abs(wavelet_2D_f))
        axs[1,0].set_title('Nº onda iluminação total')
        axs[1,1].imshow(np.abs(wavelet_2D_f_masked))
        axs[1,1].set_title('Nº onda iluminação restrita')
        st.pyplot(fig)

    #go back to the time domain
    wavelet_2D_f_masked = fft.ifftshift(wavelet_2D_f_masked)
    psf = fft.ifft2(wavelet_2D_f_masked)
    psf = np.real(psf)

    return amp*psf, mask

def make_rc(imp):
    """
    Compute reflection coefficients.
    """
    imp = np.pad(imp, pad_width=[(0, 1), (0, 0)], mode='edge')
    upper = imp[ :-1, :]
    lower = imp[1:  , :]
    
    return (lower - upper) / (lower + upper)


size = 64 #st.slider('Tamanho da wavelet (m): ', 32, 128, 64)
f = st.slider('Frequência da wavelet ricker (hz): ', 15, 120, 50)
dt = 4 #st.slider('Amostragem em tempo (ms): ', 1, 8, 4)
velocidade = st.slider('Velocidade média (m/s): ', 1500, 3500, 2000)

t = np.arange(-size*2, size*2, dt)
lam = velocidade/f
a = velocidade/(f*2*np.pi)
r2d = ricker(a, size)

on_off_w = st.toggle('Plota wavelet', value=False)
if on_off_w:
    fig, axs = plt.subplots()
    axs.plot(t,r2d[:,size//2])
    axs.set_title('Wavelet ricker')
    # plt.ylim(-0.08,0.25)
    st.pyplot(fig)

#st.write("Comprimento de onda da wavelet em profundidade (m): ", lam)
 #create the psf
st.write('Criando a psf')
aperture = st.slider('Angulo de iluminação: ', 0, 89, 30)
angle = st.slider('Angulo de incidência: ', -89, 89, 0)

on_off = st.toggle('Mostra máscasra e wavelet', value=False)
                

psf, mask = create_psf(size, a, aperture, angle, amp=1,  mask_on = True, print_on = on_off, smooth_mask = False,sigma=0.5)

fig, axs = plt.subplots()
axs.imshow(psf, cmap='gray_r')
axs.set_title('PSF')
axs.set_axis_off()
plt.colorbar()
st.pyplot(fig)

uploaded_file = st.file_uploader("Escolha o arrquivo com a foto do afloramento para criar uma sísmica sintética:")
if uploaded_file is not None:
    # To read file as bytes:
    
    im = Image.open(uploaded_file).convert('L')
    st.image(im)
    w,h= im.size
    r = h/w
    im = im.resize((int(800),int(800*r)))
    fig, axs = plt.subplots()
    axs.imshow(im)
    im = np.array(im)
    #convert im to impedance scale
    ip = im/255
    ip = 6000000+(9000000*ip)
    #convert the impedance to reflection coefficients
    rc = make_rc(-ip)
    #create the synthetic
    import time
    with st.spinner('Aguarde...'):
        time.sleep(1)
        syn = signal.convolve2d(rc, psf, mode='same', boundary='symm')
    st.success('Pronto!')



    # with st.spinner('Aguarde...'):
    #     syn = signal.convolve2d(rc, psf, mode='same', boundary='symm')
    # st.success('Pronto!')

    fig, axs = plt.subplots()
    axs.imshow(syn, cmap='gray_r')
    axs.set_axis_off()
    st.pyplot(fig)


   

    



    


    

