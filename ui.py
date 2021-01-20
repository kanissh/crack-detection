import PySimpleGUI as sg
import cv2 as cv
import numpy as np
import methods


def select_image():
    image_path = '';
    while True:
        image_path = sg.popup_get_file('Select image')
        if image_path == '':
            sg.popup('Please select an image')

        if image_path != '':
            break
    image = cv.imread(image_path)
    return image


def resize(image_original):
    image_original_copy = np.ndarray.copy(image_original)
    cv.imshow('Original Image', image_original)

    layout_request = [
        [
            sg.Text('Do you wish to resize the image?'),
            sg.Button('Resize'), sg.Button('No')
        ]
    ]

    layout_resize = [
        [
            sg.Image(data='', key='-IMAGE-'),

        ]
    ]

    layout_slider = [
        [
            sg.Slider(range=(20, 100),
                      default_value=100,
                      size=(20, 15),
                      orientation='horizontal', key='-SLIDER-', enable_events=True)
        ],
        [
            sg.Button('Done resizing')
        ]
    ]

    window_request = sg.Window(title='Alert', layout=layout_request)
    event1, values1 = window_request.read()

    if event1 == 'Resize':
        window_request.close()
        cv.destroyWindow('Original Image')

        window_resize = sg.Window(title='Resize Window', layout=layout_resize, finalize=True)

        im_new = cv.imencode('.png', image_original)[1].tobytes()

        window_resize['-IMAGE-'].update(data=im_new)

        window_slider = sg.Window(title='', layout=layout_slider)

        while True:
            window_resize.read(timeout=50)
            event3, values3 = window_slider.read()

            if event3 == sg.WIN_CLOSED:
                break
            elif event3 == 'Done resizing':
                window_resize.close()
                window_slider.close()

            # resize image

            scale_factor = values3['-SLIDER-']
            width = int(image_original.shape[1] * scale_factor / 100)
            height = int(image_original.shape[0] * scale_factor / 100)
            dim = (width, height)
            image_original_copy = cv.resize(image_original, dim, interpolation=cv.INTER_AREA)

            im_new = cv.imencode('.png', image_original_copy)[1].tobytes()
            window_resize['-IMAGE-'].update(data=im_new)


    else:
        cv.destroyWindow('Original Image')
        window_request.close()

    return image_original_copy


def mark_crack(image):
    image_bw = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    output_image = np.ndarray.copy(image)
    layout_sliders = [
        [
            sg.Text('Alpha'),
            sg.Slider(range=(0, 10),
                      default_value=5,
                      size=(50, 15),
                      orientation='horizontal', key='-ALPHA-', enable_events=True)
        ],
        [
            sg.Text('Beta'),
            sg.Slider(range=(0, 10),
                      default_value=5,
                      size=(50, 15),
                      orientation='horizontal', key='-BETA-', enable_events=True)
        ],
        [
            sg.Text('gamma'),
            sg.Slider(range=(0, 100),
                      default_value=5,
                      size=(50, 15),
                      orientation='horizontal', key='-GAMMA-', enable_events=True)
        ],
        [
            sg.In(key='-CONTOUR_AREA-', default_text='100'),
            sg.Button(button_text='Set Contour Area', key='-SET_CONTOUR-')
        ],
        [
            sg.Text('Show crack on input image'), sg.Button('Show')
        ]
    ]

    layout_im = [
        [
            sg.Image(data='', key='-IMAGE-'),

        ]
    ]

    window_im = sg.Window(title='Output Image', layout=layout_im, finalize=True)

    im_new = cv.imencode('.png', image)[1].tobytes()
    window_im['-IMAGE-'].update(data=im_new)
    window_sliders = sg.Window(title='Editing Parameters', layout=layout_sliders)
    contour_area = 100
    while True:

        window_im.read(timeout=10)
        event_sliders, values_sliders = window_sliders.read()

        if event_sliders == sg.WIN_CLOSED:
            break
        elif event_sliders == '-SET_CONTOUR-':
            area = values_sliders['-CONTOUR_AREA-']
            contour_area = validate_contour_area(area)
        elif event_sliders == 'Show':
            img = methods.draw_contour_on_image(image, output_image)
            cv.imshow('Output Image', img)
            cv.waitKey()

        alpha = values_sliders['-ALPHA-']
        beta = values_sliders['-BETA-']
        gamma = values_sliders['-GAMMA-']

        output_image = mark_crack_method(image=image_bw, alpha=alpha, beta=beta, gamma=(gamma/100),
                                         contour_area=contour_area)
        output_image_png = cv.imencode('.png', output_image)[1].tobytes()
        window_im['-IMAGE-'].update(data=output_image_png)





def validate_contour_area(input_string):
    try:
        float(input_string)
    except ValueError:
        sg.popup_error('Invalid input\nPlease Try Again\n\n\nContour Area set to 100', title='Error')
        return 100
    else:
        return float(input_string)


def mark_crack_method(image, alpha, beta, gamma, contour_area):
    _image = methods.filter(image)
    _image = methods.adjust_contrast(image=_image, alpha=alpha, beta=beta)
    _image = methods.gamma_correction(_image, gamma)
    _image = methods.invert_image(_image)
    _image = methods.thresh_otsu(_image)
    contours = methods.draw_contours(_image)
    _image = methods.remove_contours(_image, contours, contour_area)
    return _image


im = select_image()
im = resize(im)
mark_crack(im)
