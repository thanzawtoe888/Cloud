import cv2
import numpy as np

def draw_zigzag(img, start, end, step=25):
    """
    Draw a zig-zag pattern from start to end with main pixel in full color
    and adjacent borders in dimmer color to produce thickness = 3 with gradient.

    Args:
        img (np.array): The image to draw on.
        start (tuple): Starting point (x, y).
        end (tuple): Ending point (x, y).
        step (int): Distance for each segment.
    """
    x, y = start
    direction = 'horizontal'
    main_color = 255
    dim_color = 128
    
    while (x < end[0] and y < end[1]):
        if direction == 'horizontal':
            x_next = x + step
            if x_next > end[0]:
                x_next = end[0]
            # Main 1px thick
            cv2.line(img, (x, y), (x_next, y), main_color, 1)
            # Outer dimmer lines above and below
            if y-1 >= 0:
                cv2.line(img, (x, y-1), (x_next, y-1), dim_color, 1)
            if y+1 < img.shape[0]:
                cv2.line(img, (x, y+1), (x_next, y+1), dim_color, 1)

            x = x_next
            direction = 'vertical'
        else:
            y_next = y + step
            if y_next > end[1]:
                y_next = end[1]
            # Main 1px thick
            cv2.line(img, (x, y), (x, y_next), main_color, 1)
            # Outer dimmer lines left and right
            if x-1 >= 0:
                cv2.line(img, (x-1, y), (x-1, y_next), dim_color, 1)
            if x+1 < img.shape[1]:
                cv2.line(img, (x+1, y), (x+1, y_next), dim_color, 1)

            y = y_next
            direction = 'horizontal'

    return img

# Example usage:
img = np.zeros((200, 200), dtype='uint8')
start = (50, 50)
end = (150, 150)

draw_zigzag(img, start, end, step=25)

cv2.imshow('Zig-Zag Thickness 3 with Fade', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
