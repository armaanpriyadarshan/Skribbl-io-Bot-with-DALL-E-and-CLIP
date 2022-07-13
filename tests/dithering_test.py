from PIL import Image
import itertools
import time
import keyboard
import pydirectinput

pydirectinput.PAUSE = 0

# RGB of each Skribbl color mapped to tuple with pixel coordinates of color on the Skribbl drawing palette
palette = {
    (255, 255, 255): (585, 861),
    (190, 194, 189): (611, 861),
    (239, 21, 10): (633, 861),
    (252, 111, 3): (656, 861),
    (254, 229, 0): (680, 861),
    (2, 205, 3): (704, 861),
    (1, 179, 254): (731, 861),
    (38, 33, 208): (754, 861),
    (164, 1, 184): (777, 861),
    (212, 125, 173): (802, 861),
    (158, 83, 42): (825, 861),
    (1, 0, 1): (585, 885),
    (77, 76, 77): (611, 885),
    (116, 10, 6): (633, 885),
    (197, 56, 0): (656, 885),
    (231, 163, 1): (680, 885),
    (2, 85, 17): (704, 885),
    (1, 85, 159): (731, 885),
    (13, 6, 101): (754, 885),
    (84, 1, 104): (777, 885),
    (164, 85, 117): (802, 885),
    (99, 49, 13): (825, 885)
}

image = Image.open("monkey.png")
resized_image = image.resize((round(0.5 * image.size[0]), round(0.5 * image.size[1])))
palette_image = Image.new('P', (484, 484))
palette_image.putpalette(list(itertools.chain(*palette.keys())))

image_dithered = resized_image.quantize(palette=palette_image, dither=1).convert("RGB")
# image_dithered.show()

x_start, y_start = (700, 300)
width, height = image_dithered.size
pixels = image_dithered.load()

image_colors = {v: k for k, v in dict(image_dithered.getcolors()).items()}
colors_by_frequency = [i[0] for i in sorted(image_colors.items(), key=lambda x: x[1], reverse=True)]

# print(colors_by_frequency)
time.sleep(2)
for color in colors_by_frequency:
    pydirectinput.click(palette[color][0], palette[color][1])
    for y in range(3, height * 3, 3):
        for x in range(3, width * 3, 3):
            if pixels[int(x / 3), int(y / 3)] == color:
                pydirectinput.click(x_start + x, y_start + y)
            if keyboard.is_pressed('q'):
                break
    sleep_time = image_colors[color]/1000
    print(sleep_time)
    time.sleep(sleep_time)
